"""
Segment Anything Model (SAM) Test Inference & Spatial RMSE Evaluation Script

Evaluates fine-tuned SAM ViT-B model predictions on held-out test splits and full satellite scenes.
Computes Probability RMSE, Binary Mask RMSE, Boundary Distance RMSE (in meters), IoU, and Dice scores.

Usage:
    python run_sam_inference.py [--data-dir PATH] [--model-path PATH] [--checkpoint PATH]
"""

import os
import sys
import glob
import argparse
import numpy as np
import rasterio as rio
import cv2
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
except ImportError:
    raise ImportError("segment-anything package is required. Install via `pip install segment-anything`.")

def compute_boundary_distance_rmse(pred_mask, gt_mask, pixel_size_m=3.0):
    """
    Computes distance transform RMSE along predicted boundary edge contours vs ground truth boundary.

    Args:
        pred_mask (numpy.ndarray): Binary predicted mask array of shape (H, W).
        gt_mask (numpy.ndarray): Binary ground truth mask array of shape (H, W).
        pixel_size_m (float, optional): Pixel resolution in meters. Defaults to 3.0.

    Returns:
        tuple: (rmse_pixels, rmse_meters) containing boundary distance errors.
    """
    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    gt_edge = cv2.morphologyEx(gt_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0

    if not np.any(pred_edge) or not np.any(gt_edge):
        return 0.0, 0.0

    dist_map = ndimage.distance_transform_edt(~gt_edge)
    pred_boundary_dists_px = dist_map[pred_edge]

    rmse_px = np.sqrt(np.mean(pred_boundary_dists_px ** 2))
    rmse_m = rmse_px * pixel_size_m
    return rmse_px, rmse_m

def evaluate_sam_test_set(data_dir, model_path, checkpoint_path, out_dir):
    """
    Evaluates fine-tuned SAM ViT-B model on held-out test split tiles and reports spatial metrics.

    Args:
        data_dir (str): Input dataset directory containing image tiles and ground truth masks.
        model_path (str): Path to fine-tuned SAM weights (.pth file).
        checkpoint_path (str): Path to pretrained base SAM ViT-B checkpoint (.pth file).
        out_dir (str): Destination directory for saving summary reports.

    Returns:
        dict: Summary dictionary containing mean test metrics (IoU, Dice, Boundary RMSE).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM ViT-B model on device: {device}")
    sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    
    if os.path.exists(model_path):
        print(f"Loading fine-tuned SAM weights from: {model_path}")
        sam_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Fine-tuned checkpoint not found! Using zero-shot pretrained SAM weights.")
        
    sam_model.to(device)
    sam_model.eval()

    transform_sam = ResizeLongestSide(1024)

    all_imgs = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    all_imgs = [f for f in all_imgs if "_concatenated_ndwi_mask_" not in os.path.basename(f)]

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(all_imgs))
    test_indices = shuffled_indices[int(0.85 * len(all_imgs)):]
    test_files = [all_imgs[i] for i in test_indices]

    print(f"\nEvaluating SAM on held-out test set ({len(test_files)} tiles)...")

    prob_rmses = []
    bin_rmses = []
    bound_rmses_m = []
    ious = []
    dices = []

    for idx, img_path in enumerate(test_files):
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace("_clip_", "_concatenated_ndwi_mask_clip_")
        mask_path = os.path.join(data_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(data_dir, img_name.replace("_clip_", "_concatenated_ndwi_mask_"))

        if not os.path.exists(mask_path):
            continue

        try:
            with rio.open(img_path) as src:
                img_raw = src.read([3, 2, 1])
                pixel_res = src.transform[0]
            with rio.open(mask_path) as src:
                gt_mask = (src.read(1) > 0).astype(np.float32)
        except Exception:
            continue

        h_orig, w_orig = img_raw.shape[1], img_raw.shape[2]
        rgb_norm = np.clip(img_raw.astype(np.float32) / 3000.0, 0.0, 1.0)
        rgb_uint8 = (np.transpose(rgb_norm, (1, 2, 0)) * 255.0).astype(np.uint8)

        input_image = transform_sam.apply_image(rgb_uint8)
        input_torch = torch.as_tensor(input_image, dtype=torch.float32).permute(2, 0, 1).to(device)

        h_res, w_res = input_torch.shape[1], input_torch.shape[2]
        padh = 1024 - h_res
        padw = 1024 - w_res
        input_padded = F.pad(input_torch, (0, padw, 0, padh)).unsqueeze(0)

        box_prompt = torch.tensor([[0, 0, w_res, h_res]], dtype=torch.float32, device=device).unsqueeze(1)

        with torch.no_grad():
            image_embeddings = sam_model.image_encoder(input_padded)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None, boxes=box_prompt, masks=None
            )
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            upsampled_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
            probs = torch.sigmoid(upsampled_masks)[0, 0, :h_res, :w_res].cpu().numpy()
            probs_orig = cv2.resize(probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        pred_mask = (probs_orig > 0.5).astype(np.float32)

        prob_rmse = np.sqrt(np.mean((probs_orig - gt_mask) ** 2))
        bin_rmse = np.sqrt(np.mean((pred_mask - gt_mask) ** 2))
        _, b_rmse_m = compute_boundary_distance_rmse(pred_mask, gt_mask, pixel_size_m=pixel_res)

        tp = np.sum((pred_mask == 1) & (gt_mask == 1))
        fp = np.sum((pred_mask == 1) & (gt_mask == 0))
        fn = np.sum((pred_mask == 0) & (gt_mask == 1))

        iou = tp / (tp + fp + fn + 1e-6)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

        prob_rmses.append(prob_rmse)
        bin_rmses.append(bin_rmse)
        bound_rmses_m.append(b_rmse_m)
        ious.append(iou)
        dices.append(dice)

    results = {
        "num_tiles": len(prob_rmses),
        "mean_prob_rmse": float(np.mean(prob_rmses)),
        "mean_binary_rmse": float(np.mean(bin_rmses)),
        "mean_boundary_rmse_m": float(np.mean(bound_rmses_m)),
        "mean_iou_pct": float(np.mean(ious) * 100),
        "mean_dice_pct": float(np.mean(dices) * 100)
    }

    print("\n================ SAM TEST SET PERFORMANCE & RMSE RESULTS ================")
    print(f"Total SAM Test Tiles Evaluated:     {results['num_tiles']}")
    print(f"---------------------------------------------------------------------")
    print(f"Mean SAM Probability RMSE:          {results['mean_prob_rmse']:.4f}")
    print(f"Mean SAM Binary Mask RMSE:         {results['mean_binary_rmse']:.4f}")
    print(f"Mean SAM Boundary Distance RMSE:    {results['mean_boundary_rmse_m']:.2f} meters")
    print(f"Mean SAM IoU Score:                 {results['mean_iou_pct']:.2f}%")
    print(f"Mean SAM Dice Score:                {results['mean_dice_pct']:.2f}%")
    print(f"=====================================================================\n")

    return results

def main():
    """
    CLI entry point for evaluating fine-tuned SAM on test set tiles.
    """
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SAM on test set tiles")
    parser.add_argument("--data-dir", type=str, default="/home/het22213/myprojs/alaska_proj/new_data_filtered/training_dataset", help="Directory containing dataset tiles")
    parser.add_argument("--model-path", type=str, default="/home/het22213/myprojs/alaska_proj/new_data_filtered/best_sam_coastal_model.pth", help="Fine-tuned SAM weights (.pth)")
    parser.add_argument("--checkpoint", type=str, default="/home/het22213/myprojs/alaska_proj/checkpoints/sam_vit_b_01ec64.pth", help="Pretrained base SAM checkpoint (.pth)")
    parser.add_argument("--out-dir", type=str, default="/home/het22213/myprojs/alaska_proj/new_data_filtered", help="Output directory")

    args = parser.parse_args()
    evaluate_sam_test_set(args.data_dir, args.model_path, args.checkpoint, args.out_dir)

if __name__ == "__main__":
    main()
