"""
Segment Anything Model (SAM) Fine-Tuning Pipeline for Coastal Segmentation

Fine-tunes the Segment Anything Model (SAM ViT-B) on coastal land-water boundary tiles.
Freezes the Vision Transformer (ViT-B) image encoder to preserve zero-shot spatial representations,
while training the lightweight mask decoder using combined BCE + Dice loss.

Usage:
    python train_sam.py [--data-dir PATH] [--epochs N] [--batch-size N] [--lr FLOAT]
"""

import os
import glob
import argparse
import numpy as np
import rasterio as rio
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
except ImportError:
    raise ImportError("segment-anything package is required. Install via `pip install segment-anything`.")

class SAMCoastalDataset(Dataset):
    """
    PyTorch Dataset for loading 3-band Planet RGB image tiles and binary water masks
    formatted and resized for SAM long-side (1024x1024) input processing.
    """

    def __init__(self, data_dir, image_size=1024, is_train=True, train_ratio=0.85, seed=42):
        """
        Initializes the SAM Coastal Dataset loader.

        Args:
            data_dir (str): Path to directory containing raster TIFF image tiles and NDWI masks.
            image_size (int, optional): Target long-side dimension for SAM input scaling. Defaults to 1024.
            is_train (bool, optional): Whether to load training split (True) or validation split (False). Defaults to True.
            train_ratio (float, optional): Proportion of data allocated for training. Defaults to 0.85.
            seed (int, optional): Random seed for reproducible dataset splitting. Defaults to 42.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform_sam = ResizeLongestSide(image_size)

        all_imgs = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
        all_imgs = [f for f in all_imgs if "_concatenated_ndwi_mask_" not in os.path.basename(f)]

        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(all_imgs))
        split_idx = int(train_ratio * len(all_imgs))

        if is_train:
            self.image_files = [all_imgs[i] for i in shuffled_indices[:split_idx]]
        else:
            self.image_files = [all_imgs[i] for i in shuffled_indices[split_idx:]]

        print(f"[{'TRAIN' if is_train else 'VAL'}] SAM Coastal Dataset: Loaded {len(self.image_files)} tiles.")

    def __len__(self):
        """
        Returns total number of sample tiles in the dataset split.

        Returns:
            int: Number of tile images.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads and preprocesses a single image-mask tile pair along with bounding box prompt tensor.

        Args:
            idx (int): Sample index.

        Returns:
            dict: Dictionary containing 'image', 'mask', 'box', 'original_size', and 'filename'.
        """
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)

        mask_name = img_name.replace("_clip_", "_concatenated_ndwi_mask_clip_")
        mask_path = os.path.join(self.data_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.data_dir, img_name.replace("_clip_", "_concatenated_ndwi_mask_"))

        with rio.open(img_path) as src:
            img_raw = src.read([3, 2, 1])  # RGB (3, H, W)
            h_orig, w_orig = img_raw.shape[1], img_raw.shape[2]

        with rio.open(mask_path) as src_m:
            mask_raw = (src_m.read(1) > 0).astype(np.uint8)

        # Normalize RGB (0..3000 -> 0..255 uint8)
        rgb_norm = np.clip(img_raw.astype(np.float32) / 3000.0, 0.0, 1.0)
        rgb_uint8 = (np.transpose(rgb_norm, (1, 2, 0)) * 255.0).astype(np.uint8)

        # Resize for SAM long side (1024)
        input_image = self.transform_sam.apply_image(rgb_uint8)
        input_image_torch = torch.as_tensor(input_image, dtype=torch.float32).permute(2, 0, 1)

        # Pad to square 1024x1024
        h_resized, w_resized = input_image_torch.shape[1], input_image_torch.shape[2]
        padh = self.image_size - h_resized
        padw = self.image_size - w_resized
        input_image_padded = F.pad(input_image_torch, (0, padw, 0, padh))

        # Resize ground truth mask to 1024x1024
        mask_resized = cv2.resize(mask_raw, (w_resized, h_resized), interpolation=cv2.INTER_NEAREST)
        mask_padded = np.pad(mask_resized, ((0, padh), (0, padw)), mode='constant', constant_values=0)
        mask_torch = torch.as_tensor(mask_padded, dtype=torch.float32).unsqueeze(0)

        # Generate Bounding Box Prompt [x_min, y_min, x_max, y_max] from GT mask
        water_y, water_x = np.where(mask_padded > 0)
        if len(water_x) > 0 and len(water_y) > 0:
            box_prompt = np.array([np.min(water_x), np.min(water_y), np.max(water_x), np.max(water_y)])
        else:
            box_prompt = np.array([0, 0, w_resized, h_resized])

        box_prompt_torch = torch.as_tensor(box_prompt, dtype=torch.float32)

        return {
            'image': input_image_padded,
            'mask': mask_torch,
            'box': box_prompt_torch,
            'original_size': (h_orig, w_orig),
            'filename': img_name
        }

class SAMDiceBCELoss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Soft Dice Loss for fine-tuning SAM mask predictions.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """
        Initializes SAMDiceBCELoss.

        Args:
            bce_weight (float, optional): Scalar weighting factor for BCE loss component. Defaults to 0.5.
            dice_weight (float, optional): Scalar weighting factor for Dice loss component. Defaults to 0.5.
        """
        super(SAMDiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_masks):
        """
        Calculates weighted sum of BCE loss and soft Dice loss between predicted logits and target masks.

        Args:
            pred_logits (torch.Tensor): Raw unnormalized logit predictions tensor of shape (B, 1, H, W).
            target_masks (torch.Tensor): Ground truth binary mask tensor of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Combined scalar loss value.
        """
        bce_loss = self.bce(pred_logits, target_masks)

        probs = torch.sigmoid(pred_logits)
        intersection = (probs * target_masks).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_masks.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def train_sam(data_dir, checkpoint_path, out_dir, num_epochs=15, batch_size=2, lr=1e-4, device="cuda"):
    """
    Executes SAM mask decoder fine-tuning loop over coastal dataset.

    Args:
        data_dir (str): Path to input dataset folder containing tiles and masks.
        checkpoint_path (str): Path to pretrained SAM ViT-B checkpoint file (.pth).
        out_dir (str): Output directory for saving model checkpoints and training plots.
        num_epochs (int, optional): Total training epochs. Defaults to 15.
        batch_size (int, optional): DataLoader batch size. Defaults to 2.
        lr (float, optional): Learning rate for AdamW optimizer. Defaults to 1e-4.
        device (str, optional): Computation target device ('cuda' or 'cpu'). Defaults to "cuda".

    Returns:
        str: File path to saved best SAM model checkpoint.
    """
    save_model_path = os.path.join(out_dir, "best_sam_coastal_model.pth")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Using device for SAM fine-tuning: {device}")
    print(f"Loading pretrained SAM ViT-B weights from: {checkpoint_path}")
    sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam_model.to(device)

    # Freeze Image Encoder & Prompt Encoder; Train Mask Decoder
    print("Freezing SAM Image Encoder parameters; Unfreezing Mask Decoder for Fine-Tuning...")
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False
    for param in sam_model.mask_decoder.parameters():
        param.requires_grad = True

    train_dataset = SAMCoastalDataset(data_dir, image_size=1024, is_train=True)
    val_dataset = SAMCoastalDataset(data_dir, image_size=1024, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = SAMDiceBCELoss(bce_weight=0.5, dice_weight=0.5)

    best_val_iou = 0.0
    train_losses = []
    val_ious = []

    print("\nStarting SAM Fine-Tuning Pipeline...")
    for epoch in range(num_epochs):
        sam_model.train()
        sam_model.image_encoder.eval()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['box'].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                image_embeddings = sam_model.image_encoder(imgs)

            batch_losses = 0.0
            for i in range(imgs.shape[0]):
                box_i = boxes[i].unsqueeze(0).unsqueeze(1)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_i,
                    masks=None,
                )

                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upsampled_masks = F.interpolate(
                    low_res_masks,
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False
                )

                loss = criterion(upsampled_masks, masks[i].unsqueeze(0))
                batch_losses += loss

            batch_losses /= imgs.shape[0]
            batch_losses.backward()
            optimizer.step()

            running_loss += batch_losses.item()

        epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        # Validation Loop
        sam_model.eval()
        val_iou_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device)
                boxes = batch['box'].to(device)

                image_embeddings = sam_model.image_encoder(imgs)

                for i in range(imgs.shape[0]):
                    box_i = boxes[i].unsqueeze(0).unsqueeze(1)
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None, boxes=box_i, masks=None
                    )
                    low_res_masks, _ = sam_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    upsampled_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
                    probs = torch.sigmoid(upsampled_masks)
                    pred_b = (probs > 0.5).float()
                    gt_b = masks[i].unsqueeze(0)

                    intersection = (pred_b * gt_b).sum()
                    union = pred_b.sum() + gt_b.sum() - intersection
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    val_iou_sum += iou.item()
                    val_count += 1

        mean_val_iou = val_iou_sum / max(1, val_count)

        train_losses.append(epoch_loss)
        val_ious.append(mean_val_iou)

        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | Train Loss: {epoch_loss:.4f} | Val IoU: {mean_val_iou*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            torch.save(sam_model.state_dict(), save_model_path)
            print(f"  --> Saved new best SAM model checkpoint to {save_model_path} (Val IoU: {best_val_iou*100:.2f}%)")

    print(f"\n==================================================================")
    print(f"SAM FINE-TUNING COMPLETE. Best Val IoU: {best_val_iou*100:.2f}%")
    print(f"Model saved to: {save_model_path}")
    print(f"==================================================================\n")

    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-o', label='Train Loss')
    plt.title('SAM Fine-Tuning Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), [i*100 for i in val_ious], 'g-o', label='Val IoU (%)')
    plt.title('SAM Validation IoU (%)')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score (%)')
    plt.grid(True, linestyle=':')
    plt.legend()

    plt.tight_layout()
    curve_path = os.path.join(out_dir, "sam_training_loss_curve.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved SAM training progress plot to: {curve_path}")

    return save_model_path

def main():
    """
    CLI entry point for launching SAM training pipeline.
    """
    parser = argparse.ArgumentParser(description="Fine-tune Segment Anything Model (SAM ViT-B) on coastal dataset")
    parser.add_argument("--data-dir", type=str, default="/home/het22213/myprojs/alaska_proj/new_data_filtered/training_dataset", help="Directory containing dataset tiles")
    parser.add_argument("--checkpoint", type=str, default="/home/het22213/myprojs/alaska_proj/checkpoints/sam_vit_b_01ec64.pth", help="Pretrained SAM ViT-B weights (.pth)")
    parser.add_argument("--out-dir", type=str, default="/home/het22213/myprojs/alaska_proj/new_data_filtered", help="Output directory for saved model and plots")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_sam(args.data_dir, args.checkpoint, args.out_dir, args.epochs, args.batch_size, args.lr, device)

if __name__ == "__main__":
    main()
