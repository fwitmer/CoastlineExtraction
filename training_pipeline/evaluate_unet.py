"""
U-Net Model Evaluation & Sept 4 / Sept 6 Coastline Analysis Script

Evaluates trained U-Net / Attention U-Net models on:
1. September 4 and September 6 test tiles (test_data_4_6_sept folder).
2. Computes spatial RMSE metrics vs ground truth coastlines (ground_truth folder).
3. Computes regional RMSE breakdown across 5 coastal regions.
4. Computes pixel-level metrics (Pixel Accuracy, Precision, Recall, F1-Score, IoU) on validation set.

Usage: python evaluate_unet.py
"""

import os
import sys
import glob
import re
import numpy as np
import geopandas as gpd
import rasterio as rio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import skimage.measure
from scipy import ndimage
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, box
import matplotlib.pyplot as plt

# Dynamic linker fix & path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from load_config import load_config, get_training_config, get_augment_tiles_output_folder, get_model_save_path
from train_unet import UNet, AttentionUNet, ClassicUNet, SegmentationDataset

UTM_ZONE_3N = 'EPSG:32603'
EPSILON = 2 ** -16

# ----------------------------
# Line & Raster Smoothing
# ----------------------------
def smooth_linestring(line, sigma=2.0, min_length=30.0):
    """
    Applies 1D Gaussian rolling mean smoothing on LineString coordinates.
    Preserves endpoints and filters out short noisy fragments.
    """
    if line is None or line.is_empty or line.length < min_length:
        return None

    coords = np.array(line.coords)
    if len(coords) < 4:
        return line

    xs, ys = coords[:, 0], coords[:, 1]
    smoothed_xs = ndimage.gaussian_filter1d(xs, sigma=sigma, mode='nearest')
    smoothed_ys = ndimage.gaussian_filter1d(ys, sigma=sigma, mode='nearest')

    smoothed_xs[0], smoothed_xs[-1] = xs[0], xs[-1]
    smoothed_ys[0], smoothed_ys[-1] = ys[0], ys[-1]

    return LineString(np.column_stack([smoothed_xs, smoothed_ys]))

def extract_model_coastline(model, image_path, transform, device, img_size=(256, 256)):
    """
    Runs model inference on full resolution tile using sliding/tiled window or resized pass.
    Extracts smoothed binary water mask & coastline LineStrings.
    """
    model.eval()
    with rio.open(image_path) as src:
        data = src.read()  # (4, H, W)
        dataset_mask = src.dataset_mask() > 0
        h_orig, w_orig = data.shape[1], data.shape[2]
        geo_transform = src.transform

    rgb = data[[2, 1, 0]]  # (3, H, W) RGB
    rgb = (np.clip(rgb.astype(np.float32) / 10000.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))
    pil_img = Image.fromarray(rgb)

    input_t = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_output = model(input_t)
        if raw_output.shape[1] == 1:
            prob_map = torch.sigmoid(raw_output).squeeze().cpu().numpy()
        else:
            prob_map = raw_output.squeeze().cpu().numpy()

    # Resize back to full raster resolution
    prob_img = Image.fromarray((prob_map * 255).astype(np.uint8)).resize((w_orig, h_orig), Image.BILINEAR)
    prob_map_full = np.array(prob_img, dtype=np.float32) / 255.0

    # Probability smoothing & mask generation
    smoothed_prob = ndimage.gaussian_filter(prob_map_full, sigma=1.5)
    water_mask = (smoothed_prob > 0.5) & dataset_mask

    eroded_mask = ndimage.binary_erosion(dataset_mask, iterations=4)
    contours = skimage.measure.find_contours(smoothed_prob, 0.5)

    lines = []
    for contour in contours:
        rows = np.clip(np.round(contour[:, 0]).astype(int), 0, h_orig - 1)
        cols = np.clip(np.round(contour[:, 1]).astype(int), 0, w_orig - 1)

        in_bounds = eroded_mask[rows, cols]
        split_indices = np.where(~in_bounds)[0]
        segments = np.split(contour, split_indices)

        for seg in segments:
            seg_clean = seg[eroded_mask[np.clip(np.round(seg[:, 0]).astype(int), 0, h_orig - 1),
                                        np.clip(np.round(seg[:, 1]).astype(int), 0, w_orig - 1)]]
            if len(seg_clean) >= 2:
                xs, ys = rio.transform.xy(geo_transform, seg_clean[:, 0], seg_clean[:, 1])
                raw_line = LineString(list(zip(xs, ys)))
                smoothed_line = smooth_linestring(raw_line, sigma=2.0, min_length=30.0)
                if smoothed_line is not None:
                    lines.append(smoothed_line)

    return lines, water_mask, geo_transform

def extract_ndwi_coastline(image_path, transform):
    """Extracts baseline NDWI threshold coastline."""
    with rio.open(image_path) as src:
        green = src.read(2).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        dataset_mask = src.dataset_mask() > 0
        h_orig, w_orig = green.shape
        geo_transform = src.transform

    ndwi = (green - nir) / (green + nir + 1e-10)
    smoothed_ndwi = ndimage.gaussian_filter(ndwi, sigma=1.5)
    ndwi_mask = (smoothed_ndwi > 0.0) & dataset_mask

    eroded_mask = ndimage.binary_erosion(dataset_mask, iterations=4)
    contours = skimage.measure.find_contours(smoothed_ndwi, 0.0)

    lines = []
    for contour in contours:
        rows = np.clip(np.round(contour[:, 0]).astype(int), 0, h_orig - 1)
        cols = np.clip(np.round(contour[:, 1]).astype(int), 0, w_orig - 1)

        in_bounds = eroded_mask[rows, cols]
        split_indices = np.where(~in_bounds)[0]
        segments = np.split(contour, split_indices)

        for seg in segments:
            seg_clean = seg[eroded_mask[np.clip(np.round(seg[:, 0]).astype(int), 0, h_orig - 1),
                                        np.clip(np.round(seg[:, 1]).astype(int), 0, w_orig - 1)]]
            if len(seg_clean) >= 2:
                xs, ys = rio.transform.xy(geo_transform, seg_clean[:, 0], seg_clean[:, 1])
                raw_line = LineString(list(zip(xs, ys)))
                smoothed_line = smooth_linestring(raw_line, sigma=2.0, min_length=30.0)
                if smoothed_line is not None:
                    lines.append(smoothed_line)

    return lines, ndwi_mask

# ----------------------------
# RMSE Computation Utilities
# ----------------------------
def calc_rmse(errs):
    errs = np.array(errs)
    if len(errs) == 0:
        return np.nan
    return np.sqrt(np.square(errs).mean())

def find_distances(transects, fst, snd):
    distances = []
    for transect in transects.itertuples():
        t_geom = transect.geometry
        fst_pts = [p for p in fst.geoms if p.distance(t_geom) < EPSILON]
        snd_pts = [p for p in snd.geoms if p.distance(t_geom) < EPSILON]

        if len(fst_pts) == 1 and len(snd_pts) == 1:
            dist = fst_pts[0].distance(snd_pts[0])
            distances.append(dist)
    return distances

def compute_transect_rmse(transects, true_gdf, pred_lines_gdf, river_removal=True):
    if pred_lines_gdf is None or len(pred_lines_gdf) == 0 or true_gdf is None or len(true_gdf) == 0:
        return np.nan, []

    if river_removal:
        removal_ids = [17336, 17335, 17334, 17333, 17332]
        transects = transects[~(transects['TransOrder'].isin(removal_ids))]

    transects = transects.to_crs(UTM_ZONE_3N)
    true_gdf = true_gdf.to_crs(UTM_ZONE_3N)
    pred_lines_gdf = pred_lines_gdf.to_crs(UTM_ZONE_3N)

    geom_true = true_gdf.unary_union.intersection(transects.unary_union)
    geom_pred = pred_lines_gdf.unary_union.intersection(transects.unary_union)

    if geom_true.is_empty or geom_pred.is_empty:
        return np.nan, []

    if not hasattr(geom_true, 'geoms'):
        geom_true = MultiPoint([geom_true]) if isinstance(geom_true, Point) else geom_true
    if not hasattr(geom_pred, 'geoms'):
        geom_pred = MultiPoint([geom_pred]) if isinstance(geom_pred, Point) else geom_pred

    dists = find_distances(transects, geom_true, geom_pred)
    rmse_val = calc_rmse(dists)
    return rmse_val, dists

def compute_regional_rmse(transects, true_gdf, pred_lines_gdf):
    regions = {
        "Western Region (R1)": transects[transects['TransOrder'] >= 17443],
        "Northern Region (R2)": transects[(transects['TransOrder'] < 17443) & (transects['TransOrder'] >= 17394)],
        "Central Region (R3)": transects[(transects['TransOrder'] < 17394) & (transects['TransOrder'] >= 17370)],
        "Town Region (R4)": transects[(transects['TransOrder'] < 17370) & (transects['TransOrder'] >= 17337)],
        "East Region (R5)": transects[transects['TransOrder'] < 17337],
    }

    res = {}
    for r_name, r_transects in regions.items():
        val, _ = compute_transect_rmse(r_transects, true_gdf, pred_lines_gdf)
        res[r_name] = val
    return res

# ----------------------------
# Validation Dataset Evaluation
# ----------------------------
def evaluate_validation_metrics(model, dataloader, device):
    model.eval()
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
    pixel_correct, pixel_total = 0, 0

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Evaluating Val Split"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            if outputs.shape[1] == 1:
                outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            tp_total += ((preds == 1) & (masks == 1)).sum().item()
            fp_total += ((preds == 1) & (masks == 0)).sum().item()
            fn_total += ((preds == 0) & (masks == 1)).sum().item()
            tn_total += ((preds == 0) & (masks == 0)).sum().item()

            pixel_correct += (preds == masks).sum().item()
            pixel_total += masks.numel()

    acc = pixel_correct / pixel_total if pixel_total > 0 else 0
    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    iou = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0

    return {
        "pixel_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "iou": iou,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "tn": tn_total
    }

# ----------------------------
# Plotting & Visualization
# ----------------------------
def save_evaluation_plot(t_path, model_lines, ndwi_lines, planet_ref_gdf, usgs_gdf, hires_gt_gdf, out_plot_path):
    with rio.open(t_path) as src:
        rgb = src.read([3, 2, 1])
        bounds = src.bounds

    rgb_disp = np.zeros_like(rgb, dtype=np.float32)
    for b in range(3):
        band = rgb[b].astype(np.float32)
        valid = band > 0
        if np.any(valid):
            p2, p98 = np.percentile(band[valid], (2, 98))
            rgb_disp[b] = np.clip((band - p2) / (p98 - p2 + 1e-5), 0, 1)

    rgb_disp = np.transpose(rgb_disp, (1, 2, 0))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb_disp, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

    if planet_ref_gdf is not None:
        planet_ref_gdf.to_crs(UTM_ZONE_3N).plot(ax=ax, color='orange', linewidth=2.5, label='Planet Labs Reference (9/9)')
    if hires_gt_gdf is not None:
        hires_gt_gdf.to_crs(UTM_ZONE_3N).plot(ax=ax, color='red', linewidth=2.0, label='Manual Hi-Res GT')
    if usgs_gdf is not None:
        usgs_gdf.to_crs(UTM_ZONE_3N).plot(ax=ax, color='green', linewidth=2.0, label='USGS Coastline')

    if model_lines:
        model_gdf = gpd.GeoDataFrame(geometry=model_lines, crs=UTM_ZONE_3N)
        model_gdf.plot(ax=ax, color='cyan', linewidth=1.8, label='Predicted Coastline (U-Net)')

    if ndwi_lines:
        ndwi_gdf = gpd.GeoDataFrame(geometry=ndwi_lines, crs=UTM_ZONE_3N)
        ndwi_gdf.plot(ax=ax, color='magenta', linewidth=1.5, label='NDWI Coastline')

    ax.set_title(f"Predicted Coastlines vs Ground Truth\n({os.path.basename(t_path)})", fontsize=14, fontweight='bold')
    ax.set_xlim([bounds.left, bounds.right])
    ax.set_ylim([bounds.bottom, bounds.top])
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_plot_path), exist_ok=True)
    plt.savefig(out_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation comparison plot to: {out_plot_path}")

# ----------------------------
# Main Evaluation Function
# ----------------------------
def main():
    config = load_config()
    training_config = get_training_config(config)

    # Set device
    device = training_config.get('device', 'auto')
    if device == 'auto' or 'cuda' in device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths to Ground Truth files (ground_truth directory)
    gt_dir = os.path.join(repo_root, "ground_truth")
    planet_ref_path = os.path.join(gt_dir, "9_9_16_PlanetCoastline.shp")
    hires_gt_path = os.path.join(gt_dir, "2016_HiRes_Final_Coastline.shp")

    # Transects & USGS Coastlines
    transects_path = os.path.join(repo_root, "USGS_Coastlines", "WestChukchi_exposed_STepr_rates", "WestChukchi_exposed_STepr_rates.shp")

    # Load Ground Truth shapefiles if available
    planet_ref_gdf = gpd.read_file(planet_ref_path) if os.path.exists(planet_ref_path) else None
    hires_gt_gdf = gpd.read_file(hires_gt_path) if os.path.exists(hires_gt_path) else None
    usgs_gdf = gpd.read_file(transects_path) if os.path.exists(transects_path) else None

    if usgs_gdf is not None:
        transects_gdf = usgs_gdf[usgs_gdf['BaselineID'] == 117]
    else:
        transects_gdf = None

    # Load Trained U-Net Model
    model_paths = [
        os.path.join(repo_root, "output_models", "best_deep_attn_unet_8epochs.pth"),
        os.path.join(repo_root, "output_models", "best_deep_unet_8epochs.pth"),
        get_model_save_path(config)
    ]

    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print("Warning: No pre-trained model weights found. Skipping spatial inference evaluation.")
    else:
        print(f"Loading model weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        is_attn = any("attn" in key for key in state_dict.keys())
        if is_attn:
            model = AttentionUNet(n_channels=3, n_classes=1)
            print("Loaded Attention U-Net architecture.")
        else:
            model = UNet(n_channels=3, n_classes=1)
            print("Loaded Deep U-Net architecture.")

        model.load_state_dict(state_dict)
        model.to(device)

        # Image preprocessing transform
        image_size = training_config.get('image_size', [256, 256])
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        # Evaluate on September 4 and September 6 test tiles
        test_dir = os.path.join(repo_root, "test_data_4_6_sept")
        test_tiles = [
            os.path.join(test_dir, "sept_4", "files", "369619_2016-09-04_RE2_3A_Analytic_SR_clip.tif"),
            os.path.join(test_dir, "sept_6", "files", "369619_2016-09-06_RE5_3A_Analytic_SR_clip.tif")
        ]

        out_dir = os.path.join(repo_root, "inference_outputs")
        os.makedirs(out_dir, exist_ok=True)

        for tile_path in test_tiles:
            if not os.path.exists(tile_path):
                print(f"Test tile not found: {tile_path}")
                continue

            tile_name = os.path.basename(tile_path)
            print(f"\n==================================================")
            print(f"EVALUATING TEST TILE: {tile_name}")
            print(f"==================================================")

            # Model inference & line extraction
            model_lines, model_mask, geo_transform = extract_model_coastline(model, tile_path, transform, device)
            ndwi_lines, ndwi_mask = extract_ndwi_coastline(tile_path, transform)

            model_gdf = gpd.GeoDataFrame(geometry=model_lines, crs=UTM_ZONE_3N) if model_lines else None
            ndwi_gdf = gpd.GeoDataFrame(geometry=ndwi_lines, crs=UTM_ZONE_3N) if ndwi_lines else None

            # Save predicted shapefile
            if model_gdf is not None:
                shp_out = os.path.join(out_dir, f"{tile_name}_eval_predicted_coastline.shp")
                model_gdf.to_file(shp_out)
                print(f"Saved predicted coastline shapefile to: {shp_out}")

            # Compute Spatial RMSE Metrics
            if transects_gdf is not None:
                rmse_planet, _ = compute_transect_rmse(transects_gdf, planet_ref_gdf, model_gdf)
                rmse_usgs, _ = compute_transect_rmse(transects_gdf, usgs_gdf, model_gdf)
                rmse_hires, _ = compute_transect_rmse(transects_gdf, hires_gt_gdf, model_gdf)

                rmse_ndwi_planet, _ = compute_transect_rmse(transects_gdf, planet_ref_gdf, ndwi_gdf)

                print(f"\n[U-Net Model RMSE Results]")
                print(f"  RMSE vs Planet Labs Ref (9/9): {rmse_planet:.2f} m" if not np.isnan(rmse_planet) else "  RMSE vs Planet Labs Ref: N/A")
                print(f"  RMSE vs USGS Coastlines:      {rmse_usgs:.2f} m" if not np.isnan(rmse_usgs) else "  RMSE vs USGS Coastlines: N/A")
                print(f"  RMSE vs Manual Hi-Res GT:      {rmse_hires:.2f} m" if not np.isnan(rmse_hires) else "  RMSE vs Manual Hi-Res GT: N/A")

                print(f"\n[NDWI Baseline RMSE Results]")
                print(f"  RMSE vs Planet Labs Ref (9/9): {rmse_ndwi_planet:.2f} m" if not np.isnan(rmse_ndwi_planet) else "  RMSE vs Planet Labs Ref: N/A")

                # Regional Breakdown
                if planet_ref_gdf is not None:
                    reg_unet = compute_regional_rmse(transects_gdf, planet_ref_gdf, model_gdf)
                    reg_ndwi = compute_regional_rmse(transects_gdf, planet_ref_gdf, ndwi_gdf)

                    print(f"\n[Regional RMSE Breakdown vs Planet Labs Ref]")
                    for r_name in reg_unet.keys():
                        u_v = f"{reg_unet[r_name]:.2f} m" if not np.isnan(reg_unet[r_name]) else "N/A"
                        n_v = f"{reg_ndwi[r_name]:.2f} m" if not np.isnan(reg_ndwi[r_name]) else "N/A"
                        print(f"  - {r_name}: U-Net={u_v} | NDWI={n_v}")

            # Plot comparison map
            plot_out_path = os.path.join(out_dir, f"{tile_name}_eval_comparison_plot.png")
            save_evaluation_plot(tile_path, model_lines, ndwi_lines, planet_ref_gdf, usgs_gdf, hires_gt_gdf, plot_out_path)

    # Optional: Evaluate dataset split pixel metrics if augment_tiles folder exists
    aug_data_dir = get_augment_tiles_output_folder(config)
    if os.path.exists(aug_data_dir):
        print(f"\n==========================================")
        print(f"EVALUATING VALIDATION DATASET PIXEL METRICS")
        print(f"==========================================")
        val_dataset = SegmentationDataset(aug_data_dir, transform=transform)
        if len(val_dataset) > 0:
            train_split = training_config.get('train_split', 0.8)
            total_sz = len(val_dataset)
            train_sz = int(train_split * total_sz)
            val_sz = total_sz - train_sz

            generator = torch.Generator().manual_seed(42)
            _, val_set = random_split(val_dataset, [train_sz, val_sz], generator=generator)
            val_loader = DataLoader(val_set, batch_size=16, num_workers=4, pin_memory=True)

            val_metrics = evaluate_validation_metrics(model, val_loader, device)
            print(f"Pixel Accuracy:       {val_metrics['pixel_accuracy']:.4%}")
            print(f"Precision (PPV):      {val_metrics['precision']:.4%}")
            print(f"Recall (Sensitivity): {val_metrics['recall']:.4%}")
            print(f"F1-Score (Dice Coeff): {val_metrics['f1_score']:.4%}")
            print(f"Mean IoU (Jaccard):   {val_metrics['iou']:.4%}")

if __name__ == "__main__":
    main()
