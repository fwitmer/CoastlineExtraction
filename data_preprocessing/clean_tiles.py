"""
Clean Tiles Script

This script performs data cleaning/cloud filtering on 512x512 satellite tiles and their corresponding 
UDM2 masks using Configuration 7 (Balanced Sensitive). 
It filters out corrupted or excessively obscured tiles before deep learning model training.

The algorithm uses a hybrid approach:
1. Native UDM2 Cloud/Haze Check (first-pass gatekeeper)
2. Visible Brightness Index Calculation (average of R, G, B)
3. Conditioned Optimization Pool for Otsu Thresholding
4. Safeguarded Adaptive Otsu Cloud/Haze classification with a physical threshold floor
5. Rule-based Custom Shadow and Custom Haze classification
6. Final decision based on an acceptable cloud budget (default <= 15%)
"""

import os
import sys
import glob
import shutil
import json
import numpy as np
import rasterio as rio
from skimage.filters import threshold_otsu

def calculate_tile_quality(rgbn_tile, udm2_tile, floor_threshold=0.20, otsu_mult=0.95, 
                           additional_shadow_thresh=0.03, additional_haze_thresh=0.14, 
                           max_allowed_cloud=0.15):
    """
    Evaluates a tile's quality using the hybrid safeguarded Configuration 7 parameters.

    Parameters:
    -----------
    rgbn_tile : ndarray
        4-Band imagery array of shape (4, H, W) -> Ordered: Blue, Green, Red, NIR
    udm2_tile : ndarray
        8-Band UDM2 array of shape (8, H, W) -> Band indices: 0: Clear, 1: Snow, 2: Shadow, etc.
    floor_threshold : float
        Minimum TOA reflectance floor to safeguard Otsu calculations (default 0.20).
    otsu_mult : float
        Otsu threshold multiplier (default 0.95).
    additional_shadow_thresh : float
        Custom brightness threshold for shadow classification (default 0.03).
    additional_haze_thresh : float
        Custom brightness threshold for haze classification (default 0.14).
    max_allowed_cloud : float
        Strict final cloud fraction threshold allowed for keeping a tile (default 0.15).
    """
    # 1. Isolate valid data pixels (exclude background no-data)
    valid_mask = rgbn_tile[0] > 0
    total_valid_pixels = np.sum(valid_mask)
    if total_valid_pixels == 0:
        return {"status": "REJECT", "reason": "No-data tile", "cloud_fraction": 1.0, "udm_share": 0.0, "otsu_share": 0.0}

    # 2. Extract UDM2 specific layers (0-indexed)
    udm_snow = udm2_tile[1] == 1
    udm_shadow = udm2_tile[2] == 1
    udm_lhaze = udm2_tile[3] == 1
    udm_hhaze = udm2_tile[4] == 1
    udm_cloud = udm2_tile[5] == 1

    # Native cloud cover is just the heavy cloud + light haze/heavy haze
    udm_native_cloud = udm_cloud | udm_lhaze | udm_hhaze

    # First-pass coarse gatekeeper check based on native cloud cover
    udm_cloud_fraction = np.sum(udm_native_cloud & valid_mask) / total_valid_pixels
    if udm_cloud_fraction > 0.20:
        return {
            "status": "REJECT", 
            "reason": "High native UDM cloud cover", 
            "cloud_fraction": float(udm_cloud_fraction), 
            "udm_share": float(udm_cloud_fraction), 
            "otsu_share": 0.0
        }

    # 3. Generate Visible Brightness Index (Mean of R, G, B)
    # Assumes 16-bit Top of Atmosphere reflectance scaled by 10000
    brightness = np.mean(rgbn_tile[0:3], axis=0) / 10000.0

    # Isolate the conditioned optimization pool (valid, snow-free, and heavy cloud-free)
    otsu_search_pool = valid_mask & ~udm_snow & ~udm_cloud

    # 4. Dynamic Fine Check via Safeguarded Otsu
    if np.sum(otsu_search_pool) > 100:
        t_otsu = threshold_otsu(brightness[otsu_search_pool])
        adjusted_threshold = t_otsu * otsu_mult

        # Apply the physical safeguard floor
        if adjusted_threshold >= floor_threshold:
            otsu_clouds = (brightness > adjusted_threshold) & otsu_search_pool
        else:
            otsu_clouds = np.zeros_like(valid_mask, dtype=bool)
    else:
        otsu_clouds = np.zeros_like(valid_mask, dtype=bool)

    # 5. Additional custom detections (Shadows and Haze)
    if additional_shadow_thresh is not None:
        custom_shadow = (brightness < additional_shadow_thresh) & valid_mask & ~udm_cloud & ~otsu_clouds
    else:
        custom_shadow = np.zeros_like(valid_mask, dtype=bool)

    if additional_haze_thresh is not None:
        custom_haze = (brightness > additional_haze_thresh) & valid_mask & ~udm_snow & ~udm_shadow & ~udm_cloud & ~otsu_clouds
    else:
        custom_haze = np.zeros_like(valid_mask, dtype=bool)

    # 6. Integrate masks and compute Final Quality Metric
    final_cloud_mask = udm_cloud | otsu_clouds | custom_haze
    final_cloud_fraction = np.sum(final_cloud_mask & valid_mask) / total_valid_pixels

    # Final decision matrix logic against the acceptable cloud budget
    decision = "KEEP" if final_cloud_fraction <= max_allowed_cloud else "REJECT"
    reason = "KEEP" if decision == "KEEP" else "Excessive final cloud cover (Otsu-refined)"

    return {
        "status": decision,
        "reason": reason,
        "cloud_fraction": float(final_cloud_fraction),
        "udm_share": float(udm_cloud_fraction),
        "otsu_share": float(np.sum(otsu_clouds) / total_valid_pixels)
    }

def clean_directory(input_tif_dir, input_udm_dir, output_tif_dir, output_udm_dir):
    """Runs quality control over directories of tiles and copies the valid ones."""
    os.makedirs(output_tif_dir, exist_ok=True)
    os.makedirs(output_udm_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(input_tif_dir, "*.tif"))
    print(f"Found {len(img_paths)} tiles to evaluate.")

    kept_count = 0
    rejected_count = 0

    for img_path in img_paths:
        filename = os.path.basename(img_path)
        parts = filename.split("_")
        item_id = "_".join(parts[0:4])
        suffix = parts[-1]
        
        # Match UDM naming convention
        mask_filename = f"{item_id}_3B_udm2_clip_mask_{suffix}"
        mask_path = os.path.join(input_udm_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(f"Warning: Corresponding mask file not found: {mask_filename}")
            continue

        try:
            with rio.open(img_path) as src:
                rgbn_tile = src.read()
            with rio.open(mask_path) as src:
                udm2_tile = src.read()

            quality = calculate_tile_quality(rgbn_tile, udm2_tile)
            if quality["status"] == "KEEP":
                shutil.copy2(img_path, os.path.join(output_tif_dir, filename))
                shutil.copy2(mask_path, os.path.join(output_udm_dir, mask_filename))
                kept_count += 1
            else:
                rejected_count += 1
        except Exception as e:
            print(f"Error evaluating tile {filename}: {e}")
            continue

    print(f"Completed! Kept: {kept_count}, Rejected: {rejected_count}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python clean_tiles.py <input_tif_dir> <input_udm_dir> <output_tif_dir> <output_udm_dir>")
    else:
        clean_directory(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
