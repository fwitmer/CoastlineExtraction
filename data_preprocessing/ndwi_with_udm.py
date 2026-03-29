"""
Module: ndwi_with_udm.py

Description: Integrated NDWI computation with UDM-based cloud masking.
             This module combines cloud masking preprocessing with the
             existing NDWI sliding window approach for improved label quality.

             Workflow:
             1. Apply UDM/QA60 cloud masking to input image
             2. Compute NDWI on cloud-masked reflectance values
             3. Use sliding window + Otsu thresholding for water classification
             4. Generate vectorized coastline shapefiles

Author: Janvis11 (GSoC 2026 Contributor)
Date: 2026-03-28

Issue: #100 - Improving NDWI Label Quality via UDM-Based Masking
"""

import os
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import cv2
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import box

# Local imports
from load_config import load_config, get_image_path, get_shapefile_path
from utils.check_crs import check_crs, crs_match
from utils.spatial_analysis import log_spatial_info
from utils.shapefile_generator import save_concatenated_ndwi_with_shapefile

# Import UDM masking functions
from data_preprocessing.udm_masking import (
    apply_udm_mask,
    read_qa60_band,
    create_cloud_mask_from_qa60,
    create_cloud_mask_from_udm2
)


# =============================================================================
# Configuration
# =============================================================================

# Gaussian blur parameters (same as ndwi_labels.py)
KSIZE_BLUR = (9, 9)
SIGMA_X = 6
SIGMA_Y = 6

# Majority voting threshold
MAJORITY_THRESHOLD = 0.55

# UDM masking options
UDM_CONFIG = {
    'mask_cirrus': True,      # Mask cirrus clouds (Sentinel-2 QA60 bit 11)
    'mask_shadow': True,      # Mask cloud shadows (Planet UDM2)
    'mask_snow': False,       # Mask snow/ice (set True for winter images)
    'min_clear_percentage': 50  # Minimum clear pixels required (%)
}


# =============================================================================
# Main NDWI with UDM Function
# =============================================================================

def get_ndwi_label_with_udm(image_path, points_path, ksize=100, blurring=True,
                            apply_cloud_mask=True, output_dir=None):
    """
    Perform NDWI classification with optional UDM-based cloud masking.

    This function extends the standard NDWI workflow by adding a cloud masking
    preprocessing step. Clouds and shadows are filtered before NDWI computation,
    reducing noise and invalid classifications in the final labels.

    Args:
        image_path (str): Path to input multispectral GeoTIFF
        points_path (str): Path to transect points shapefile
        ksize (int, optional): Sliding window size in pixels. Defaults to 100.
        blurring (bool, optional): Apply Gaussian blur. Defaults to True.
        apply_cloud_mask (bool, optional): Apply UDM cloud masking. Defaults to True.
        output_dir (str, optional): Output directory. Defaults to 'result_ndwi_udm'

    Returns:
        dict: Results containing NDWI arrays, cloud mask, and output paths
    """
    if output_dir is None:
        output_dir = "result_ndwi_udm"

    os.makedirs(output_dir, exist_ok=True)

    # Validate inputs
    check_crs(image_path, verbose=True)
    check_crs(points_path, verbose=True)

    # =========================================================================
    # Step 1: Apply UDM Cloud Masking (if enabled)
    # =========================================================================

    cloud_mask = None
    spectral_data = None

    if apply_cloud_mask:
        print("=" * 60)
        print("STEP 1: Applying UDM-based cloud masking")
        print("=" * 60)

        udm_result = apply_udm_mask(
            image_path,
            output_dir=output_dir,
            mask_cirrus=UDM_CONFIG['mask_cirrus'],
            mask_shadow=UDM_CONFIG['mask_shadow'],
            mask_snow=UDM_CONFIG['mask_snow'],
            visualize=True
        )

        if udm_result:
            cloud_mask = udm_result['cloud_mask']
            spectral_data = udm_result['masked_image']

            # Check if enough clear pixels
            if udm_result['clear_percentage'] < UDM_CONFIG['min_clear_percentage']:
                print(f"WARNING: Only {udm_result['clear_percentage']:.1f}% clear pixels.")
                print("         Results may be unreliable due to cloud cover.")

            # Use masked image path for subsequent processing
            processed_image_path = udm_result['masked_tiff']
        else:
            print("Cloud masking skipped (no QA60/UDM band found)")
            processed_image_path = image_path
    else:
        processed_image_path = image_path

    # =========================================================================
    # Step 2: Read spectral bands and compute NDWI
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 2: Computing NDWI")
    print("=" * 60)

    with rio.open(processed_image_path) as src:
        # Determine green and NIR band indices
        # For 4-band (RGBN): Green=2, NIR=4
        # For 5-band (RGBN+mask): Green=2, NIR=4 or 5

        green_band = 2
        nir_band = src.count if src.count >= 4 else 4

        # Read bands (use pre-masked data if available)
        if spectral_data is not None:
            green = spectral_data[green_band - 1]
            nir = spectral_data[nir_band - 1]
        else:
            green = src.read(green_band).astype(np.float32)
            nir = src.read(min(nir_band, src.count)).astype(np.float32)

        # Compute NDWI
        np.seterr(divide='ignore', invalid='ignore')
        ndwi = (green - nir) / (green + nir)
        ndwi[np.isnan(ndwi)] = 0

        ndwi_profile = src.profile.copy()

        # Apply Gaussian blur
        if blurring:
            print("Applying Gaussian blur...")
            ndwi = cv2.GaussianBlur(ndwi, KSIZE_BLUR, SIGMA_X, SIGMA_Y)

        # Initialize output arrays
        label = np.zeros((src.height, src.width), dtype=np.uint8)
        buffer_numbers = np.zeros((src.height, src.width), dtype=np.uint8)
        water_count = np.zeros((src.height, src.width), dtype=np.uint8)

        src_crs = src.crs
        pixel_size = abs(src.transform[0])
        raster_bounds = src.bounds

    print(f"NDWI range: [{ndwi.min():.3f}, {ndwi.max():.3f}]")

    # =========================================================================
    # Step 3: Sliding Window Analysis
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 3: Sliding window classification")
    print("=" * 60)

    # Prepare points
    points_shp = gpd.read_file(points_path)
    points_geom = points_shp.geometry

    if points_shp.crs != src_crs:
        print(f"Reprojecting points from {points_shp.crs} to {src_crs}")
        points_geom = points_geom.to_crs(src_crs)

    # Save reprojected points
    reprojected_points_path = os.path.join(output_dir, "reprojected_points.shp")
    gpd.GeoDataFrame(geometry=points_geom).to_file(reprojected_points_path)

    # Validate CRS
    if not crs_match(processed_image_path, reprojected_points_path):
        raise ValueError("CRS mismatch after reprojection!")

    # Log spatial info
    log_spatial_info(raster_bounds, points_geom)

    # Process each point
    otsu_thresholds = []
    skipped = 0

    for multipoint in points_geom:
        for point in multipoint.geoms:
            buffer = point.buffer(ksize * pixel_size, cap_style=3)

            ndwi_profile.update(count=1, nodata=0, dtype=rio.float32)

            with MemoryFile() as memfile:
                with memfile.open(**ndwi_profile) as mem_data:
                    mem_data.write_band(1, ndwi)

                with memfile.open() as dataset:
                    raster_bounds_geom = box(*dataset.bounds)

                    if not buffer.intersects(raster_bounds_geom):
                        skipped += 1
                        continue

                    out_image, out_transform = mask(
                        dataset, shapes=[buffer], nodata=-1, crop=False
                    )
                    out_image = out_image[0]

                    out_image_clipped, _ = mask(
                        dataset, shapes=[buffer], nodata=-1, crop=True
                    )
                    out_image_clipped = out_image_clipped[0]

                    # Skip small windows
                    if out_image_clipped.shape[0] < 200 or out_image_clipped.shape[1] < 200:
                        skipped += 1
                        continue

                    # Convert to 8-bit for Otsu
                    out_image_8bit = ((out_image * 127) + 128).astype(np.uint8)
                    out_image_clipped_8bit = ((out_image_clipped * 127) + 128).astype(np.uint8)

                    # Otsu thresholding
                    threshold, _ = cv2.threshold(
                        out_image_clipped_8bit, 0, 1,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                    # Validate threshold
                    if threshold == 0.0 or threshold == 1.0:
                        skipped += 1
                        continue

                    otsu_thresholds.append(threshold)

                    # Apply threshold to full window
                    threshold_window = np.where(out_image_8bit >= threshold, 1, 0).astype(np.uint8)
                    label = label | threshold_window
                    water_count = water_count + threshold_window

                    # Update buffer count
                    mask_array = np.where(out_image_8bit != -1, 1, 0).astype(np.uint8)
                    buffer_numbers = buffer_numbers + mask_array

    print(f"Valid windows processed: {len(otsu_thresholds)}")
    print(f"Windows skipped: {skipped}")

    # =========================================================================
    # Step 4: Combine Results
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 4: Combining classification results")
    print("=" * 60)

    # Majority voting
    label_majority = np.where(
        water_count > (buffer_numbers * MAJORITY_THRESHOLD), 1, 0
    )

    # Global threshold
    if otsu_thresholds:
        mean_threshold = np.mean(otsu_thresholds) + 10
    else:
        mean_threshold = 128

    ndwi_8bit = ((ndwi * 127) + 128).astype(np.uint8)
    ndwi_classified = np.where(ndwi_8bit >= mean_threshold, 1, 0)

    # Combine local and global classification
    ndwi_concatenated = ndwi_classified.copy()
    sliding_windows = np.where(buffer_numbers > 0, 1, 0)
    water_areas = np.where(label == 1, 1, 0)
    ndwi_concatenated = np.where(
        (sliding_windows == 1) & (water_areas == 1), 1, ndwi_concatenated
    )

    # Print statistics
    print(f"\nClassification Statistics:")
    print(f"  Mean threshold (8-bit): {mean_threshold:.1f}")
    print(f"  Water pixels (NDWI classified): {np.sum(ndwi_classified):,}")
    print(f"  Water pixels (majority voting): {np.sum(label_majority):,}")
    print(f"  Water pixels (concatenated): {np.sum(ndwi_concatenated):,}")

    if apply_cloud_mask and cloud_mask is not None:
        cloudy_water = np.sum(ndwi_classified & ~cloud_mask)
        print(f"  Water pixels in cloudy areas: {cloudy_water:,}")

    # =========================================================================
    # Step 5: Save Outputs
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 5: Saving outputs")
    print("=" * 60)

    # Save shapefile
    try:
        save_concatenated_ndwi_with_shapefile(
            ndwi_concatenated, ndwi_profile, processed_image_path, output_dir
        )
    except Exception as e:
        print(f"Warning: Shapefile generation failed: {e}")

    # Save GeoTIFF
    base_name = os.path.splitext(os.path.basename(processed_image_path))[0]
    tiff_path = os.path.join(output_dir, f"{base_name}_ndwi_udm.tif")

    ndwi_profile.update(count=1, dtype='uint8', nodata=255)
    with rio.open(tiff_path, 'w', **ndwi_profile) as dst:
        dst.write(ndwi_concatenated.astype(np.uint8), 1)

    print(f"Saved: {tiff_path}")

    # Save plots
    save_comparison_plots(
        ndwi, ndwi_classified, label_majority, ndwi_concatenated,
        cloud_mask, output_dir, base_name
    )

    return {
        'ndwi': ndwi,
        'ndwi_classified': ndwi_classified,
        'label_majority': label_majority,
        'ndwi_concatenated': ndwi_concatenated,
        'cloud_mask': cloud_mask,
        'output_tiff': tiff_path,
        'thresholds': otsu_thresholds
    }


def save_comparison_plots(ndwi, ndwi_classified, label_majority,
                         ndwi_concatenated, cloud_mask, output_dir, base_name):
    """
    Save comparison plots showing the effect of UDM masking.

    Args:
        ndwi (np.ndarray): NDWI values
        ndwi_classified (np.ndarray): Binary classification
        label_majority (np.ndarray): Majority voting result
        ndwi_concatenated (np.ndarray): Combined result
        cloud_mask (np.ndarray): Cloud mask (None if not applied)
        output_dir (str): Output directory
        base_name (str): Base name for files
    """
    if cloud_mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Row 1
        axes[0, 0].imshow(ndwi, cmap='RdYlBu')
        axes[0, 0].set_title('NDWI Values')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cloud_mask, cmap='gray')
        axes[0, 1].set_title('Cloud Mask (White=Clear)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(ndwi_classified)
        axes[0, 2].set_title('NDWI Classified (Global Threshold)')
        axes[0, 2].axis('off')

        # Row 2
        axes[1, 0].imshow(label_majority)
        axes[1, 0].set_title('Majority Voting')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(ndwi_concatenated)
        axes[1, 1].set_title('Final Combined Result')
        axes[1, 1].axis('off')

        # Difference plot
        diff = np.abs(ndwi_classified.astype(float) - label_majority.astype(float))
        im = axes[1, 2].imshow(diff, cmap='Reds')
        axes[1, 2].set_title('Difference (Local vs Global)')
        axes[1, 2].axis('off')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].imshow(ndwi, cmap='RdYlBu')
        axes[0, 0].set_title('NDWI Values')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(ndwi_classified)
        axes[0, 1].set_title('NDWI Classified')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(label_majority)
        axes[1, 0].set_title('Majority Voting')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(ndwi_concatenated)
        axes[1, 1].set_title('Final Result')
        axes[1, 1].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{base_name}_ndwi_udm_comparison.png"), dpi=150)
    plt.close()

    print(f"Saved visualization: {os.path.join(output_dir, f'{base_name}_ndwi_udm_comparison.png')}")


# =============================================================================
# Comparison Function: With vs Without UDM
# =============================================================================

def compare_ndwi_with_and_without_udm(image_path, points_path, ksize=100):
    """
    Run NDWI classification with and without UDM masking for comparison.

    This function helps evaluate the improvement from UDM-based cloud masking
    by running both versions and comparing the results.

    Args:
        image_path (str): Input image path
        points_path (str): Transect points shapefile
        ksize (int): Sliding window size

    Returns:
        dict: Comparison results
    """
    print("\n" + "=" * 70)
    print("COMPARISON: NDWI with UDM vs NDWI without UDM")
    print("=" * 70)

    # Without UDM
    print("\n--- Running NDWI WITHOUT cloud masking ---")
    result_no_udm = get_ndwi_label_with_udm(
        image_path, points_path, ksize=ksize,
        apply_cloud_mask=False,
        output_dir="result_ndwi_comparison/no_udm"
    )

    # With UDM
    print("\n--- Running NDWI WITH cloud masking ---")
    result_with_udm = get_ndwi_label_with_udm(
        image_path, points_path, ksize=ksize,
        apply_cloud_mask=True,
        output_dir="result_ndwi_comparison/with_udm"
    )

    # Compare statistics
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    if result_no_udm and result_with_udm:
        print(f"\nWater pixels detected:")
        print(f"  Without UDM: {np.sum(result_no_udm['ndwi_concatenated']):,}")
        print(f"  With UDM:    {np.sum(result_with_udm['ndwi_concatenated']):,}")

        print(f"\nOtsu thresholds (mean):")
        if result_no_udm['thresholds']:
            print(f"  Without UDM: {np.mean(result_no_udm['thresholds']):.1f}")
        if result_with_udm['thresholds']:
            print(f"  With UDM:    {np.mean(result_with_udm['thresholds']):.1f}")

        if result_with_udm['cloud_mask'] is not None:
            clear_pct = (np.sum(result_with_udm['cloud_mask']) /
                        result_with_udm['cloud_mask'].size) * 100
            print(f"\nCloud cover: {100-clear_pct:.1f}%")

    return {
        'without_udm': result_no_udm,
        'with_udm': result_with_udm
    }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    config = load_config()

    # Get paths from config
    try:
        image_path = get_image_path(config, 0)
        points_path = get_shapefile_path(config, 0)

        print(f"Image: {image_path}")
        print(f"Points: {points_path}")

        # Run with UDM masking
        result = get_ndwi_label_with_udm(
            image_path, points_path,
            apply_cloud_mask=True,
            output_dir="result_ndwi_udm"
        )

        print("\nProcessing complete!")
        print(f"Output: {result['output_tiff']}")

    except Exception as e:
        print(f"Error: {e}")
        print("Ensure config.json is properly configured")
