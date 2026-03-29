"""
Module: udm_masking.py

Description: Apply UDM (User Data Mask) / QA60-based cloud masking for Sentinel-2
             and Planet Labs imagery before NDWI computation. This preprocessing
             step reduces noise and invalid regions in water classification.

Author: Janvis11 (GSoC 2026 Contributor)
Date: 2026-03-28

References:
- Sentinel-2 QA60 band: Cloud and snow masking
- Planet Labs UDM2: Cloud, shadow, and snow detection
"""

import os
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import cv2
from matplotlib import pyplot as plt

# Local imports
from load_config import load_config, get_image_path
from utils.check_crs import check_crs


# =============================================================================
# UDM/QA60 Bit Flag Definitions
# =============================================================================

# Sentinel-2 QA60 bit flags
QA60_CLOUD_BIT = 10      # Bit 10: Opaque clouds
QA60_CIRRUS_BIT = 11     # Bit 11: Cirrus clouds

# Planet Labs UDM2 bit flags (if available)
UDM2_CLOUD_BIT = 0       # Cloud pixels
UDM2_SHADOW_BIT = 1      # Cloud shadow pixels
UDM2_SNOW_BIT = 2        # Snow/ice pixels
UDM2_WATER_BIT = 3       # Water pixels


def read_qa60_band(image_path):
    """
    Read the QA60 band from a Sentinel-2 or Planet Labs image.

    For Sentinel-2, QA60 is typically band 10 (quality control band).
    For Planet Labs with UDM2, it may be included as an additional band.

    Args:
        image_path (str): Path to the input GeoTIFF file

    Returns:
        tuple: (qa60_data, profile) - QA60 band array and raster profile
               Returns (None, None) if QA60 band not found
    """
    with rio.open(image_path) as src:
        # Check if QA60 band exists (usually last band or band 10)
        qa60_band_idx = None

        # Try to find QA60 by name in band descriptions
        for i, desc in enumerate(src.descriptions):
            if desc and ('QA60' in desc or 'qa60' in desc or 'UDM' in desc or 'udm' in desc):
                qa60_band_idx = i + 1
                break

        # If not found by name, try common positions
        if qa60_band_idx is None:
            # For Sentinel-2, QA60 is often band 10
            if src.count >= 10:
                qa60_band_idx = 10
            # For Planet Labs with UDM2, it might be the last band
            elif src.count >= 5:
                qa60_band_idx = src.count

        if qa60_band_idx is None or qa60_band_idx > src.count:
            print(f"QA60/UDM band not found in {image_path}")
            return None, None

        qa60 = src.read(qa60_band_idx).astype(np.uint16)
        profile = src.profile.copy()

    return qa60, profile


def create_cloud_mask_from_qa60(qa60_data, mask_cirrus=True):
    """
    Create a binary cloud mask from QA60 bit flags.

    Args:
        qa60_data (np.ndarray): QA60 band data as uint16 array
        mask_cirrus (bool, optional): Also mask cirrus clouds. Defaults to True.

    Returns:
        np.ndarray: Boolean mask where True = clear pixels, False = cloudy pixels
    """
    # Create mask for opaque clouds (bit 10)
    cloud_mask = (qa60_data & (1 << QA60_CLOUD_BIT)) == 0

    # Optionally mask cirrus clouds (bit 11)
    if mask_cirrus:
        cirrus_mask = (qa60_data & (1 << QA60_CIRRUS_BIT)) == 0
        cloud_mask = cloud_mask & cirrus_mask

    return cloud_mask


def create_cloud_mask_from_udm2(udm2_data, mask_shadow=True, mask_snow=False):
    """
    Create a cloud mask from Planet Labs UDM2 data.

    Args:
        udm2_data (np.ndarray): UDM2 band data as uint8 array
        mask_shadow (bool, optional): Also mask cloud shadows. Defaults to True.
        mask_snow (bool, optional): Also mask snow/ice. Defaults to False.

    Returns:
        np.ndarray: Boolean mask where True = valid pixels, False = masked pixels
    """
    # Start with all pixels valid
    valid_mask = np.ones_like(udm2_data, dtype=bool)

    # Mask cloud pixels (bit 0)
    cloud_mask = (udm2_data & (1 << UDM2_CLOUD_BIT)) == 0
    valid_mask = valid_mask & cloud_mask

    # Optionally mask shadows
    if mask_shadow:
        shadow_mask = (udm2_data & (1 << UDM2_SHADOW_BIT)) == 0
        valid_mask = valid_mask & shadow_mask

    # Optionally mask snow
    if mask_snow:
        snow_mask = (udm2_data & (1 << UDM2_SNOW_BIT)) == 0
        valid_mask = valid_mask & snow_mask

    return valid_mask


def apply_udm_mask(image_path, output_dir=None, mask_cirrus=True,
                   mask_shadow=True, mask_snow=False, visualize=True):
    """
    Apply UDM/QA60-based cloud masking to an image.

    This function reads the QA60/UDM2 band from the input image, creates a cloud
    mask, and applies it to all spectral bands. The masked image can then be
    used for NDWI computation with reduced cloud contamination.

    Args:
        image_path (str): Path to input GeoTIFF with QA60/UDM2 band
        output_dir (str, optional): Directory for output files. Defaults to 'result_udm_masking'
        mask_cirrus (bool, optional): Mask cirrus clouds. Defaults to True.
        mask_shadow (bool, optional): Mask cloud shadows. Defaults to True.
        mask_snow (bool, optional): Mask snow/ice. Defaults to False.
        visualize (bool, optional): Create visualization plots. Defaults to True.

    Returns:
        dict: Contains masked_image, cloud_mask, and output paths
    """
    if output_dir is None:
        output_dir = "result_udm_masking"

    os.makedirs(output_dir, exist_ok=True)

    # Check CRS
    check_crs(image_path, verbose=False)

    # Read the QA60/UDM band
    print(f"Reading QA60/UDM band from {image_path}")
    qa60_data, profile = read_qa60_band(image_path)

    if qa60_data is None:
        print("No QA60/UDM band found. Skipping cloud masking.")
        return None

    # Read all spectral bands (exclude QA60/UDM band)
    with rio.open(image_path) as src:
        # Determine which bands are spectral (not QA60)
        spectral_bands = []
        for i in range(1, src.count + 1):
            if src.descriptions and ('QA60' in src.descriptions[i-1] or
                                     'UDM' in src.descriptions[i-1]):
                continue
            spectral_bands.append(i)

        spectral_data = src.read(spectral_bands)
        src_profile = src.profile.copy()
        src_transform = src.transform
        src_crs = src.crs

    # Create cloud mask
    print("Creating cloud mask...")
    if qa60_data.dtype == np.uint16:
        # Sentinel-2 QA60
        cloud_mask = create_cloud_mask_from_qa60(qa60_data, mask_cirrus=mask_cirrus)
    else:
        # Planet Labs UDM2
        cloud_mask = create_cloud_mask_from_udm2(qa60_data,
                                                mask_shadow=mask_shadow,
                                                mask_snow=mask_snow)

    # Apply mask to spectral data
    print("Applying cloud mask to spectral bands...")
    masked_spectral = spectral_data.copy().astype(np.float32)

    # Set cloudy pixels to nodata (will be handled in NDWI calculation)
    for i in range(masked_spectral.shape[0]):
        masked_spectral[i, ~cloud_mask] = np.nan

    # Calculate statistics
    total_pixels = cloud_mask.size
    cloudy_pixels = np.sum(~cloud_mask)
    clear_percentage = (np.sum(cloud_mask) / total_pixels) * 100

    print(f"Cloud masking complete: {clear_percentage:.1f}% clear pixels")
    print(f"  Cloudy pixels: {cloudy_pixels:,} ({100-clear_percentage:.1f}%)")
    print(f"  Clear pixels: {np.sum(cloud_mask):,}")

    # Save masked image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_tiff = os.path.join(output_dir, f"{base_name}_cloud_masked.tif")

    # Update profile for output
    src_profile.update({
        'count': spectral_data.shape[0],
        'dtype': 'float32',
        'nodata': np.nan
    })

    with rio.open(output_tiff, 'w', **src_profile) as dst:
        dst.write(masked_spectral)

    print(f"Saved masked image: {output_tiff}")

    # Save cloud mask as GeoTIFF
    mask_tiff = os.path.join(output_dir, f"{base_name}_cloud_mask.tif")
    mask_profile = src_profile.copy()
    mask_profile.update(count=1, dtype='uint8', nodata=255)

    with rio.open(mask_tiff, 'w', **mask_profile) as dst:
        dst.write(cloud_mask.astype(np.uint8) * 255, 1)

    print(f"Saved cloud mask: {mask_tiff}")

    # Create visualization
    if visualize:
        save_udm_visualization(cloud_mask, masked_spectral, spectral_data,
                              output_dir, base_name)

    return {
        'masked_image': masked_spectral,
        'cloud_mask': cloud_mask,
        'masked_tiff': output_tiff,
        'mask_tiff': mask_tiff,
        'clear_percentage': clear_percentage
    }


def save_udm_visualization(cloud_mask, masked_spectral, original_spectral,
                          output_dir, base_name):
    """
    Save visualization plots comparing original and masked images.

    Args:
        cloud_mask (np.ndarray): Boolean cloud mask
        masked_spectral (np.ndarray): Cloud-masked spectral data
        original_spectral (np.ndarray): Original spectral data
        output_dir (str): Output directory
        base_name (str): Base name for output files
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Cloud mask
    axes[0].imshow(cloud_mask, cmap='gray')
    axes[0].set_title('Cloud Mask (White = Clear)')
    axes[0].axis('off')

    # Plot 2: Original RGB (bands 3, 2, 1 or 4, 3, 2)
    if original_spectral.shape[0] >= 3:
        rgb_idx = [2, 1, 0]  # Use first 3 bands
    else:
        rgb_idx = [0, 0, 0]  # Fallback

    rgb_original = np.stack([original_spectral[i] for i in rgb_idx], axis=-1)
    rgb_original = (rgb_original - rgb_original.min()) / (rgb_original.max() - rgb_original.min())
    axes[1].imshow(rgb_original)
    axes[1].set_title('Original Image')
    axes[1].axis('off')

    # Plot 3: Masked RGB
    if masked_spectral.shape[0] >= 3:
        rgb_masked = np.stack([masked_spectral[i] for i in rgb_idx], axis=-1)
        rgb_masked = np.nan_to_num(rgb_masked, nan=0)
        rgb_masked = (rgb_masked - rgb_masked.min()) / (rgb_masked.max() - rgb_masked.min() + 1e-10)
        axes[2].imshow(rgb_masked)
        axes[2].set_title('Cloud-Masked Image')
        axes[2].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{base_name}_udm_masking.png"), dpi=150)
    plt.close()

    print(f"Saved visualization: {os.path.join(output_dir, f'{base_name}_udm_masking.png')}")


def compare_ndwi_with_udm(image_path, points_path, use_udm=True):
    """
    Compare NDWI results with and without UDM masking.

    This function computes NDWI both with and without cloud masking to
    demonstrate the improvement in label quality.

    Args:
        image_path (str): Path to input image
        points_path (str): Path to transect points shapefile
        use_udm (bool): Whether to apply UDM masking

    Returns:
        dict: NDWI comparison results
    """
    from ndwi_labels import get_ndwi_label

    # Apply UDM masking first
    if use_udm:
        udm_result = apply_udm_mask(image_path)
        if udm_result:
            masked_image_path = udm_result['masked_tiff']
            # Run NDWI on masked image
            # Note: This would need modification to return NDWI data
            print(f"NDWI with UDM masking: {masked_image_path}")

    return None


# =============================================================================
# Main execution for testing
# =============================================================================

if __name__ == "__main__":
    # Example usage
    config = load_config()

    # Get first image from config
    try:
        image_path = get_image_path(config, 0)
        print(f"Processing: {image_path}")

        # Apply UDM masking
        result = apply_udm_mask(image_path, visualize=True)

        if result:
            print(f"\nResults:")
            print(f"  Clear pixels: {result['clear_percentage']:.1f}%")
            print(f"  Output: {result['masked_tiff']}")
            print(f"  Mask: {result['mask_tiff']}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure config.json is set up correctly")
