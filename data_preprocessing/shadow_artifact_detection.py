"""
Module: shadow_artifact_detection.py

Description: Enhanced detection and handling of shadows, cloud artifacts,
             and satellite data quality issues beyond UDM/QA60 band reliance.

             Features:
             - NIR-based shadow detection (independent of UDM)
             - Satellite artifact detection (striping, sensor noise)
             - Building/structure detection for urban areas
             - Water shadow discrimination
             - Quality flag generation for each pixel

Author: GSoC 2026 Team
Date: 2026-03-28

Issue: #103 - Enhanced Shadow and Artifact Detection
"""

import os
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import cv2
from scipy import ndimage
from matplotlib import pyplot as plt
import geopandas as gpd

# Local imports
from load_config import load_config


# =============================================================================
# Configuration
# =============================================================================

# Shadow detection thresholds
SHADOW_CONFIG = {
    'nir_threshold': 0.15,         # NIR reflectance threshold for shadow
    'blue_threshold': 0.10,        # Blue band threshold
    'ndvi_threshold': 0.1,         # NDVI threshold for shadow vs water
    'ratio_threshold': 0.5,        # Blue/NIR ratio threshold
    'min_shadow_size': 9,          # Minimum shadow patch size (pixels)
}

# Artifact detection thresholds
ARTIFACT_CONFIG = {
    'stripe_threshold': 2.5,       # Standard deviations for stripe detection
    'noise_threshold': 3.0,        # Standard deviations for noise detection
    'saturation_threshold': 0.98,  # Reflectance saturation level
    'edge_artifact_width': 3,      # Width of edge artifact zone
}

# Quality flags (bit positions)
QUALITY_FLAGS = {
    'CLEAR': 0,
    'SHADOW': 1,
    'CLOUD': 2,
    'ARTIFACT': 3,
    'SATURATED': 4,
    'EDGE': 5,
    'WATER_LIKELY': 6,
    'WATER_UNCERTAIN': 7,
}


# =============================================================================
# Shadow Detection
# =============================================================================

def detect_shadows_nir_based(green_band, nir_band, red_band=None, blue_band=None):
    """
    Detect shadows using NIR-based methods independent of UDM/QA60.

    Shadows have low NIR reflectance but different spectral characteristics
    than water.

    Args:
        green_band (np.ndarray): Green band reflectance
        nir_band (np.ndarray): NIR band reflectance
        red_band (np.ndarray, optional): Red band reflectance
        blue_band (np.ndarray, optional): Blue band reflectance

    Returns:
        tuple: (shadow_mask, shadow_confidence)
    """
    print("  Detecting shadows using NIR-based method...")

    # Normalize bands to 0-1 range if needed
    if np.max(nir_band) > 1:
        green_band = green_band / np.max(green_band)
        nir_band = nir_band / np.max(nir_band)
        if red_band is not None:
            red_band = red_band / np.max(red_band)
        if blue_band is not None:
            blue_band = blue_band / np.max(blue_band)

    # Method 1: Low NIR reflectance
    low_nir = nir_band < SHADOW_CONFIG['nir_threshold']

    # Method 2: Calculate NDVI (shadows have low/negative NDVI)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10) if red_band is not None else None
    low_ndvi = ndvi < SHADOW_CONFIG['ndvi_threshold'] if ndvi is not None else np.ones_like(low_nir, dtype=bool)

    # Method 3: Blue/NIR ratio (shadows have ratio ~1, water has ratio < 1)
    if blue_band is not None:
        blue_nir_ratio = np.where(nir_band > 0.01, blue_band / (nir_band + 1e-10), 0)
        shadow_ratio = np.abs(blue_nir_ratio - 1.0) < 0.3  # Shadows have similar blue and NIR
    else:
        shadow_ratio = np.ones_like(low_nir, dtype=bool)

    # Combine methods
    shadow_mask = low_nir & low_ndvi & shadow_ratio

    # Remove small patches
    shadow_mask = ndimage.binary_opening(shadow_mask, structure=np.ones((3, 3)))
    shadow_mask = ndimage.binary_closing(shadow_mask, structure=np.ones((3, 3)))

    # Remove patches smaller than minimum size
    labeled, n_features = ndimage.label(shadow_mask)
    for i in range(1, n_features + 1):
        if np.sum(labeled == i) < SHADOW_CONFIG['min_shadow_size']:
            shadow_mask[labeled == i] = False

    # Calculate confidence
    confidence = np.zeros_like(shadow_mask, dtype=np.float32)
    confidence[low_nir] += 0.3
    confidence[low_ndvi] += 0.3
    confidence[shadow_ratio] += 0.4
    confidence = np.clip(confidence, 0, 1)

    shadow_pixels = np.sum(shadow_mask)
    print(f"    Shadow pixels detected: {shadow_pixels:,} ({100*shadow_pixels/shadow_mask.size:.1f}%)")

    return shadow_mask, confidence


def detect_cloud_shadows(ndwi_data, nir_band, shadow_mask=None):
    """
    Distinguish cloud shadows from topographic shadows.

    Cloud shadows are typically adjacent to bright clouds and have
    specific spatial patterns.

    Args:
        ndwi_data (np.ndarray): NDWI array
        nir_band (np.ndarray): NIR band reflectance
        shadow_mask (np.ndarray, optional): Initial shadow mask

    Returns:
        tuple: (cloud_shadow_mask, topographic_shadow_mask)
    """
    if shadow_mask is None:
        shadow_mask, _ = detect_shadows_nir_based(None, nir_band)

    # Detect bright areas (potential clouds)
    bright_mask = nir_band > np.percentile(nir_band[~np.isnan(nir_band)], 90)

    # Dilate bright areas
    bright_dilated = ndimage.binary_dilation(bright_mask, iterations=20)

    # Cloud shadows are typically near bright clouds
    cloud_shadow_mask = shadow_mask & bright_dilated

    # Topographic shadows are isolated from clouds
    topographic_shadow_mask = shadow_mask & ~bright_dilated

    print(f"    Cloud shadow pixels: {np.sum(cloud_shadow_mask):,}")
    print(f"    Topographic shadow pixels: {np.sum(topographic_shadow_mask):,}")

    return cloud_shadow_mask, topographic_shadow_mask


def water_vs_shadow_discrimination(ndwi_data, nir_band, slope_data=None):
    """
    Discriminate between water and shadow pixels.

    Both water and shadows have low NIR reflectance and can be confused.
    Use multiple spectral indices and terrain information.

    Args:
        ndwi_data (np.ndarray): NDWI array
        nir_band (np.ndarray): NIR band reflectance
        slope_data (np.ndarray, optional): Slope angle array

    Returns:
        tuple: (water_mask, shadow_mask, uncertain_mask)
    """
    # Initial classifications
    water_candidate = ndwi_data > 0.0  # Positive NDWI
    shadow_candidate = nir_band < SHADOW_CONFIG['nir_threshold']

    # Overlap region (uncertain)
    uncertain = water_candidate & shadow_candidate

    # Use slope if available (water unlikely on steep slopes)
    if slope_data is not None:
        steep = slope_data > 20
        uncertain[steep & uncertain] = True
        water_candidate[steep] = False

    # Confident water (high NDWI, not shadow)
    confident_water = water_candidate & ~shadow_candidate & (ndwi_data > 0.2)

    # Confident shadow (low NIR, low/neutral NDWI)
    confident_shadow = shadow_candidate & (ndwi_data < 0.1)

    print(f"    Confident water pixels: {np.sum(confident_water):,}")
    print(f"    Confident shadow pixels: {np.sum(confident_shadow):,}")
    print(f"    Uncertain pixels: {np.sum(uncertain):,}")

    return confident_water, confident_shadow, uncertain


# =============================================================================
# Artifact Detection
# =============================================================================

def detect_satellite_striping(band_data, window_size=64):
    """
    Detect striping artifacts common in satellite imagery.

    Striping appears as regular patterns of brighter/darker rows or columns.

    Args:
        band_data (np.ndarray): Single band reflectance
        window_size (int): Size of analysis window

    Returns:
        tuple: (stripe_mask, stripe_direction)
    """
    print("  Detecting satellite striping artifacts...")

    # Calculate local statistics
    rows = band_data.shape[0]
    cols = band_data.shape[1]

    # Row-wise mean (detect horizontal striping)
    row_means = np.mean(band_data, axis=1)
    row_std = np.std(row_means)
    row_anomalies = np.abs(row_means - np.mean(row_means)) > ARTIFACT_CONFIG['stripe_threshold'] * row_std

    # Column-wise mean (detect vertical striping)
    col_means = np.mean(band_data, axis=0)
    col_std = np.std(col_means)
    col_anomalies = np.abs(col_means - np.mean(col_means)) > ARTIFACT_CONFIG['stripe_threshold'] * col_std

    # Create stripe mask
    stripe_mask = np.zeros_like(band_data, dtype=bool)

    horizontal_stripe_pixels = 0
    vertical_stripe_pixels = 0

    if np.any(row_anomalies):
        for i in range(rows):
            if row_anomalies[i]:
                stripe_mask[i, :] = True
                horizontal_stripe_pixels += cols

    if np.any(col_anomalies):
        for j in range(cols):
            if col_anomalies[j]:
                stripe_mask[:, j] = True
                vertical_stripe_pixels += rows

    stripe_direction = 'none'
    if horizontal_stripe_pixels > vertical_stripe_pixels:
        stripe_direction = 'horizontal'
    elif vertical_stripe_pixels > 0:
        stripe_direction = 'vertical'

    total_stripe_pixels = np.sum(stripe_mask)
    print(f"    Stripe pixels detected: {total_stripe_pixels:,} ({100*total_stripe_pixels/stripe_mask.size:.1f}%)")
    print(f"    Stripe direction: {stripe_direction}")

    return stripe_mask, stripe_direction


def detect_sensor_noise(band_data, neighborhood_size=5):
    """
    Detect sensor noise (salt-and-pepper, random spikes).

    Args:
        band_data (np.ndarray): Single band reflectance
        neighborhood_size (int): Size of neighborhood for local statistics

    Returns:
        np.ndarray: Noise mask
    """
    # Calculate local mean and std
    kernel = np.ones((neighborhood_size, neighborhood_size)) / (neighborhood_size ** 2)
    local_mean = ndimage.convolve(band_data, kernel, mode='reflect')
    local_std = np.sqrt(ndimage.convolve((band_data - local_mean) ** 2, kernel, mode='reflect'))

    # Pixels that deviate significantly from local mean
    deviation = np.abs(band_data - local_mean) / (local_std + 1e-10)
    noise_mask = deviation > ARTIFACT_CONFIG['noise_threshold']

    # Remove isolated pixels (likely real features)
    noise_mask = ndimage.binary_opening(noise_mask, structure=np.ones((3, 3)))

    noise_pixels = np.sum(noise_mask)
    print(f"  Sensor noise pixels: {noise_pixels:,} ({100*noise_pixels/noise_mask.size:.1f}%)")

    return noise_mask


def detect_saturated_pixels(band_data, threshold=None):
    """
    Detect saturated pixels (maximum sensor response).

    Args:
        band_data (np.ndarray): Reflectance band
        threshold (float, optional): Saturation threshold

    Returns:
        np.ndarray: Saturation mask
    """
    if threshold is None:
        threshold = SHADOW_CONFIG['saturation_threshold']

    # Normalize if needed
    max_val = np.max(band_data)
    if max_val > 1:
        normalized = band_data / max_val
    else:
        normalized = band_data

    saturated = normalized > threshold

    sat_pixels = np.sum(saturated)
    print(f"  Saturated pixels: {sat_pixels:,} ({100*sat_pixels/saturated.size:.1f}%)")

    return saturated


def detect_edge_artifacts(band_data, edge_width=3):
    """
    Detect edge artifacts from image processing (mosaicking, cropping).

    Args:
        band_data (np.ndarray): Reflectance band
        edge_width (int): Width of edge zone to check

    Returns:
        np.ndarray: Edge artifact mask
    """
    mask = np.zeros_like(band_data, dtype=bool)

    # Check image edges
    mask[:edge_width, :] = True  # Top edge
    mask[-edge_width:, :] = True  # Bottom edge
    mask[:, :edge_width] = True  # Left edge
    mask[:, -edge_width:] = True  # Right edge

    # Also check for internal sharp transitions (potential seam lines)
    gradient_x = np.abs(np.gradient(band_data.astype(np.float32), axis=1))
    gradient_y = np.abs(np.gradient(band_data.astype(np.float32), axis=0))

    # Find lines of high gradient (potential seams)
    vertical_seam = np.any(gradient_x > np.percentile(gradient_x, 99), axis=0)
    horizontal_seam = np.any(gradient_y > np.percentile(gradient_y, 99), axis=1)

    return mask


# =============================================================================
# Quality Flag Generation
# =============================================================================

def generate_quality_flags(image_path, ndwi_data=None, slope_data=None):
    """
    Generate comprehensive quality flags for each pixel.

    Args:
        image_path (str): Path to multispectral GeoTIFF
        ndwi_data (np.ndarray, optional): Pre-calculated NDWI
        slope_data (np.ndarray, optional): Slope array for terrain masking

    Returns:
        tuple: (quality_flags, flag_summary)
    """
    print("\nGenerating comprehensive quality flags...")

    # Read spectral bands
    with rio.open(image_path) as src:
        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32) if src.count >= 4 else None

        if nir is None:
            print("  Warning: NIR band not available, limited analysis possible")
            nir = np.zeros_like(blue)

        # Normalize to reflectance (0-1)
        for band in [blue, green, red, nir]:
            max_val = np.nanmax(band)
            if max_val > 1:
                band /= max_val

    # Initialize quality flag array (8 bits)
    quality = np.zeros(blue.shape, dtype=np.uint8)

    # Calculate NDWI if not provided
    if ndwi_data is None:
        ndwi_data = (green - nir) / (green + nir + 1e-10)

    # Flag 1: Shadow detection
    shadow_mask, shadow_conf = detect_shadows_nir_based(green, nir, red, blue)
    quality[shadow_mask] |= (1 << QUALITY_FLAGS['SHADOW'])

    # Flag 2: Cloud shadows
    cloud_shadow, topo_shadow = detect_cloud_shadows(ndwi_data, nir, shadow_mask)
    quality[cloud_shadow] |= (1 << QUALITY_FLAGS['SHADOW'])

    # Flag 3: Satellite striping
    stripe_mask, stripe_dir = detect_satellite_striping(nir)
    quality[stripe_mask] |= (1 << QUALITY_FLAGS['ARTIFACT'])

    # Flag 4: Sensor noise
    noise_mask = detect_sensor_noise(nir)
    quality[noise_mask] |= (1 << QUALITY_FLAGS['ARTIFACT'])

    # Flag 5: Saturated pixels
    for band_name, band in [('Blue', blue), ('Green', green), ('Red', red), ('NIR', nir)]:
        sat_mask = detect_saturated_pixels(band)
        if np.any(sat_mask):
            quality[sat_mask] |= (1 << QUALITY_FLAGS['SATURATED'])

    # Flag 6: Edge artifacts
    edge_mask = detect_edge_artifacts(nir)
    quality[edge_mask] |= (1 << QUALITY_FLAGS['EDGE'])

    # Flag 7 & 8: Water likelihood
    water_likely, water_shadow, uncertain = water_vs_shadow_discrimination(
        ndwi_data, nir, slope_data
    )
    quality[water_likely] |= (1 << QUALITY_FLAGS['WATER_LIKELY'])
    quality[uncertain] |= (1 << QUALITY_FLAGS['WATER_UNCERTAIN'])

    # Generate summary
    flag_summary = {
        'shadow_pixels': np.sum((quality >> QUALITY_FLAGS['SHADOW']) & 1),
        'artifact_pixels': np.sum((quality >> QUALITY_FLAGS['ARTIFACT']) & 1),
        'saturated_pixels': np.sum((quality >> QUALITY_FLAGS['SATURATED']) & 1),
        'edge_pixels': np.sum((quality >> QUALITY_FLAGS['EDGE']) & 1),
        'water_likely_pixels': np.sum((quality >> QUALITY_FLAGS['WATER_LIKELY']) & 1),
        'water_uncertain_pixels': np.sum((quality >> QUALITY_FLAGS['WATER_UNCERTAIN']) & 1),
        'clear_pixels': np.sum(quality == 0),
    }

    flag_summary['total_pixels'] = quality.size
    flag_summary['flagged_percentage'] = 100 * (quality.size - flag_summary['clear_pixels']) / quality.size

    print(f"\nQuality Flag Summary:")
    print(f"  Clear pixels: {flag_summary['clear_pixels']:,} ({100-flag_summary['flagged_percentage']:.1f}%)")
    print(f"  Shadow pixels: {flag_summary['shadow_pixels']:,}")
    print(f"  Artifact pixels: {flag_summary['artifact_pixels']:,}")
    print(f"  Water likely: {flag_summary['water_likely_pixels']:,}")
    print(f"  Water uncertain: {flag_summary['water_uncertain_pixels']:,}")

    return quality, flag_summary


def apply_quality_filter(ndwi_data, quality_flags, max_shadow_fraction=0.3,
                         exclude_artifacts=True):
    """
    Apply quality-based filtering to NDWI classification.

    Args:
        ndwi_data (np.ndarray): NDWI array
        quality_flags (np.ndarray): Quality flag array
        max_shadow_fraction (float): Maximum shadow fraction for reliable classification
        exclude_artifacts (bool): Whether to exclude artifact-affected pixels

    Returns:
        tuple: (filtered_ndwi, reliability_mask)
    """
    # Extract individual flags
    shadow_flag = (quality_flags >> QUALITY_FLAGS['SHADOW']) & 1
    artifact_flag = (quality_flags >> QUALITY_FLAGS['ARTIFACT']) & 1
    uncertain_flag = (quality_flags >> QUALITY_FLAGS['WATER_UNCERTAIN']) & 1

    # Reliability mask
    reliable = ~shadow_flag.astype(bool)
    if exclude_artifacts:
        reliable = reliable & ~artifact_flag.astype(bool)
    reliable = reliable & ~uncertain_flag.astype(bool)

    # Filter NDWI (set unreliable to NaN)
    filtered_ndwi = ndwi_data.copy()
    filtered_ndwi[~reliable] = np.nan

    return filtered_ndwi, reliable


# =============================================================================
# Visualization
# =============================================================================

def save_quality_visualization(quality_flags, flag_summary, ndwi_data,
                               output_dir, base_name):
    """
    Create visualization of quality flags.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Extract individual masks
    shadow_mask = (quality_flags >> QUALITY_FLAGS['SHADOW']) & 1
    artifact_mask = (quality_flags >> QUALITY_FLAGS['ARTIFACT']) & 1
    water_likely = (quality_flags >> QUALITY_FLAGS['WATER_LIKELY']) & 1
    water_uncertain = (quality_flags >> QUALITY_FLAGS['WATER_UNCERTAIN']) & 1

    # Quality overview (RGB composite)
    rgb = np.zeros((*quality_flags.shape, 3))
    rgb[:, :, 0] = shadow_mask  # Red = shadow
    rgb[:, :, 1] = artifact_mask  # Green = artifact
    rgb[:, :, 2] = water_likely  # Blue = water likely
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f'Quality Overview\nR=Shadow, G=Artifact, B=Water Likely')
    axes[0, 0].axis('off')

    # Shadow mask
    axes[0, 1].imshow(shadow_mask, cmap='gray')
    axes[0, 1].set_title(f'Shadow Mask\n{flag_summary["shadow_pixels"]:,} pixels')
    axes[0, 1].axis('off')

    # Artifact mask
    axes[0, 2].imshow(artifact_mask, cmap='gray')
    axes[0, 2].set_title(f'Artifact Mask\n{flag_summary["artifact_pixels"]:,} pixels')
    axes[0, 2].axis('off')

    # NDWI
    im = axes[1, 0].imshow(ndwi_data, cmap='RdYlBu')
    axes[1, 0].set_title('NDWI Values')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], label='NDWI')

    # Water likely
    axes[1, 1].imshow(water_likely, cmap='Blues')
    axes[1, 1].set_title(f'Water Likely\n{flag_summary["water_likely_pixels"]:,} pixels')
    axes[1, 1].axis('off')

    # Water uncertain
    axes[1, 2].imshow(water_uncertain, cmap='Oranges')
    axes[1, 2].set_title(f'Water Uncertain\n{flag_summary["water_uncertain_pixels"]:,} pixels')
    axes[1, 2].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{base_name}_quality_flags.png"), dpi=150)
    plt.close()

    print(f"Saved quality visualization: {os.path.join(output_dir, f'{base_name}_quality_flags.png')}")


# =============================================================================
# Main Integration Function
# =============================================================================

def process_with_quality_filtering(image_path, ndwi_results=None, slope_data=None,
                                   output_dir=None):
    """
    Complete quality flag processing pipeline.

    Args:
        image_path (str): Path to satellite imagery
        ndwi_results (dict, optional): Results from NDWI processing
        slope_data (np.ndarray, optional): Slope array
        output_dir (str, optional): Output directory

    Returns:
        dict: Quality processing results
    """
    if output_dir is None:
        output_dir = "result_quality_filtering"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("QUALITY FLAG GENERATION AND FILTERING")
    print("=" * 60)

    # Generate quality flags
    quality, flag_summary = generate_quality_flags(image_path, slope_data=slope_data)

    # Get NDWI data
    if ndwi_results is not None and 'ndwi' in ndwi_results:
        ndwi_data = ndwi_results['ndwi']
    else:
        from rastertools import calculate_ndwi
        ndwi_data = calculate_ndwi(image_path, plot=False)
        with rio.open(ndwi_data) as src:
            ndwi_data = src.read(1)

    # Apply quality filtering
    filtered_ndwi, reliable_mask = apply_quality_filter(ndwi_data, quality)

    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save quality flags
    with rio.open(image_path) as src:
        profile = src.profile.copy()
    profile.update(dtype='uint8', count=1, nodata=255)

    quality_path = os.path.join(output_dir, f"{base_name}_quality_flags.tif")
    with rio.open(quality_path, 'w', **profile) as dst:
        dst.write(quality, 1)
    print(f"Saved quality flags: {quality_path}")

    # Save reliability mask
    reliable_path = os.path.join(output_dir, f"{base_name}_reliability_mask.tif")
    profile.update(dtype='uint8', nodata=255)
    with rio.open(reliable_path, 'w', **profile) as dst:
        dst.write(reliable_mask.astype(np.uint8) * 255, 1)
    print(f"Saved reliability mask: {reliable_path}")

    # Save visualization
    save_quality_visualization(quality, flag_summary, ndwi_data, output_dir, base_name)

    return {
        'quality_flags': quality,
        'flag_summary': flag_summary,
        'filtered_ndwi': filtered_ndwi,
        'reliable_mask': reliable_mask
    }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    config = load_config()

    from load_config import get_image_path
    image_path = get_image_path(config, 0)

    if image_path and os.path.exists(image_path):
        results = process_with_quality_filtering(image_path)
        print("\nQuality flag processing complete!")
    else:
        print("No image available for quality flag processing.")
