"""
Module: dem_integration.py

Description: Integrate Digital Elevation Model (DEM) data for enhanced coastline
             segmentation, particularly in cliff areas and steep terrain.

             Features:
             - DEM preprocessing and coregistration with satellite imagery
             - Slope and aspect calculation
             - Cliff/steep terrain detection
             - Elevation-stratified NDWI thresholding
             - Terrain-based mask generation for challenging areas

Author: GSoC 2026 Team
Date: 2026-03-28

Issue: #102 - DEM Integration for Cliff Area Segmentation
"""

import os
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
import cv2
from scipy.ndimage import sobel
from matplotlib import pyplot as plt
import geopandas as gpd

# Local imports
from load_config import load_config
from utils.check_crs import check_crs, crs_match


# =============================================================================
# Configuration
# =============================================================================

# Slope thresholds for cliff detection (in degrees)
SLOPE_CONFIG = {
    'cliff_threshold': 30.0,      # Slope angle >= 30 degrees = cliff
    'steep_threshold': 15.0,      # Slope angle >= 15 degrees = steep terrain
    'moderate_threshold': 5.0,    # Slope angle >= 5 degrees = moderate terrain
}

# DEM processing parameters
DEM_CONFIG = {
    'target_crs': 'EPSG:32603',   # UTM Zone 3N (matches Alaska coastline)
    'target_resolution': 3.125,   # Match PlanetLabs resolution (meters)
    'dem_no_data': -9999,         # Common DEM nodata value
}


# =============================================================================
# DEM Preprocessing
# =============================================================================

def preprocess_dem(dem_path, target_bounds=None, target_transform=None,
                   target_crs='EPSG:32603', target_resolution=3.125):
    """
    Preprocess DEM to match satellite imagery extent and resolution.

    Args:
        dem_path (str): Path to input DEM GeoTIFF
        target_bounds (tuple): Target bounds (left, bottom, right, top)
        target_transform (Affine): Target geotransform
        target_crs (str): Target CRS (default: EPSG:32603)
        target_resolution (float): Target pixel resolution in meters

    Returns:
        tuple: (dem_array, transform, profile) - Preprocessed DEM data
    """
    print(f"Preprocessing DEM: {dem_path}")

    with rio.open(dem_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        dem_profile = src.profile.copy()

        # Read DEM data
        dem_data = src.read(1).astype(np.float32)

        # Handle nodata
        nodata = src.nodata if src.nodata is not None else DEM_CONFIG['dem_no_data']
        dem_data = np.where(dem_data == nodata, np.nan, dem_data)

        print(f"  Original DEM shape: {dem_data.shape}")
        print(f"  Original CRS: {dem_crs}")
        print(f"  Elevation range: {np.nanmin(dem_data):.1f} - {np.nanmax(dem_data):.1f} m")

    # Reproject if needed
    if str(dem_crs) != target_crs:
        print(f"  Reprojecting from {dem_crs} to {target_crs}...")
        dem_data, dem_transform = reproject_dem(
            dem_data, dem_transform, dem_crs,
            target_crs, target_resolution
        )

    # Crop to target bounds if specified
    if target_bounds is not None:
        dem_data, dem_transform = crop_dem_to_bounds(
            dem_data, dem_transform, target_bounds
        )

    # Create output profile
    height, width = dem_data.shape
    output_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'width': width,
        'height': height,
        'transform': dem_transform,
        'crs': target_crs,
        'nodata': np.nan,
    }

    print(f"  Preprocessed DEM shape: {dem_data.shape}")

    return dem_data, dem_transform, output_profile


def reproject_dem(dem_data, src_transform, src_crs, target_crs, target_resolution):
    """
    Reproject DEM to target CRS and resolution.

    Args:
        dem_data (np.ndarray): Source DEM array
        src_transform (Affine): Source geotransform
        src_crs (CRS): Source CRS
        target_crs (str): Target CRS
        target_resolution (float): Target resolution in meters

    Returns:
        tuple: (reprojected_dem, new_transform)
    """
    from rasterio.warp import calculate_default_transform, reproject

    # Calculate output dimensions
    src_bounds = rio.transform.array_bounds(
        dem_data.shape[0], dem_data.shape[1], src_transform
    )

    # Estimate output size based on target resolution
    width = int((src_bounds[2] - src_bounds[0]) / target_resolution)
    height = int((src_bounds[3] - src_bounds[1]) / target_resolution)

    # Calculate transform
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, width, height, *src_bounds
    )

    # Reproject
    dst_data = np.empty((dst_height, dst_width), dtype=np.float32)

    reproject(
        source=dem_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan
    )

    return dst_data, dst_transform


def crop_dem_to_bounds(dem_data, dem_transform, target_bounds):
    """
    Crop DEM to target bounds.

    Args:
        dem_data (np.ndarray): DEM array
        dem_transform (Affine): DEM geotransform
        target_bounds (tuple): Target bounds (left, bottom, right, top)

    Returns:
        tuple: (cropped_dem, new_transform)
    """
    from rasterio.windows import from_bounds, transform

    # Calculate window
    window = from_bounds(*target_bounds, dem_transform)

    # Round window to integer bounds
    col_off = int(window.col_off)
    row_off = int(window.row_off)
    width = int(window.width)
    height = int(window.height)

    # Ensure bounds are within array
    col_off = max(0, col_off)
    row_off = max(0, row_off)
    width = min(width, dem_data.shape[1] - col_off)
    height = min(height, dem_data.shape[0] - row_off)

    # Crop
    cropped_dem = dem_data[row_off:row_off+height, col_off:col_off+width]

    # Update transform
    new_transform = transform(window, dem_transform)

    return cropped_dem, new_transform


# =============================================================================
# Terrain Analysis
# =============================================================================

def calculate_slope(dem_data, pixel_size=3.125):
    """
    Calculate slope angle from DEM using Sobel operator.

    Args:
        dem_data (np.ndarray): DEM elevation array
        pixel_size (float): Pixel size in meters

    Returns:
        np.ndarray: Slope angle in degrees
    """
    # Calculate gradients using Sobel operator
    dy, dx = sobel(dem_data, axis=0), sobel(dem_data, axis=1)

    # Convert to slope angle (degrees)
    slope_radians = np.arctan(np.sqrt(dx*dx + dy*dy) / (2 * pixel_size))
    slope_degrees = np.degrees(slope_radians)

    return slope_degrees


def calculate_aspect(dem_data, pixel_size=3.125):
    """
    Calculate aspect (direction of steepest slope) from DEM.

    Args:
        dem_data (np.ndarray): DEM elevation array
        pixel_size (float): Pixel size in meters

    Returns:
        np.ndarray: Aspect angle in degrees (0-360, 0=North, clockwise)
    """
    # Calculate gradients
    dy, dx = sobel(dem_data, axis=0), sobel(dem_data, axis=1)

    # Calculate aspect
    aspect_radians = np.arctan2(-dx, dy)
    aspect_degrees = np.degrees(aspect_radians)

    # Convert to 0-360 range (0 = North, clockwise)
    aspect_degrees = (aspect_degrees + 360) % 360

    return aspect_degrees


def calculate_terrain_ruggedness(dem_data, window_size=3):
    """
    Calculate Terrain Ruggedness Index (TRI).

    TRI = mean absolute difference between center pixel and neighbors

    Args:
        dem_data (np.ndarray): DEM elevation array
        window_size (int): Size of neighborhood window

    Returns:
        np.ndarray: TRI values
    """
    from scipy.ndimage import generic_filter

    def tri_calc(window):
        center = window[len(window)//2]
        return np.mean(np.abs(window - center))

    tri = generic_filter(dem_data, tri_calc, size=window_size)

    return tri


def detect_cliff_areas(slope_data, aspect_data=None, tri_data=None):
    """
    Detect cliff and steep terrain areas based on slope threshold.

    Args:
        slope_data (np.ndarray): Slope angle array (degrees)
        aspect_data (np.ndarray, optional): Aspect array
        tri_data (np.ndarray, optional): Terrain ruggedness array

    Returns:
        dict: Cliff detection results
    """
    # Binary cliff mask
    cliff_mask = slope_data >= SLOPE_CONFIG['cliff_threshold']
    steep_mask = slope_data >= SLOPE_CONFIG['steep_threshold']
    moderate_mask = slope_data >= SLOPE_CONFIG['moderate_threshold']

    # Classify terrain
    terrain_class = np.zeros_like(slope_data, dtype=np.uint8)
    terrain_class[moderate_mask] = 1      # Moderate (5-15 degrees)
    terrain_class[steep_mask] = 2         # Steep (15-30 degrees)
    terrain_class[cliff_mask] = 3         # Cliff (>30 degrees)

    result = {
        'cliff_mask': cliff_mask,
        'steep_mask': steep_mask,
        'moderate_mask': moderate_mask,
        'terrain_class': terrain_class,
        'cliff_pixels': np.sum(cliff_mask),
        'steep_pixels': np.sum(steep_mask) - np.sum(cliff_mask),
        'moderate_pixels': np.sum(moderate_mask) - np.sum(steep_mask),
    }

    if tri_data is not None:
        result['tri'] = tri_data
        result['high_ruggedness'] = tri_data > np.percentile(tri_data, 75)

    if aspect_data is not None:
        result['aspect'] = aspect_data

    return result


# =============================================================================
# Elevation-Stratified NDWI Thresholding
# =============================================================================

def elevation_stratified_threshold(ndwi_data, dem_data, slope_data,
                                   base_threshold=None, n_elevation_zones=5):
    """
    Apply elevation-stratified thresholding for NDWI classification.

    Different elevation zones may have different water characteristics
    (e.g., shadows in valleys, snow at high elevations).

    Args:
        ndwi_data (np.ndarray): NDWI array
        dem_data (np.ndarray): DEM elevation array
        slope_data (np.ndarray): Slope angle array
        base_threshold (float, optional): Base NDWI threshold
        n_elevation_zones (int): Number of elevation zones

    Returns:
        tuple: (classified_array, zone_thresholds)
    """
    if base_threshold is None:
        # Use Otsu threshold as base
        ndwi_8bit = ((ndwi_data + 1) * 127).astype(np.uint8)
        _, base_threshold = cv2.threshold(
            ndwi_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        base_threshold = (base_threshold / 127) - 1

    # Create elevation zones
    dem_valid = dem_data[~np.isnan(dem_data)]
    elevation_percentiles = np.percentile(dem_valid, np.linspace(0, 100, n_elevation_zones + 1))

    # Create output array
    classified = np.zeros_like(ndwi_data, dtype=np.uint8)
    zone_thresholds = {}

    for i in range(n_elevation_zones):
        # Create zone mask
        zone_mask = (dem_data >= elevation_percentiles[i]) & \
                    (dem_data < elevation_percentiles[i + 1])

        # Adjust threshold based on slope
        zone_slope = slope_data[zone_mask]
        if len(zone_slope) > 0:
            mean_slope = np.mean(zone_slope)

            # Steeper slopes need higher thresholds (shadows look like water)
            slope_adjustment = 0.01 * (mean_slope / 10)  # +0.01 per 10 degrees

            zone_threshold = min(base_threshold + slope_adjustment, 0.5)
        else:
            zone_threshold = base_threshold

        zone_thresholds[f'zone_{i}'] = {
            'elevation_range': (elevation_percentiles[i], elevation_percentiles[i + 1]),
            'threshold': zone_threshold,
            'slope_adjustment': slope_adjustment if len(zone_slope) > 0 else 0
        }

        # Apply threshold
        zone_ndwi = ndwi_data[zone_mask]
        zone_classified = zone_ndwi > zone_threshold
        classified[zone_mask] = zone_classified.astype(np.uint8)

    return classified, zone_thresholds


def terrain_masked_ndwi(ndwi_data, slope_data, dem_data=None,
                        max_slope_for_water=45.0):
    """
    Create terrain-masked NDWI classification.

    Water is unlikely to occur on very steep slopes - these are likely shadows.

    Args:
        ndwi_data (np.ndarray): NDWI array
        slope_data (np.ndarray): Slope angle array
        dem_data (np.ndarray, optional): DEM for additional filtering
        max_slope_for_water (float): Maximum slope where water can occur

    Returns:
        np.ndarray: Terrain-masked water classification
    """
    # Initial NDWI classification
    ndwi_8bit = ((ndwi_data + 1) * 127).astype(np.uint8)
    _, threshold = cv2.threshold(
        ndwi_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    ndwi_classified = ndwi_data > ((threshold / 127) - 1)

    # Create slope mask (exclude very steep areas)
    slope_mask = slope_data < max_slope_for_water

    # Apply slope mask
    terrain_masked = ndwi_classified & slope_mask

    return terrain_masked


# =============================================================================
# Integration with NDWI Pipeline
# =============================================================================

def process_with_dem(image_path, dem_path, points_path=None, output_dir=None):
    """
    Complete processing pipeline integrating DEM data with NDWI classification.

    Args:
        image_path (str): Path to satellite imagery
        dem_path (str): Path to DEM file
        points_path (str, optional): Path to transect points
        output_dir (str, optional): Output directory

    Returns:
        dict: Processing results
    """
    from ndwi_labels import get_ndwi_label

    if output_dir is None:
        output_dir = "result_dem_integration"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("DEM-INTEGRATED COASTLINE EXTRACTION")
    print("=" * 60)

    # Step 1: Preprocess DEM
    print("\nStep 1: Preprocessing DEM...")
    with rio.open(image_path) as src:
        target_bounds = src.bounds
        target_transform = src.transform
        target_crs = str(src.crs)
        pixel_size = abs(src.transform[0])

    dem_data, dem_transform, dem_profile = preprocess_dem(
        dem_path, target_bounds, target_transform, target_crs, pixel_size
    )

    # Step 2: Calculate terrain metrics
    print("\nStep 2: Calculating terrain metrics...")
    slope_data = calculate_slope(dem_data, pixel_size)
    aspect_data = calculate_aspect(dem_data, pixel_size)
    tri_data = calculate_terrain_ruggedness(dem_data)

    print(f"  Slope range: {np.nanmin(slope_data):.1f} - {np.nanmax(slope_data):.1f} degrees")
    print(f"  Cliff pixels detected: {np.sum(slope_data >= SLOPE_CONFIG['cliff_threshold']):,}")

    # Step 3: Detect cliff areas
    print("\nStep 3: Detecting cliff/steep terrain areas...")
    terrain_results = detect_cliff_areas(slope_data, aspect_data, tri_data)

    # Step 4: Run standard NDWI
    print("\nStep 4: Running NDWI classification...")
    ndwi_results = get_ndwi_label(image_path, points_path, output_dir=output_dir)

    # Step 5: Apply terrain-based corrections
    print("\nStep 5: Applying terrain-based corrections...")
    ndwi_data = ndwi_results.get('ndwi', ndwi_results.get('ndwi_array'))

    if ndwi_data is not None:
        # Terrain-masked classification
        terrain_masked = terrain_masked_ndwi(ndwi_data, slope_data)

        # Elevation-stratified thresholding
        stratified_class, zone_thresholds = elevation_stratified_threshold(
            ndwi_data, dem_data, slope_data
        )

        print(f"  Standard NDWI water pixels: {np.sum(ndwi_results.get('ndwi_classified', np.zeros_like(terrain_masked))):,}")
        print(f"  Terrain-masked water pixels: {np.sum(terrain_masked):,}")
        print(f"  Stratified water pixels: {np.sum(stratified_class):,}")

    # Step 6: Save outputs
    print("\nStep 6: Saving outputs...")

    # Save DEM derivatives
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save slope
    slope_path = os.path.join(output_dir, f"{base_name}_slope.tif")
    save_raster(slope_data, dem_profile, slope_path)
    print(f"  Saved slope: {slope_path}")

    # Save cliff mask
    cliff_path = os.path.join(output_dir, f"{base_name}_cliff_mask.tif")
    cliff_profile = dem_profile.copy()
    cliff_profile.update(dtype='uint8')
    save_raster(terrain_results['cliff_mask'].astype(np.uint8), cliff_profile, cliff_path)
    print(f"  Saved cliff mask: {cliff_path}")

    # Save visualization
    save_dem_visualization(
        dem_data, slope_data, terrain_results, ndwi_results,
        output_dir, base_name
    )

    return {
        'dem_data': dem_data,
        'slope': slope_data,
        'aspect': aspect_data,
        'tri': tri_data,
        'terrain_results': terrain_results,
        'ndwi_results': ndwi_results,
        'zone_thresholds': zone_thresholds if 'zone_thresholds' in dir() else None
    }


def save_raster(data, profile, output_path):
    """Save array as GeoTIFF."""
    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)


def save_dem_visualization(dem_data, slope_data, terrain_results, ndwi_results,
                           output_dir, base_name):
    """
    Create visualization of DEM integration results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # DEM
    im = axes[0, 0].imshow(dem_data, cmap='terrain')
    axes[0, 0].set_title('Digital Elevation Model')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], label='Elevation (m)')

    # Slope
    im = axes[0, 1].imshow(slope_data, cmap='YlOrRd')
    axes[0, 1].set_title(f'Slope (degrees)\nCliff pixels: {terrain_results["cliff_pixels"]:,}')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], label='Slope (°)')

    # Terrain classification
    im = axes[0, 2].imshow(terrain_results['terrain_class'], cmap='viridis')
    axes[0, 2].set_title('Terrain Classification')
    axes[0, 2].axis('off')

    # NDWI
    if 'ndwi' in ndwi_results:
        im = axes[1, 0].imshow(ndwi_results['ndwi'], cmap='RdYlBu')
        axes[1, 0].set_title('NDWI Values')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], label='NDWI')

    # NDWI classified
    if 'ndwi_classified' in ndwi_results:
        im = axes[1, 1].imshow(ndwi_results['ndwi_classified'], cmap='Blues')
        axes[1, 1].set_title('NDWI Classified')
        axes[1, 1].axis('off')

    # Cliff overlay
    if 'ndwi_classified' in ndwi_results:
        overlay = np.zeros((*ndwi_results['ndwi_classified'].shape, 3))
        overlay[ndwi_results['ndwi_classified'] == 1] = [0, 0, 1]  # Water = blue
        overlay[terrain_results['cliff_mask']] = [1, 0, 0]  # Cliff = red

        im = axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Water (blue) and Cliffs (red)')
        axes[1, 2].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{base_name}_dem_integration.png"), dpi=150)
    plt.close()

    print(f"  Saved visualization: {os.path.join(output_dir, f'{base_name}_dem_integration.png')}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    config = load_config()

    # Example usage
    print("DEM Integration Module - Test Run")
    print("=" * 60)

    # Get DEM path from environment or config
    dem_path = os.environ.get('DEM_PATH', 'data/dem/Alaska_DEM.tif')

    if not os.path.exists(dem_path):
        print(f"DEM file not found: {dem_path}")
        print("Set DEM_PATH environment variable to test.")
    else:
        # Get image path from config
        from load_config import get_image_path
        image_path = get_image_path(config, 0)

        if image_path and os.path.exists(image_path):
            results = process_with_dem(image_path, dem_path)
            print("\nProcessing complete!")
        else:
            print("No image path available in config.")
