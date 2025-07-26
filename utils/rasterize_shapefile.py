"""
Rasterize Shapefile Module

This script converts the 2016_HiRes_Final_Coastline.shp shapefile (vector format)
into a raster GeoTIFF file. The output raster is useful as a base image for georeferencing,
model training, or other spatial analysis tasks.

The script uses configuration values from config_template.json to locate input and 
output paths, ensuring portability across systems without hardcoding directories.

Input:
    - ground_truth/2016_HiRes_Final_Coastline.shp

Output:
    - ground_truth/2016_HiRes_Final_Coastline.tif

Usage:
    Run this script from the CoastlineExtraction directory:
        python utils/rasterize_shapefile.py

Requirements:
    - config_template.json must define:
        * 'ground_truth_folder'
        * 'ground_truth_files' (first file should be "2016_HiRes_Final_Coastline.shp")
    - Required Python packages: geopandas, rasterio, numpy
"""

import geopandas as gpd  
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np
import os
import sys

# Add parent directory to path to import load_config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from load_config import load_config, get_ground_truth_path

def rasterize_shapefile():
    """
    Converts the input coastline shapefile into a binary raster (GeoTIFF).
    
    The function reads the coastline shapefile from the ground_truth folder and
    creates a binary raster where pixels representing the coastline are set to 1
    and background pixels are set to 0. The raster uses a pixel size of 0.0001
    degrees and maintains the original coordinate reference system.

    Parameters:
        None (uses configuration from config_template.json)

    Returns:
        None

    Raises:
        ValueError: If the shapefile contains no features or if the computed raster 
                    dimensions are invalid (e.g., due to small bounds or pixel size).

    Output:
        ground_truth/2016_HiRes_Final_Coastline.tif (GeoTIFF format)
        - Binary raster with values 0 (background) and 1 (coastline)
        - Pixel size: 0.0001 degrees
        - Coordinate system: Same as input shapefile
    """
    
    # Load configuration
    config = load_config()
    
    # Get input shapefile path from config
    shapefile_path = get_ground_truth_path(config, 0)  # First ground truth file (2016_HiRes_Final_Coastline.shp) in ground_truth folder
    
    # Get output path in ground_truth folder 
    ground_truth_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['ground_truth_folder'])
    output_tif_path = os.path.join(ground_truth_dir, '2016_HiRes_Final_Coastline.tif')
    
    # Load shapefile 
    gdf = gpd.read_file(shapefile_path)
    
    # Diagnostics to check if the shapefile is loaded correctly 
    print("Number of features in shapefile:", len(gdf))
    print("Bounds:", gdf.total_bounds)
    
    if len(gdf) == 0:
        raise ValueError(f"Shapefile {shapefile_path} contains no features. Please check your data.")
    
    # Define raster size, bounds, and resolution
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    pixel_size = 0.0001
    width = int((bounds[2] - bounds[0]) / pixel_size)
    height = int((bounds[3] - bounds[1]) / pixel_size)
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster dimensions: width={width}, height={height}.\n"
                         f"Check your shapefile bounds and pixel_size.\n"
                         f"Bounds: {bounds}, pixel_size: {pixel_size}\n"
                         f"Try reducing pixel_size if your bounds are small.")
    
    transform = from_origin(bounds[0], bounds[3], pixel_size, pixel_size)
    
    # Rasterize the shapefile
    shapes = ((geom, 1) for geom in gdf.geometry)
    raster = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
    
    # Save to .tif file format
    with rasterio.open(
        output_tif_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=gdf.crs,
        transform=transform
    ) as dst:
        dst.write(raster, 1)
    
    print(f"Rasterization complete. Output saved to: {output_tif_path}")
    

if __name__ == "__main__":
    rasterize_shapefile()  