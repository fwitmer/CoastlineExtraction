"""
Spatial Analysis Utilities
This module contains utility functions for spatial data analysis and logging.
"""

import geopandas as gpd


def log_spatial_info(raster_bounds, points_geom):
    """
    This function helps you check the spatial relationship between your raster image and point data.

    It prints:
    - The boundary box (edges) of the raster image.
    - The number of points in your shapefile and their boundary area.
    - Whether the points overlap with the image, and if yes, the overlapping area.
    - A warning if there are no points or no overlap between the image and the points.

    Args:
        raster_bounds: The boundary box of the raster image (from rasterio).
        points_geom: A list of point shapes (GeoSeries from GeoPandas).
    """
    print("--- Spatial Information ---")
    print("Raster bounds:", raster_bounds)
    
    if points_geom.empty:
        print("WARNING: No points loaded or geometry is empty.")
        return

    print(f"Number of points loaded: {len(points_geom)}")
    points_bounds_tuple = points_geom.total_bounds
    print(f"Points bounds (projected):left={points_bounds_tuple[0]:.2f}, bottom={points_bounds_tuple[1]:.2f}, right={points_bounds_tuple[2]:.2f}, top={points_bounds_tuple[3]:.2f}")
    
    # Check spatial overlap
    overlap_left = max(raster_bounds.left, points_bounds_tuple[0])
    overlap_bottom = max(raster_bounds.bottom, points_bounds_tuple[1])
    overlap_right = min(raster_bounds.right, points_bounds_tuple[2])
    overlap_top = min(raster_bounds.top, points_bounds_tuple[3])
    
    if overlap_right > overlap_left and overlap_top > overlap_bottom:
        print(f"Overlap area: left={overlap_left:.2f}, bottom={overlap_bottom:.2f}, right={overlap_right:.2f}, top={overlap_top:.2f}")
    else:
        print("WARNING: No spatial overlap between raster and points!")
    print("--------------------------")


__all__ = ['log_spatial_info'] 