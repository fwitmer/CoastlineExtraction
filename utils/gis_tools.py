"""
utils/gis_tools.py

This module provides reusable GIS utility functions for spatial analysis and shapefile processing,
specifically tailored for coastline and transect-based workflows.

Functions included:
- create_transect_points: Computes and saves intersection points between transects and coastline lines.
- clip_shp: Clips a shapefile to a given boundary defined as a GeoJSON-like polygon.

These functions help with preprocessing and visualization of geospatial datasets during coastline
detection and accuracy assessment tasks.
"""

import geopandas as gpd
import os
from shapely.geometry import Polygon, shape
from matplotlib import pyplot as plt

def create_transect_points(transect_path, line_path, out_path):
    """
    Generates and saves intersection points between transect lines and a coastline shapefile.

    This function reads a shapefile containing transect lines and another containing a coastline.
    It computes the intersection points between the two geometries, visualizes them,
    and saves the resulting points as a new shapefile.

    Args:
        transect_path (str): Path to the shapefile containing transect lines.
        line_path (str): Path to the shapefile containing the coastline geometry.
        out_path (str): File path where the output shapefile of intersection points will be saved.

    Returns:
        None. Saves the intersection points as a new shapefile and displays a plot
        showing transects (red), coastline (blue), and intersection points (green).
    """
    transects = gpd.read_file(transect_path)
    coastline = gpd.read_file(line_path)
    points = coastline.unary_union.intersection(transects.unary_union)
    fig, ax = plt.subplots(figsize=(14,14))
    plot_points = gpd.GeoSeries(points)
    plot_points.plot(ax=ax, color='green')
    transects.plot(ax=ax, color='red')
    coastline.plot(ax=ax, color='blue')

    plt.show()

    plot_points.to_file(out_path)


def clip_shp(path_to_shp, boundary_geojson):
    """
    Clips a shapefile to a specified boundary polygon and saves the result.

    Args:
        path_to_shp (str): File path to the input shapefile (.shp).
        boundary_geojson (dict): Polygon in GeoJSON-like format used for clipping.

    Returns:
        None. Saves the clipped shapefile in the same folder with '_clipped' suffix.
    """
    path_name = os.path.dirname(path_to_shp) + "/"
    shp_name = os.path.basename(path_to_shp)
    shp_base, shp_extension = os.path.splitext(shp_name)
    shp_data = gpd.read_file(path_to_shp)

    poly_boundary = Polygon(shape(boundary_geojson))

    shp_clipped = gpd.clip(shp_data, poly_boundary)
    fig, ax = plt.subplots(figsize=(12,8))
    shp_data.plot(ax=ax, color='red')
    plot_shp = gpd.GeoSeries(poly_boundary)
    plot_shp.plot(ax=ax, color='green')
    plt.show()

    out_path = path_name + shp_base + "_clipped.shp"
    shp_clipped.to_file(out_path) 