"""
Shapefile Creation Utilities
This module contains utility functions for creating shapefiles from raster data,
particularly for coastline extraction and contour processing.
"""

import os
import numpy as np
import rasterio as rio
import cv2
import matplotlib.pyplot as plt
import skimage
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from skimage import measure
import shapely
from shapely.geometry import mapping, MultiLineString, LineString
import fiona
from fiona.crs import from_epsg
import geopandas as gpd

__all__ = [
    'morph_transform',
    'extract_contours', 
    'contours_to_multilinestring',
    'save_shapefile',
    'coastline_shp_from_raster',
    'create_intersect_points',
    'save_and_process',
    'save_concatenated_ndwi_with_shapefile'
]


def morph_transform(dat, kwidth, kheight, outname=None):
    """
    Perform opening/closing morphological operations to reduce noise.
    
    Args:
        dat: Input array or file path
        kwidth: Kernel width
        kheight: Kernel height
        outname: Optional output file name
        
    Returns:
        Processed array or saves to file if outname provided
    """
    try:
        if isinstance(dat, str):
            dat = cv2.imread(dat, cv2.IMREAD_ANYDEPTH)
    except:
        pass  # Assume it's already an array
    
    kernel = np.ones((kheight, kwidth), np.uint8)
    opened = cv2.morphologyEx(dat, cv2.MORPH_OPEN, kernel)
    opened_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    if outname:
        return cv2.imwrite(outname, opened_closed)
    else:
        return opened_closed


def extract_contours(image, plot=False):
    """
    Extract contours from a binary image.
    
    Args:
        image: Binary image array
        plot: Whether to plot the contours
        
    Returns:
        List of contour arrays
    """
    contours = skimage.measure.find_contours(image, 0.5)
    
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()
    
    return contours


def contours_to_multilinestring(contours, transform):
    """
    Convert contours to shapely MultiLineString with proper georeferencing.
    
    Args:
        contours: List of contour arrays
        transform: Rasterio transform object
        
    Returns:
        Shapely MultiLineString geometry
    """
    poly = []
    for contour in contours:
        x, y = contour[:, 1], contour[:, 0]
        aa = rio.transform.xy(transform, y, x)
        poly.append(LineString([(i[0], i[1]) for i in zip(aa[0], aa[1])]))
    
    list_lstrings = [shapely.wkt.loads(p.wkt) for p in poly]
    return shapely.geometry.MultiLineString(list_lstrings)


def save_shapefile(multilinestring, filepath, crs_epsg=32603):
    """
    Save a MultiLineString geometry to a shapefile.
    
    Args:
        multilinestring: Shapely MultiLineString geometry
        filepath: Output file path
        crs_epsg: EPSG code for coordinate reference system
    """
    schema = {'geometry': 'MultiLineString', 'properties': {'id': 'int'}}
    crs = from_epsg(crs_epsg)
    
    with fiona.open(filepath, 'w', 'ESRI Shapefile', schema=schema, crs=crs) as c:
        c.write({'geometry': mapping(multilinestring), 'properties': {'id': 1}})


def coastline_shp_from_raster(file, plot=False):
    """
    Generate coastline shapefile from raster using morphological active contours.
    
    This function takes a preprocessed raster image and uses morphological active 
    contours to extract coastline boundaries. It performs noise reduction and 
    converts the result to a georeferenced shapefile.
    
    Args:
        file: Path to input raster file (e.g., NDWI processed image)
        plot: Whether to show plots during processing
        
    Returns:
        MultiLineString geometry and saves shapefile as "{filename}_Coast_Contour.shp"
        
    Example:
        >>> multi_line_contour = coastline_shp_from_raster("path/to/ndwi_image.tif", plot=True)
        >>> # This will create "ndwi_image_Coast_Contour.shp" in the same directory
    """
    iterations = 5
    
    with rio.open(file, driver='GTiff') as src:
        input_image = src.read(1).astype(rio.uint8)
        transform = src.transform

    # Apply morphological active contours
    init_level_set = checkerboard_level_set(input_image.shape)
    lvl_set = morphological_chan_vese(input_image, iterations, init_level_set=init_level_set, smoothing=1)
    
    # Reduce noise with morphological operations
    noise_reduced = morph_transform(lvl_set.astype(rio.uint8), 9, 9)
    
    # Extract contours
    contours = extract_contours(noise_reduced, plot=plot)
    
    # Convert to georeferenced MultiLineString
    multi_line_contour = contours_to_multilinestring(contours, transform)

    # Save shapefile
    raster_filepath = os.path.dirname(file)
    raster_filename = os.path.basename(file).split('.')[0]
    shapefile_path = os.path.join(raster_filepath, f"{raster_filename}_Coast_Contour.shp")
    save_shapefile(multi_line_contour, shapefile_path)

    return multi_line_contour


def create_intersect_points(transect_path, contour_path, out_path):
    """
    Calculate and map the intersection points of transects and coastline contours.
    
    Args:
        transect_path: Path to transect shapefile
        contour_path: Path to coastline contour shapefile
        out_path: Output path for intersection points
        
    Output: Writes plotted points to shapefile
    """
    transects = gpd.read_file(transect_path)
    transects = transects.to_crs(epsg=32603)  # Set Transect EPSG to maintain geographical continuity
    coastline = gpd.read_file(contour_path)
    
    points = coastline.unary_union.intersection(transects.unary_union)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    plot_points = gpd.GeoSeries(points)
    plot_points.plot(ax=ax, color='red')
    transects.plot(ax=ax, color='black')
    coastline.plot(ax=ax, color='blue')

    plt.show()

    plot_points.to_file(out_path)
    print('Saving Intersections to', out_path)


def save_and_process(array, profile, folder_path, image_name):
    """
    Save array as TIFF and NPY files, then process with coastline_shp_from_raster.
    
    Args:
        array: Input array
        profile: Rasterio profile
        folder_path: Output folder path
        image_name: Base name for output files
    """
    with rio.open(f"{folder_path}/{image_name}.tif", 'w', **profile) as dst:
        dst.write(array, 1)
    
    np.save(f"{folder_path}/{image_name}.npy", array)
    coastline_shp_from_raster(f"{folder_path}/{image_name}.tif")


def save_concatenated_ndwi_with_shapefile(ndwi_concatenated, profile, original_image_path):
    """
    Save concatenated NDWI array as TIFF and generate shapefile from it.
    
    Args:
        ndwi_concatenated: Concatenated NDWI array (binary water/land classification)
        profile: Rasterio profile from original image
        original_image_path: Path to original image for naming output files
    """
    # Get output directory and filename
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    
    # Use existing result_ndwi_labels directory
    output_dir = "result_ndwi_labels"
    
    # Create output filename for concatenated NDWI
    concatenated_filename = f"{base_name}_concatenated_ndwi"
    concatenated_path = os.path.join(output_dir, f"{concatenated_filename}.tif")
    
    # Update profile for single band binary image
    profile_updated = profile.copy()
    profile_updated.update(
        count=1,
        dtype=rio.uint8,
        nodata=0
    )
    
    # Save concatenated NDWI as TIFF
    print(f"Saving concatenated NDWI as: {concatenated_path}")
    with rio.open(concatenated_path, 'w', **profile_updated) as dst:
        dst.write(ndwi_concatenated.astype(rio.uint8), 1)
    
    # Generate shapefile from the concatenated NDWI
    print("Generating shapefile from concatenated NDWI...")
    try:
        multi_line_contour = coastline_shp_from_raster(concatenated_path, plot=False)
        print(f"Shapefile generated successfully: {base_name}_concatenated_ndwi_Coast_Contour.shp")
        print(f"Files saved in: {output_dir}/")
        return multi_line_contour
    except Exception as e:
        print(f"Error generating shapefile: {e}")
        return None 
    
    