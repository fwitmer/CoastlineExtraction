import rasterio as rio
from rasterio import mask
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.io import MemoryFile
import shapely
from shapely.geometry import Polygon, shape
import geopandas as gpd
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
# Function to create transect points
def create_transect_points(transect_path, line_path, out_path):
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

# Function to clip shapefile
def clip_shp(path_to_shp, boundary_geojson):
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
# Function to get NDWI label using sliding window approach
def get_ndwi_label(image_path, points_path, ksize=100, window_size=50):
    # Open raster image 
    with rio.open(image_path, driver='GTiff') as src_raster:
        green = src_raster.read(2).astype(np.float32)
        nir_num = src_raster.count  
        nir = src_raster.read(nir_num).astype(np.float32)
        np.seterr(divide='ignore', invalid='ignore')
        ndwi = (green - nir) / (green + nir)
        ndwi[np.isnan(ndwi)] = 0
        ndwi_profile = src_raster.profile
        label = np.zeros((src_raster.height, src_raster.width)).astype(np.uint8)
        agg_mask = np.zeros((src_raster.height, src_raster.width)).astype(np.uint8)
        src_CRS = src_raster.crs
        pixel_size = abs(src_raster.transform[0])
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot NDWI image
        ndwi_image = ax.imshow(ndwi, cmap='gray', extent=(0, src_raster.transform.a * ndwi.shape[1], src_raster.transform.e * ndwi.shape[0], 0))
        ax.set_title("NDWI")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ndwi_colorbar = plt.colorbar(ndwi_image, ax=ax, label="NDWI Value")
        ndwi_colorbar.set_label("NDWI Value", rotation=90)
        plt.show()
    # Read shapefile points
    points_shp = gpd.read_file(points_path)
    points_shp.plot(ax=plt.gca(), color='red', edgecolor='black', alpha=0.5)
    plt.title("points_shp ")
    plt.show()
    # Set CRS and plot points
    points_geom = points_shp.geometry
    points_geom.plot()
    plt.title("points_geom ")
    plt.show()

    points_geom = points_geom.set_crs(epsg=4326)
    points_geom = points_geom.to_crs(src_CRS)
    points_geom.plot()
    plt.title("points_geom in raster CRS")
    plt.show()

    otsu_thresholds = []
    skipped = 0

    ndwi_profile.update(count=1, nodata=0, dtype=rio.float32)

    for multipoint in points_geom:
        for point in multipoint.geoms:
            buffer = point.buffer(ksize * pixel_size, cap_style=3)
            buffer_series = gpd.GeoSeries(buffer)
            for window in buffer_series.buffer(window_size * pixel_size):
               with MemoryFile() as memfile:
                   with memfile.open(**ndwi_profile) as mem_data:
                       mem_data.write_band(1, ndwi)
                   with memfile.open() as dataset:
                       out_image, out_transform = mask(dataset, shapes=[window], nodata=0, crop=False)
                       out_image = out_image[0]
                       out_image = (out_image * 127) + 128
                       out_image = out_image.astype(np.uint8)
                       
                       if out_image.shape[0] < 200 or out_image.shape[1] < 200:
                           skipped += 1
                           continue
                       else: 
                           otsu_threshold, image_result = cv2.threshold(out_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                           otsu_thresholds.append(otsu_threshold)
                           mask_values = np.ma.getmask(out_image).astype(np.uint8)
                           agg_mask += mask_values
                           threshold_window = np.where(out_image >= otsu_threshold, 1, 0).astype(np.uint8)
                           label += threshold_window.astype(np.uint8)
                            
    print("Total number of valid thresholds: {}".format(len(otsu_thresholds)))
    print("Number of skipped windows: {}".format(skipped))
    print("Actual thresholds (8-bit unsigned): \n{}".format(otsu_thresholds))
    print("Average threshold value (8-bit unsigned): {}".format(np.mean(otsu_thresholds)))
    print("Average threshold value (-1 to 1 NDWI range): {}".format((np.mean(otsu_thresholds) - 128) / 127))

    print("\nLabel max: {}".format(np.nanmax(label)))
    print("Label min: {}".format(np.nanmin(label)))
    plt.imshow(label)
    plt.title("Label")
    plt.show()

    print("\nMask max: {}".format(np.nanmax(agg_mask)))
    print("Mask min: {}".format(np.nanmin(agg_mask)))
    plt.imshow(agg_mask)
    plt.title("Aggregated Mask")
    plt.show()

    mean_threshold = np.mean(otsu_thresholds) + 10
    ndwi_8bit = ((ndwi * 127) + 128).astype(np.uint8)
    ndwi_classified = np.where(ndwi_8bit >= mean_threshold, 1, 0)
    plt.imshow(ndwi_classified, cmap="gray")
    plt.title("NDWI Classified")
    plt.show()

    points_geom.plot(color='blue', markersize=5)
    plt.title("points_geom in raster CRS-2")
    plt.show()


boundary = {'type': 'Polygon',
            'coordinates': [[[-162.8235626220703, 66.05622435812153], 
                             [-162.674560546875, 66.05622435812153], 
                             [-162.674560546875, 66.10883816429516],
                             [-162.8235626220703, 66.10883816429516], 
                             [-162.8235626220703, 66.05622435812153]]]}


image_path = "sample_data/PlanetLabs/20160909_213103_0e19_3B_AnalyticMS_SR_clip.tif"
points_path = "USGS_Coastlines/Deering_transect_points_2016_fw.shp"
get_ndwi_label(image_path, points_path)

# path_to_shp = "C:\\Users\\kjcar\\Downloads\\Deering_DSAS_Calculations\\WestChukchi_exposed_STepr_rates\\WestChukchi_exposed_STepr_rates.shp"
# clip_shp(path_to_shp, boundary)

