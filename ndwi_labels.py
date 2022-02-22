import rasterio as rio
from rasterio import mask
from rasterio.plot import show
from rasterio.mask import mask
import shapely
from shapely.geometry import Polygon, shape
import geopandas as gpd
import os
from matplotlib import pyplot as plt
import numpy as np

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
    

def get_ndwi_label(image_path, points_path, ksize = 100):
    # establish the ndwi calculation and copy metadata
    with rio.open(image_path, driver='GTiff') as src_raster:
        green = src_raster.read(2).astype(rio.float64)
        nir_num = src_raster.count  # adjusting NIR band to 4 or 5 band images
        nir = src_raster.read(nir_num)
        np.seterr(divide='ignore', invalid='ignore')
        ndwi = (green - nir) / (green + nir)
        meta = src_raster.meta
        # blank label layer
        label = np.zeros((src_raster.height, src_raster.width))
        src_CRS = src_raster.crs
        # getting pixel size for correct calculation of buffer
        pixel_size = abs(src_raster.transform[0])
        figs, ax = plt.subplots(figsize=(12, 8))
        show(ndwi, transform=src_raster.transform, ax=ax, cmap='gray')

        

    # preparing points for creating label masks
    points_shp = gpd.read_file(points_path)
    points_geom = points_shp.geometry
    points_geom = points_geom.set_crs(epsg=4326)
    points_geom = points_geom.to_crs(src_CRS)
    
    # processing each point found
    for multipoint in points_geom:
        for point in multipoint.geoms:
            buffer = point.buffer(ksize * pixel_size, cap_style=3)
            buffer_series = gpd.GeoSeries(buffer)
            buffer_series.exterior.plot(ax=ax, color='red', linewidth=1)
            # out_image, out_transform = mask(ndwi, buffer)
            # if out_image.shape[0] < 100 or out_image.shape[1] < 100:
            #     print("Masked image was too small.")
            # else:
            #     print("Masked image was correct size.")
            
    points_geom.plot(ax=ax, color='blue', markersize=5)
    plt.show()
    pass


boundary = {'type': 'Polygon',
            'coordinates': [[[-162.8235626220703, 66.05622435812153], 
                             [-162.674560546875, 66.05622435812153], 
                             [-162.674560546875, 66.10883816429516],
                             [-162.8235626220703, 66.10883816429516], 
                             [-162.8235626220703, 66.05622435812153]]]}
image_path = "data/input/369619_2016-09-04_RE2_3A_Analytic_SR_clip.tif"
points_path = "data/Deering_transect_points_2016.shp"
get_ndwi_label(image_path, points_path)

# path_to_shp = "C:\\Users\\kjcar\\Downloads\\Deering_DSAS_Calculations\\WestChukchi_exposed_STepr_rates\\WestChukchi_exposed_STepr_rates.shp"
# clip_shp(path_to_shp, boundary)
