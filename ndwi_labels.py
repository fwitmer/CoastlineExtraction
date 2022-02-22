import rasterio as rio
import shapely
from shapely.geometry import Polygon, shape
import geopandas as gpd
import os
from matplotlib import pyplot as plt

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
    

def get_ndwi_label(image_path, points_path, ksize = (500, 500)):
    # establish the ndwi calculation and copy metadata
    with rio.open(image_path, driver='GTiff') as src_raster:
        green = src_raster.read(2).astype(rio.float64)
        try: # may be 5-banded image, in which case NIR is on band 5
            nir = src_raster.read(5).astype(rio.float64)
        except: # otherwise it's on band 4
            nir = src_raster.read(4).astype(rio.float64)
        ndwi = (green - nir) / (green + nir)
        meta = src_raster.meta

    # create blank label layer

    pass


boundary = {'type': 'Polygon',
            'coordinates': [[[-162.8235626220703, 66.05622435812153], 
                             [-162.674560546875, 66.05622435812153], 
                             [-162.674560546875, 66.10883816429516],
                             [-162.8235626220703, 66.10883816429516], 
                             [-162.8235626220703, 66.05622435812153]]]}

# path_to_shp = "C:\\Users\\kjcar\\Downloads\\Deering_DSAS_Calculations\\WestChukchi_exposed_STepr_rates\\WestChukchi_exposed_STepr_rates.shp"
# clip_shp(path_to_shp, boundary)
