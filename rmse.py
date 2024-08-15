import sys
import numpy as np
import geopandas as gpd
import argparse
import matplotlib.pyplot as plt
import os



transects_lines_path = "D:/GSoC2024/experiments/coastlines_and_transects/TransectLines/WestChukchi_exposed_STepr_rates/WestChukchi_exposed_STepr_rates.shp"
predicted_shp_path = "D:/GSoC2024/experiments/output/attempt1/2016-09-06/ndwi_labelled/concatenated_ndwi_smoothing_Coast_Contour.shp"
true_shp_path = "D:/GSoC2024/experiments/coastlines_and_transects/PlanetCoastline/09_09_2016/9_9_16_PlanetCoastline.shp"


EPSILON = 2 ** -16 # Small epsilon value to find points very close to the transect
UTM_ZONE_3N = 'EPSG:32603' # EPSG code for UTM zone 3N projection
RIVER_REMOVAL = True


def calc_rmse(errs):
    errs = np.array(errs)
    rmse = np.sqrt(np.square(errs).mean())
    return rmse


def find_distances(transects, fst, snd):
    distances = []
    
    for transect in transects.itertuples():
        
        transect_geom = transect.geometry
        # Find points in fst and snd that are within epsilon distance from the transect
        fst_points = [point for point in fst.geoms if point.distance(transect_geom) < EPSILON]
        snd_points = [point for point in snd.geoms if point.distance(transect_geom) < EPSILON]

        # Ensure there's exactly one intersecting point in each list
        if len(fst_points) == len(snd_points) == 1:
            fst_point = fst_points[0]
            snd_point = snd_points[0]
            # Calculate the distance between the intersecting points
            dist = fst_point.distance(snd_point) 
            distances.append(dist)
    return distances


def calc_transects_rmse(transects, true_shap_path, predicted_shp_path):
    # Load true and predicted shapefiles into GeoDataFrames
    gdf1 = gpd.GeoDataFrame.from_file(true_shap_path)
    gdf2 = gpd.GeoDataFrame.from_file(predicted_shp_path)
    
    
    # IDs of transects to be removed
    if RIVER_REMOVAL:
        removal_ids = [17336, 17335, 17334, 17333, 17332]
        transects = transects[~(transects['TransOrder'].isin(removal_ids))]
        
    # Reproject the transects and shapefiles to UTM zone 3N
    transects = transects.to_crs(UTM_ZONE_3N)
    gdf1 = gdf1.to_crs(UTM_ZONE_3N)
    gdf2 = gdf2.to_crs(UTM_ZONE_3N)
        
        
    # Find the intersection of the true and predicted shapefile with transects
    gdf1 = gdf1.unary_union.intersection(transects.unary_union)
    gdf2 = gdf2.unary_union.intersection(transects.unary_union)
        
    distances = find_distances(transects, gdf1, gdf2)
    rmse = calc_rmse(distances)
        
    return distances, rmse
    
    

transects = gpd.GeoDataFrame.from_file(transects_lines_path)
transects = transects[transects['BaselineID'] == 117]
distances_mean, rmse_mean = calc_transects_rmse(transects, true_shp_path, predicted_shp_path)


# Western Coastline Region
region_1 = transects[transects['TransOrder'] >= 17443]
distances_region1, rmse_region1 = calc_transects_rmse(region_1, true_shp_path, predicted_shp_path)


# Northern Cliff Region
region_2 = transects[(transects['TransOrder'] < 17443) & (transects['TransOrder'] >= 17394)]
distances_region2, rmse_region2 = calc_transects_rmse(region_2, true_shp_path, predicted_shp_path)


# Central Shoreline Region
region_3 = transects[(transects['TransOrder'] < 17394) & (transects['TransOrder'] >= 17370)]
distances_region3, rmse_region3 = calc_transects_rmse(region_3, true_shp_path, predicted_shp_path)


# Town Shoreline Region
region_4 = transects[(transects['TransOrder'] < 17370) & (transects['TransOrder'] >= 17337)]
distances_region4, rmse_region4 = calc_transects_rmse(region_4, true_shp_path, predicted_shp_path)


# East Shoreline and Cliff Region
region_5 = transects[transects['TransOrder'] < 17337]
distances_region5, rmse_region5 = calc_transects_rmse(region_5, true_shp_path, predicted_shp_path)