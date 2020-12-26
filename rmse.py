'''
Calculates the RMSE between two sets of points that intersect with
a common set of transects. Results are written to a csv file.

Run with two shapefiles without specifying by date:
    python3 rmse.py --transects transect_shapefile -sf1 first_shapefile -sf2 second_shapefile -o results_file

Run with two shapefiles but filter features by date in the first shapefile
    python3 rmse.py --transects transect_shapefile -sf1 first_shapefile -d1 date -ch1 date_header -sf2 second_shapefile -o results_file
    the date argument should follow the same formatting as found in the source shapefile
    the date_header argument needs to match the name of the attribute in the source file that holds the date
'''


import sys
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
from geopy.distance import distance
import argparse


def calc_rmse(errs):
    errs = np.array(errs)
    rmse = np.sqrt(np.square(errs).mean())
    return rmse


def find_distances(transects, fst, snd):
    '''
    Finds the distances between pairs of points from different coastlines that
    intesect a common transect.

    PARAMETERS:
        transects: a GeoDataFrame of multilines representing the coastal transects
        fst, snd: GeoDataFrames containing 1 or more shapely points

    RETURNS:
        a list of distances in meters between the corresonding points in the two
        sets of coordinates
    '''

    distances = []
    intersects = {}
    epsilon = 2**-16

    # for each transect find intersecting points in each gdf
    for i, transect in transects.iterrows():
        intersects[i] = {'fst':[], 'snd':[]}

        for j, point in fst.iterrows():
            dist = point.geometry.distance(transect.geometry)
            if dist < epsilon:
                intersects[i]['fst'].append(point)

        for j, point in snd.iterrows():
            dist = point.geometry.distance(transect.geometry)
            if dist < epsilon:
                intersects[i]['snd'].append(point)

    # for each pair of points corresponding to a transect, caluctulate the distance between the points
    for k in intersects.keys():
        if len(intersects[k]['fst']) == len(intersects[k]['snd']) == 1:
            import pdb; pdb.set_trace()
            dist = distance(intersects[k]['fst'][0].geometry.coords[0][::-1], intersects[k]['snd'][0].geometry.coords[0][::-1]).m
            #dist = intersects[k]['fst'][0].geometry.distance(intersects[k]['snd'][0].geometry)
            distances.append(dist)

    return distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transects', required=True, help='Shapefile containing coast transects.')
    parser.add_argument('-sf1', required=True, help='First shapefile to read points from.')
    parser.add_argument('-d1', help='Date to filter points by in sf1. Use same date format as source file.')
    parser.add_argument('--col-header1', help='String that is the column header for the date column in the dataframe made from sf1. Required if d1 is present.')
    parser.add_argument('-sf2', required=True, help='Second shapefile to read points from.')
    parser.add_argument('-d2', help='Date to filter points by in sf2. Use same date format as source file.')
    parser.add_argument('--col-header2', help='String that is the column header for the date column in the dataframe made from sf2. Reqired if d2 is present.')
    parser.add_argument('-o', required=True, help='Name of file to write results to.')
    args = parser.parse_args()

    transects = gpd.GeoDataFrame.from_file(args.transects)
    # following limits transects to the ones around the area we have been looking at
    transects = transects[transects['BaselineID'] == 117]

    gdf1 = gpd.GeoDataFrame.from_file(args.sf1)
    gdf1 = gdf1.unary_union.intersection(transects.unary_union)

    gdf2 = gpd.GeoDataFrame.from_file(args.sf2)
    gdf2 = gdf2.unary_union.intersection(transects.unary_union)

    # filter by date for dataframe from first shapefile
    if args.d1:
        if args.col_header1 is None:
            parser.error("-d1 requires --col-header1.")
        else:
            gdf1 = gdf1[gdf1[args.col_header1] == args.d1]

    # filter by date for dataframe from second shapefile
    if args.d2:
        if args.col_header2 is None:
            parser.error("-d2 requires --col-header2.")
        else:
            gdf2 = gdf2[gdf2[args.col_header2] == args.d2]

    distances = find_distances(transects, gdf1, gdf2)

    rmse = calc_rmse(distances)

    result_file = args.o
    result_file = result_file if result_file[:-4] == '.csv' else result_file + '.csv'
    with open(result_file, 'w') as out:
        out.write(f'source file 1,{args.sf1}\n')
        if args.d1:
            out.write(f'source file 1 date,{args.sh1}\n')
        out.write(f'source file 2,{sf2}\n')
        if args.d2:
            out.write(f'source file 2 date,{args.sf2}\n')
        out.write(f'RMSE (m),{rmse}\n')
