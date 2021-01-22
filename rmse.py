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
import argparse
import matplotlib.pyplot as plt
import os


def calc_rmse(errs):
    errs = np.array(errs)
    rmse = np.sqrt(np.square(errs).mean())
    return rmse


def find_distances(transects, fst, snd, in_coast):
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
    epsilon = 2 ** -16

    try:
        a = os.path.basename(args.sf1)
        a = a[:-4]
        b = os.path.basename(in_coast)
        b = b[:-4]
        plt.title("Transect Distances: " + " " + a + ",\n" + b)
    except:
        plt.title("Transect Distances")

    plt.xlabel('Transect')
    plt.ylabel('Distance (m)')

    # for each transect find intersecting points in each gdf
    for i, transect in transects.iterrows():
        intersects[i] = {'fst': [], 'snd': []}

        for point in fst:
            dist = point.distance(transect.geometry)
            if dist < epsilon:
                intersects[i]['fst'].append(point)

        for point in snd:
            dist = point.distance(transect.geometry)
            if dist < epsilon:
                intersects[i]['snd'].append(point)

    # for each pair of points corresponding to a transect, caluctulate the distance between the points
    for k in intersects.keys():
        if len(intersects[k]['fst']) == len(intersects[k]['snd']) == 1:
            # dist = distance(intersects[k]['fst'][0].coords[0][::-1], intersects[k]['snd'][0].coords[0][::-1]).m
            # import pdb; pdb.set_trace()
            # dist = intersects[k]['fst'][0].distance(intersects[k]['snd'][0].geometry)
            dist = intersects[k]['fst'][0].distance(intersects[k]['snd'][0])
            distances.append(dist)

    return distances


def single_rmse(transects, in_coast, metadata):
    gdf1 = gpd.GeoDataFrame.from_file(args.sf1)
    gdf2 = gpd.GeoDataFrame.from_file(in_coast)

    # make crs consistent
    utm_zone_3n = 'EPSG:32603'
    transects = transects.to_crs(utm_zone_3n)
    if gdf1.crs == {}:
        gdf1.geometry.crs = utm_zone_3n
    else:
        gdf1 = gdf1.to_crs(utm_zone_3n)

    if gdf2.crs == {}:
        gdf2.geometry.crs = utm_zone_3n
    else:
        gdf2 = gdf2.to_crs(utm_zone_3n)

    # Find intersection points
    gdf1 = gdf1.unary_union.intersection(transects.unary_union)

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

    distances = find_distances(transects, gdf1, gdf2, in_coast)

    rmse = calc_rmse(distances)

    # Additional split RMSE calc by region if '--sr' flag True
    all_distances = []
    if args.sr:

        # Western Coastline Region
        region_1 = transects[transects['TransOrder'] >= 17443]

        # Northern Cliff Region
        region_2 = transects[transects['TransOrder'] < 17443]
        region_2 = region_2[region_2['TransOrder'] >= 17394]

        # Central Shoreline Region
        region_3 = transects[transects['TransOrder'] < 17394]
        region_3 = region_3[region_3['TransOrder'] >= 17370]

        # Town Shoreline Region
        region_4 = transects[transects['TransOrder'] < 17370]
        region_4 = region_4[region_4['TransOrder'] >= 17337]

        # East Shoreline and Cliff Region
        region_5 = transects[transects['TransOrder'] < 17337]

        coast1 = gpd.GeoDataFrame.from_file(args.sf1)
        coast2 = gpd.GeoDataFrame.from_file(in_coast)

        utm_zone_3n = 'EPSG:32603'
        if coast1.crs == {}:
            coast1.geometry.crs = utm_zone_3n
        else:
            coast1 = coast1.to_crs(utm_zone_3n)

        if coast2.crs == {}:
            coast2.geometry.crs = utm_zone_3n
        else:
            coast2 = coast2.to_crs(utm_zone_3n)

        regions = [region_1, region_2, region_3, region_4, region_5]

        # Calculate intersections for each region
        intersections_1 = []
        intersections_2 = []
        for i in regions:
            intersections_1.append(coast1.unary_union.intersection(i.unary_union))
            intersections_2.append(coast2.unary_union.intersection(i.unary_union))

        for i in intersections_1:
            # filter by date for dataframe from first shapefile
            if args.d1:
                i = i[i[args.col_header1] == args.d1]

        for i in intersections_2:
            # filter by date for dataframe from second shapefile
            if args.d2:
                i = i[i[args.col_header2] == args.d2]

        # Find distances, calculate RMSE
        RMSEs = []
        for i in range(0, len(regions)):
            distances = find_distances(regions[i], intersections_1[i], intersections_2[i], in_coast)
            all_distances.append(distances)
            RMSEs.append(calc_rmse(distances))

    # If arg g called, generate graph
    if args.g:
        # Generate single graph if sr flag not set
        if not args.sr:
            xs = []
            ct = 1
            for distance in distances:
                xs.append(ct)
                ct = ct + 1
            plt.plot(xs, distances)
            result_file = args.o

        # Generate area-separated graph if sr flag set
        else:
            area_index = ["Western Coastline Region", "Northern Cliff Region", "Central Shoreline Region",
                          "Town Shoreline Region", "East Shoreline and Cliff Region"]
            i = 0
            ct = 1
            for distances in all_distances:
                xs = []
                for distance in distances:
                    xs.append(ct)
                    ct = ct + 1
                plt.plot(xs, distances, label=area_index[i])
                plt.legend()
                i = i + 1

        # Save plot
        result_file = args.o

        try:
            a = os.path.basename(args.sf1)
            a = a[:-4]
            b = os.path.basename(in_coast)
            b = b[:-4]
            result_file = result_file if result_file[
                                         :-4] == '.png' else result_file + '\Transect_Plot_' + a + '_' + b + '.png'
        except:
            result_file = result_file if result_file[:-4] == '.png' else result_file + '\Transect_Plot.png'

        print('Saving plot to', result_file)
        plt.savefig(result_file)

    result_file = args.o
    try:
        a = os.path.basename(args.sf1)
        a = a[:-4]
        b = os.path.basename(in_coast)
        b = b[:-4]
        result_file = result_file if result_file[
                                     :-4] == '.txt' else result_file + '\RMSE_Calculations_' + a + '_' + b + '.txt'
    except:
        result_file = result_file if result_file[:-4] == '.txt' else result_file + '\RMSE_Calculations.txt'
    with open(result_file, 'w+') as out:
        print('writing to file', result_file)
        out.write(f'source file 1: {args.sf1}\n')
        if args.d1:
            out.write(f'source file 1 date: {args.d1}\n')
        out.write(f'source file 2: {in_coast}\n')
        if args.d2:
            out.write(f'source file 2 date: {args.d2}\n')
        out.write(f'Complete Shoreline RMSE: {rmse} (m)\n')
        if args.sr:
            out.write(f'RMSE for West Coastline Region (m): {RMSEs[0]}\n')
            out.write(f'RMSE for Northern Cliff Region (m): {RMSEs[1]}\n')
            out.write(f'RMSE for Central Shoreline Region (m): {RMSEs[2]}\n')
            out.write(f'RMSE for Town Shoreline Region (m): {RMSEs[3]}\n')
            out.write(f'RMSE for Eastern Shoreline/Cliff Region (m): {RMSEs[4]}\n')

    # Save info for meta graph
    if args.sr:
        b = os.path.basename(in_coast)
        b = b[:-4]
        metadata.append((b, RMSEs, rmse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transects', required=True, help='Shapefile containing coast transects.')
    parser.add_argument('-sf1', required=True, help='Baseline Coast to read points from.')
    parser.add_argument('-d1', help='Date to filter points by in sf1. Use same date format as source file.')
    parser.add_argument('--col-header1',
                        help='String that is the column header for the date column in the dataframe made from sf1. Required if d1 is present.')
    parser.add_argument('-sf2', required=True,
                        help='List of coasts to calculate rmse for. Seperate multiple coasts with a comma (coast1,coast2,...)')
    parser.add_argument('-d2', help='Date to filter points by in sf2. Use same date format as source file.')
    parser.add_argument('--col-header2',
                        help='String that is the column header for the date column in the dataframe made from sf2. Reqired if d2 is present.')
    parser.add_argument('-o', required=True, help='Name of file to write results to.')
    parser.add_argument('--r', help='Call flag if river mouth transects should be excluded from RMSE calculation')
    parser.add_argument('--sr', help='Call flag to split RMSE calculation by region')
    parser.add_argument('--g', help='Set true to save a graphic depicting transect intersection distances')
    parser.add_argument('--mg', help='Set True to save a meta graph comparing all input coastlines')
    args = parser.parse_args()

    # Set size of plots
    plt.figure(figsize=(15, 10))

    transects = gpd.GeoDataFrame.from_file(args.transects)
    # following limits transects to the ones around the area we have been looking at
    transects = transects[transects['BaselineID'] == 117]

    # If '--r' flag set, remove river mouth transects
    if args.r:
        removal_ids = [17336, 17335, 17334, 17333, 17332]
        for removal_id in removal_ids:
            transects = transects[transects['TransOrder'] != removal_id]

    all_coasts = args.sf2.split(',')

    if args.mg and not args.sr:
        raise AssertionError('Metagraph cannot be created without --sr=True')

    metadata = []
    for c in all_coasts:
        single_rmse(transects, c, metadata)
        plt.clf()

    # Create meta graph
    if args.mg:

        # Establish titles
        try:
            a = os.path.basename(args.sf1)
            a = a[:-4]
        except:
            a = 'Baseline Coast'

        title = 'RMSE Comparisons Relative To ' + a
        plt.title(title)
        plt.ylabel('RMSE (m)')

        area_index = ["Western Coastline Region", "Northern Cliff Region", "Central Shoreline Region",
                      "Town Shoreline Region", "East Shoreline and Cliff Region"]

        # Plot points
        print(metadata)
        for data in metadata:
            srt = round(data[2], 2)
            l = str(data[0]) + (' (Overall RMSE: ' + str(srt) + 'm)')
            plt.plot(area_index, data[1], label=l)
        plt.legend()

        # Save plot
        result_file = args.o
        try:
            result_file = result_file if result_file[
                                         :-4] == '.png' else result_file + '\\' + a + '_RMSE_Comparisons.png'
        except:
            result_file = result_file if result_file[:-4] == '.png' else result_file + '\\Transect_Plot.png'

        print('Saving plot to', result_file)
        plt.savefig(result_file)
