import netCDF4 as nc
from pyproj import Transformer
import numpy as np
import geopandas as gpd
from scipy import interpolate
import os
import pandas as pd
from datetime import datetime, date, timedelta
import time
import math

# WKT Used to convert Lat/Lon to projection used in snap data
CRS_WKT = 'PROJCS["unnamed",GEOGCS["unnamed ellipse",DATUM["unknown",SPHEROID["unnamed",6370000,0]],PRIMEM["Greenwich",' \
          '0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",64],' \
          'PARAMETER["central_meridian",-152],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],' \
          'PARAMETER["false_northing",0],UNIT["Meter",1],EXTENSION["PROJ4","+units=m +proj=stere +lat_ts=64.0 ' \
          '+lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +wktext"]] '

# Center of region in lat/lon
REGION_CENTER = (66.0756, -162.7172)

# Filepaths for SNAP data

TSK_FILEPATH = '/usr/local/coastal/snap_processing/snap_data/tsk'
U10_FILEPATH = '/usr/local/coastal/snap_processing/snap_data/u10'
V10_FILEPATH = '/usr/local/coastal/snap_processing/snap_data/v10'
SEAICE_FILEPATH = '/usr/local/coastal/snap_processing/snap_data/seaice'
PSFC_FILEPATH = '/usr/local/coastal/snap_processing/snap_data/psfc'
T2_FILEPATH = '/usr/local/coastal/snap_processing/snap_data/t2'
TRANSECT_FILEPATH = '/usr/local/coastal/snap_processing/transect_data/WestChukchi_exposed_STepr_rates.shp'

# Filepath for I/O csv data
OUTPUT_FILEPATH = '/usr/local/coastal/snap_processing/snap_output/SNAP_daily_by_transect'

# Create global variables
global transformer_to_snap_proj
global transformer_to_lat_lon
global mod_center
global transect_points


# Function to read transects and return transect points for interpolation
def get_transect_points(filepath):
    transects = gpd.GeoDataFrame.from_file(filepath)

    # following limits transects to the ones around the area we have been looking at
    transects = transects[transects['BaselineID'] == 117]

    # This takes a representative point from each transect to use in interpolation
    transect_points = []
    for i in transects['geometry']:
        transect_points.append((i.representative_point().y, i.representative_point().x))

    print('Transects read and extracted')
    return transect_points, transects


# Function to create pandas dataframe to store all data in
def create_dataframe(transects, y):

    start_date = date(y, 1, 1)
    end_date = date(y, 12, 31)
    delta = timedelta(days=1)

    data = []

    print(f'Creating dataframe for {y}')

    while start_date <= end_date:

        for i in range(0, len(transects)):
            temp_transorder = transects.iloc[i]['TransOrder']
            temp_date = datetime.strftime(start_date, '%m-%d-%Y')
            temp_ID = str(temp_date) + '_' + str(temp_transorder)
            data.append((temp_date, temp_transorder, temp_ID, 0, 0, 0, 0, 0, 0))

        start_date += delta

    df = pd.DataFrame(data, columns=['date', 'transect', 'ID', 'tsk', 'u10', 'v10', 'seaice', 'psfc', 't2'])

    return df


# Function that will import SNAP data and return array with all datasets
# filepath can be either a nc file or a directory containing nc files
# All files in filepath must contain the same SNAP data variable (tsk, u10, v10, etc.)
def read_data(filepath):
    # Variable to store return tsk data
    data = []

    # Check if input is filepath
    if os.path.isdir(filepath):
        for filename in os.scandir(filepath):
            try:
                data.append(nc.Dataset(filepath + '/' + filename.name))
            except OSError:
                pass

    # Check if input is single file
    elif os.path.isfile(filepath):
        try:
            # Try to read in file
            data.append(nc.Dataset(filepath))
        except OSError:
            print('Input filepath ' + filepath + 'is not a directory or a file of nc format')
            return None

    # Invalid input
    else:
        print('Input filepath ' + filepath + ' was not valid')
        return None

    # Return dataset if not empty
    if not data:
        print('No nc files found in directory ' + filepath)
        return None
    else:
        return data


# Method to get the closest coordinate to REGION_CENTER
# Input should be a nc Dataset with SNAP data parameters
def get_closest_coords(ds):
    arr = np.array(ds['xc'])
    difference_array = np.absolute(arr - mod_center[0])
    closest_x = difference_array.argmin()

    arr = np.array(ds['yc'])
    difference_array = np.absolute(arr - mod_center[1])
    closest_y = difference_array.argmin()

    return closest_x, closest_y


# Crop SNAP surface temperature data to Deering region
# (3 x coordinates and 3 y coordinates to make the 9 points closest to Deering)
def crop_snap(ds, data, closest_x, closest_y):
    # Storage var declaration
    modified_data = np.empty((len(data), 3, 3))
    i_ct = 0
    ordered_coords = []

    for i in range(closest_y - 1, closest_y + 2):
        j_ct = 0
        for j in range(closest_x - 1, closest_x + 2):
            for k in range(0, len(data)):
                modified_data[k][i_ct][j_ct] = data[k][i][j]
            j_ct = j_ct + 1
            ordered_coords.append((ds['xc'][j].data.item(), ds['yc'][i].data.item()))
        i_ct = i_ct + 1

    return modified_data, ordered_coords


# Method to downscale snap data from hourly to daily
def downscale_data(modified_data):
    # Declare storage var
    downscaled_data = np.empty(((int(len(modified_data) / 24)), len(modified_data[0]), len(modified_data[0][0])))

    # Downscale from hourly to daily
    for i in range(0, (int(len(modified_data) / 24))):
        for j in range(len(modified_data[0])):
            for k in range(len(modified_data[0][0])):
                avg = 0
                start = i * 24
                for l in range(start, start + 24):
                    avg = avg + modified_data[l][j][k]
                downscaled_data[i][j][k] = (avg / 24)

    return downscaled_data


# Process to reformat downscaled_data to work with ordered_coords for interpolation
# Converts downscaled_data from arr[364][7][7] to arr[364][49]
def finalize_data(downscaled_data):
    finalized_data = []
    for i in range(len(downscaled_data)):
        temp_data = []
        for j in range(len(downscaled_data[i])):
            for k in range(len(downscaled_data[i][j])):
                temp_data.append(downscaled_data[i][j][k])
        finalized_data.append(temp_data)
    return finalized_data


# Method that processes and saves data
def process_data(data, data_name, dataframe, transects, ordered_coords, y):

    for ds in data:

        # Grab year from data
        year = int(ds.__dict__['reference_time'][0:4])

        # Check corresponding year
        if year == y:

            # Find X/Y coordinate on SNAP data grid closest do Deering center
            closest_x, closest_y = get_closest_coords(ds)

            # Separate variable from rest of SNAP data (Time cost ~20 sec/ds)
            current_data = ds[data_name][:]

            # Crop SNAP surface temperature data to Deering region
            if not ordered_coords:
                # Extract data with coords
                modified_data, ordered_coords = crop_snap(ds, current_data, closest_x, closest_y)

                # Transform data from SNAP projection to lat/lon
                for i in range(len(ordered_coords)):
                    ordered_coords[i] = transformer_to_lat_lon.transform(ordered_coords[i][0], ordered_coords[i][1])

            else:
                # Extract data without coords
                modified_data, grb = crop_snap(ds, current_data, closest_x, closest_y)

            # Downscale surface temperature from hourly to daily #
            downscaled_data = downscale_data(modified_data)

            # Reformat downscaled_data to work with ordered_coords for interpolation
            finalized_data = finalize_data(downscaled_data)

            # Initializing start date
            strt_date = date(int(year), 1, 1)

            # Interpolate this year's tsk data
            for i in range(0, len(finalized_data)):
                print(f'Processing {data_name} data: {i+1}/{len(finalized_data)}')

                # Get date from day
                day_num = str(i)
                day_num.rjust(3 + len(day_num), '0')

                res_date = strt_date + timedelta(days=int(day_num))
                final_date = res_date.strftime("%m-%d-%Y")

                interpolator = interpolate.LinearNDInterpolator(ordered_coords, finalized_data[i])
                interpolated_points = interpolator(transect_points)

                # Iterate through each transect
                # NOTE:For test purposes, replace len(transects) with a small number

                for j in range(0, len(transects)):
                    # Create ID identifier for date/transect combo
                    temp_ID = str(final_date) + '_' + str(transects.iloc[j]['TransOrder'])

                    # Update data value
                    dataframe.loc[dataframe['ID'] == temp_ID, data_name] = interpolated_points[j]

            return dataframe
    return dataframe


# Method to calculate wind speed and direction
def calculate_wind_data(dataframe):
    dataframe['Wind Speed'] = ((dataframe['u10'] ** 2) + (dataframe['v10'] ** 2)).pow(1. / 2)
    dataframe['Wind Direction'] = np.arcsin(((0 - dataframe['u10']) / dataframe['Wind Speed']).astype(np.float64))
    return dataframe


if __name__ == '__main__':

    # Import Transects and extract points
    transect_points, transects = get_transect_points(TRANSECT_FILEPATH)

    # Create transformer to convert from lat/long to snap projection
    transformer_to_snap_proj = Transformer.from_crs('epsg:4326', CRS_WKT)

    # Create transformer to convert from snap projection to lat/long
    transformer_to_lat_lon = Transformer.from_crs(CRS_WKT, 'epsg:4326')

    # Transform center of Deering to SNAP coordinates
    mod_center = transformer_to_snap_proj.transform(REGION_CENTER[0], REGION_CENTER[1])

    # Create variable to store ordered coordinates
    coords = []
    year_list = []

    # Import surface skin temperature data
    tsk_data = read_data(TSK_FILEPATH)
    if tsk_data:
        print('Read in tsk data')
        for ds in tsk_data:
            # Grab year from data
            year = int(ds.__dict__['reference_time'][0:4])
            if year not in year_list:
                year_list.append(year)

    # Import wind speed at 10m data (u component)
    u10_data = read_data(U10_FILEPATH)
    if u10_data:
        print('Read in u10 data')
        for ds in u10_data:
            # Grab year from data
            year = int(ds.__dict__['reference_time'][0:4])
            if year not in year_list:
                year_list.append(year)

    # Import wind speed at 10m data (v component)
    v10_data = read_data(V10_FILEPATH)
    if v10_data:
        print('Read in v10 data')
        for ds in v10_data:
            # Grab year from data
            year = int(ds.__dict__['reference_time'][0:4])
            if year not in year_list:
                year_list.append(year)

    # Import ice concentration data
    seaice_data = read_data(SEAICE_FILEPATH)
    if seaice_data:
        print('Read in seaice data')
        for ds in seaice_data:
            # Grab year from data
            year = int(ds.__dict__['reference_time'][0:4])
            if year not in year_list:
                year_list.append(year)

    # Import surface pressure data
    psfc_data = read_data(PSFC_FILEPATH)
    if psfc_data:
        print('Read in psfc data')
        for ds in psfc_data:
            # Grab year from data
            year = int(ds.__dict__['reference_time'][0:4])
            if year not in year_list:
                year_list.append(year)

    # Import 2m temperature data
    t2_data = read_data(T2_FILEPATH)
    if t2_data:
        print('Read in t2 data')
        for ds in t2_data:
            # Grab year from data
            year = int(ds.__dict__['reference_time'][0:4])
            if year not in year_list:
                year_list.append(year)

    year_list.sort()

    # Iterate through each year
    for y in year_list:

        # Create dataframe
        df = create_dataframe(transects, y)

        # Read and process tsk data #
        if tsk_data:
            print('Processing tsk data')
            df = process_data(tsk_data, 'tsk', df, transects, coords, y)
            print(f'{y} tsk data processed.')

        # Read and process u10 data #
        if u10_data:
            print('Processing u10 data')
            df = process_data(u10_data, 'u10', df, transects, coords, y)
            print(f'{y} u10 data processed.')

        # Read and process v10 data #
        if v10_data:
            print('Processing v10 data')
            df = process_data(v10_data, 'v10', df, transects, coords, y)
            print(f'{y} v10 data processed.')

        # Read and process seaice data #
        if seaice_data:
            print('Processing seaice data')
            df = process_data(seaice_data, 'seaice', df, transects, coords, y)
            print(f'{y} seaice data processed.')

        # Read and process psfc data #
        if psfc_data:
            print('Processing psfc data')
            df = process_data(psfc_data, 'psfc', df, transects, coords, y)
            print(f'{y} psfc data processed.')

        # Read and process t2 data #
        if t2_data:
            print('Processing t2 data')
            df = process_data(t2_data, 't2', df, transects, coords, y)
            print(f'{y} t2 data processed.')

        print('Calculating wind speed and direction')
        try:
            df = calculate_wind_data(df)
        except:
            print(f'Unable to perform wind calculations for year {y}. Please check dataframe for missing data.')
        print(f'{y} Wind data processed. Writing final dataframe')

        df.to_csv(OUTPUT_FILEPATH + f'_{y}.csv')
