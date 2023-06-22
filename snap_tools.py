# Jack Carroll
# 12/28/21
# This file contains a number of helper functions used as part of the SNAP Process
# compare_seaice() generates a plot comparing Yearly seaice values
# narrow_csv_to_transect narrows an input csv file to a single transect
# merge_t2 merged the t2_only SNAP output files with the rest of the data
# merge_all_dataframes merged all of the individual yearly SNAP dataframes into one large dataframe
# year_gap_comparison creates many plots to both compare the divide between 2005 and 2006 SNAP data
# as well as the variation between 2005, 2006 and 2007 SNAP data
# avg_annual_temp calculates the average temperature for each year of SNAP data and creates a plot
# transect_comparison plots the difference between many different transects on a daily basis over the course of a year

import numpy as np
import os
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt

DATA_INPUT_DIRECTORY = '/usr/local/coastal/snap_processing/snap_output/'
SNAP_COMBINED_DIRECTORY = 'SNAP_daily_by_transect_combined.csv'
STATION_DATA_DIRECTORY = 'DeeringGHCN-D.csv'

# Helper function for the compare_seaice method used to filter out the
# t2_only files from the rest of the SNAP Processing output files
def read_data(filepath):

    data = []

    for filename in os.scandir(filepath):

        if 't2' not in str(filename):

            data.append(pd.read_csv(filename))

    return data

# Generates a plot comparing Yearly seaice values
# This method is OUTDATED, as it is based around reading each year's input file individually
# rather than reading in the completed SNAP output file
def compare_seaice():

    print('Comparing Seaice')

    # Read data
    print('Reading Data')
    data = read_data(DATA_INPUT_DIRECTORY)

    plt.xlabel('Year')
    plt.xticks(fontsize=6, rotation=90)
    plt.ylabel('Total Seaice')
    plt.title('Yearly Seaice Comparison')

    years = []
    vals = []

    print('Plotting')

    for d in data:

        try:
            year = d.iat[3, 1]
            year = year[-4:]
            years.append(year)

            seaice_sum = d['seaice'].sum()
            vals.append(seaice_sum)
        except:
            print('Please ensure input directory is empty aside from valid data. Errors may occur.')

    years, vals = (list(t) for t in zip(*sorted(zip(years, vals))))
    plt.plot(years, vals)
    plt.savefig('Yearly_Seaice_Comparison.png')

# This method takes in an input csv containing data over many transects and returns the csv file
# filtered to a single transect
def narrow_csv_to_transect(transect, filepath):

    df = pd.read_csv(filepath)

    filtered_df = df.loc[df['transect'] == transect]

    filtered_df.to_csv('modified_csv.csv')

# This method is a helper function tomerged the t2_only SNAP output files with the rest of the data
def merge_t2(data_fp_arr, t2_fp_arr):

    if len(data_fp_arr) != len(t2_fp_arr):
        print('Unequal array lengths')
        return

    i = 0
    while i < len(data_fp_arr):
        df = pd.read_csv(data_fp_arr[i])
        df = df.iloc[:, 1:]
        t2_df = pd.read_csv(t2_fp_arr[i])
        t2_df = t2_df.iloc[:, 1:]

        t2_data = t2_df['t2']
        df.insert(9, 't2z', t2_data)

        save_str = (data_fp_arr[i])[:-4] + '_combined.csv'
        df.to_csv(save_str)

        i = i + 1

# This method merged the t2_only SNAP output files with the rest of the data on a yearly basis
def merge_all_t2s():

    data_fps = []
    t2_fps = []
    for i in range(1970, 2025):
        data_fps.append(f'SNAP_daily_by_transect_{i}.csv')
        t2_fps.append(f'SNAP_daily_by_transect_{i}_t2_only.csv')

    merge_t2(data_fps, t2_fps)

# This method merged all of the individual yearly SNAP datarames into one large dataframe
def merge_all_dataframes():

    data_fps = []
    for i in range(1971, 2026):
        data_fps.append(f'SNAP_daily_by_transect_{i}_combined.csv')

    for i in range(2026, 2101):
        data_fps.append(f'SNAP_daily_by_transect_{i}.csv')

    combined_df = pd.read_csv('SNAP_daily_by_transect_1970_combined.csv')
    combined_df = combined_df.iloc[:, 1:]

    for i in data_fps:
        try:
            temp_df = pd.read_csv(i)
            temp_df = temp_df.iloc[:, 1:]
            combined_df = combined_df.append(temp_df)
        except:
            print(f'Could not find {i}')

    combined_df.to_csv('SNAP_daily_by_transect_combined.csv', index=False)

# This method creates many plots to both compare the divide between 2005 and 2006 SNAP data
# as well as the variation between 2005, 2006 and 2007 SNAP data
def year_gap_comparison():

    station_data = pd.read_csv(STATION_DATA_DIRECTORY)
    df = pd.read_csv(SNAP_COMBINED_DIRECTORY)

    days_2005 = []
    days_2006 = []
    days_2007 = []
    tsk_vals_2005 = []
    tsk_vals_2006 = []
    tsk_vals_2007 = []
    t2_vals_2005 = []
    t2_vals_2006 = []
    t2_vals_2007 = []
    u10_vals_2005 = []
    u10_vals_2006 = []
    u10_vals_2007 = []
    v10_vals_2005 = []
    v10_vals_2006 = []
    v10_vals_2007 = []
    seaice_vals_2005 = []
    seaice_vals_2006 = []
    seaice_vals_2007 = []
    psfc_vals_2005 = []
    psfc_vals_2006 = []
    psfc_vals_2007 = []


    start_2005 = date(2005, 1, 1)
    end_2005 = date(2005, 12, 31)
    start_2006 = date(2006, 1, 1)
    end_2006 = date(2006, 12, 31)
    start_2007 = date(2007, 1, 1)
    end_2007 = date(2007, 12, 31)
    delta = timedelta(days=1)

    df = df.set_index('ID')

    while start_2005 < end_2005:

        month = str(start_2005.month).zfill(2)
        day = str(start_2005.day).zfill(2)

        tsk_vals_2005.append(df.loc[f'{month}-{day}-{start_2005.year}_17222', 'tsk'])
        t2_vals_2005.append(df.loc[f'{month}-{day}-{start_2005.year}_17222', 't2'])
        u10_vals_2005.append(df.loc[f'{month}-{day}-{start_2005.year}_17222', 'u10'])
        v10_vals_2005.append(df.loc[f'{month}-{day}-{start_2005.year}_17222', 'v10'])
        seaice_vals_2005.append(df.loc[f'{month}-{day}-{start_2005.year}_17222', 'seaice'])
        psfc_vals_2005.append(df.loc[f'{month}-{day}-{start_2005.year}_17222', 'psfc'])

        start_2005 += delta

        days_2005.append(f'{month}-{day}')

    while start_2006 < end_2006:

        month = str(start_2006.month).zfill(2)
        day = str(start_2006.day).zfill(2)

        tsk_vals_2006.append(df.loc[f'{month}-{day}-{start_2006.year}_17222', 'tsk'])
        t2_vals_2006.append(df.loc[f'{month}-{day}-{start_2006.year}_17222', 't2'])
        u10_vals_2006.append(df.loc[f'{month}-{day}-{start_2006.year}_17222', 'u10'])
        v10_vals_2006.append(df.loc[f'{month}-{day}-{start_2006.year}_17222', 'v10'])
        seaice_vals_2006.append(df.loc[f'{month}-{day}-{start_2006.year}_17222', 'seaice'])
        psfc_vals_2006.append(df.loc[f'{month}-{day}-{start_2006.year}_17222', 'psfc'])

        start_2006 += delta

        days_2006.append(f'{month}-{day}')

    while start_2007 < end_2007:

        month = str(start_2007.month).zfill(2)
        day = str(start_2007.day).zfill(2)

        tsk_vals_2007.append(df.loc[f'{month}-{day}-{start_2007.year}_17222', 'tsk'])
        t2_vals_2007.append(df.loc[f'{month}-{day}-{start_2007.year}_17222', 't2'])
        u10_vals_2007.append(df.loc[f'{month}-{day}-{start_2007.year}_17222', 'u10'])
        v10_vals_2007.append(df.loc[f'{month}-{day}-{start_2007.year}_17222', 'v10'])
        seaice_vals_2007.append(df.loc[f'{month}-{day}-{start_2007.year}_17222', 'seaice'])
        psfc_vals_2007.append(df.loc[f'{month}-{day}-{start_2007.year}_17222', 'psfc'])

        start_2007 += delta

        days_2007.append(f'{month}-{day}')

    plt.ylabel('Daily tsk')
    plt.title('Yearly tsk Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, len(days_2005), 12), fontsize=6, rotation=90)

    plt.plot(days_2005, tsk_vals_2005, label='2005')
    plt.plot(days_2006, tsk_vals_2006, label='2006')
    plt.plot(days_2007, tsk_vals_2007, label='2007')
    plt.legend()
    plt.savefig('Yearly_tsk_Comparison.png')

    plt.close()

    plt.ylabel('Daily t2')
    plt.title('Yearly t2 Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, len(days_2005), 12), fontsize=6, rotation=90)

    plt.plot(days_2005, t2_vals_2005, label='2005')
    plt.plot(days_2006, t2_vals_2006, label='2006')
    plt.plot(days_2007, t2_vals_2007, label='2007')
    plt.legend()
    plt.savefig('Yearly_t2_Comparison.png')

    plt.close()

    plt.ylabel('Daily u10')
    plt.title('Yearly u10 Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, len(days_2005), 12), fontsize=6, rotation=90)

    plt.plot(days_2005, u10_vals_2005, label='2005')
    plt.plot(days_2006, u10_vals_2006, label='2006')
    plt.plot(days_2007, u10_vals_2007, label='2007')
    plt.legend()
    plt.savefig('Yearly_u10_Comparison.png')

    plt.close()

    plt.ylabel('Daily v10')
    plt.title('Yearly v10 Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, len(days_2005), 12), fontsize=6, rotation=90)

    plt.plot(days_2005, v10_vals_2005, label='2005')
    plt.plot(days_2006, v10_vals_2006, label='2006')
    plt.plot(days_2007, v10_vals_2007, label='2007')
    plt.legend()
    plt.savefig('Yearly_v10_Comparison.png')

    plt.close()

    plt.ylabel('Daily seaice')
    plt.title('Yearly seaice Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, len(days_2005), 12), fontsize=6, rotation=90)

    plt.plot(days_2005, seaice_vals_2005, label='2005')
    plt.plot(days_2006, seaice_vals_2006, label='2006')
    plt.plot(days_2007, seaice_vals_2007, label='2007')
    plt.legend()
    plt.savefig('Yearly_seaice_Comparison.png')

    plt.close()

    plt.ylabel('Daily psfc')
    plt.title('Yearly psfc Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, len(days_2005), 12), fontsize=6, rotation=90)

    plt.plot(days_2005, psfc_vals_2005, label='2005')
    plt.plot(days_2006, psfc_vals_2006, label='2006')
    plt.plot(days_2007, psfc_vals_2007, label='2007')
    plt.legend()
    plt.savefig('Yearly_psfc_Comparison.png')

    plt.close()

    ### NOW WE ARE MAKING PLOTS TO COMPARE LAST 3 MONTHS 2005 WITH FIRST 3 2006

    plt.ylabel('Daily tsk')
    plt.title('2005-2006 tsk Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, 180, 6), fontsize=6, rotation=90)

    plt.plot(days_2005[-90:], tsk_vals_2005[-90:], label='2005')
    plt.plot(days_2006[0:90], tsk_vals_2006[0:90], label='2006')
    plt.legend()
    plt.savefig('2005_2006_tsk_Comparison.png')

    plt.close()

    plt.ylabel('Daily Temperature (C)')
    plt.title('2005-2006 t2 Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, 180, 6), fontsize=6, rotation=90)

    # Grab values from weather station
    station_vals = []
    for i in range(2579, 2759):
        tavg = (station_data.iloc[i]['TMAX'] + station_data.iloc[i]['TMIN']) / 2
        tavg = ((tavg + 459.67) * 5) / 9
        station_vals.append(tavg)

    # Convert t2 vals to Celsius
    c_vals_2005 = [x - 273.1 for x in tsk_vals_2005]
    c_vals_2006 = [x - 273.1 for x in tsk_vals_2006]
    c_station_vals = [x - 273.1 for x in station_vals]

    plt.plot(days_2005[-90:], c_vals_2005[-90:], label='2005')
    plt.plot(days_2006[0:90], c_vals_2006[0:90], label='2006')
    combined_days = days_2005[-90:] + days_2006[0:90]
    plt.plot(combined_days, c_station_vals, label='Station Values')
    plt.legend()
    plt.savefig('2005_2006_t2_Comparison.png')

    plt.close()

    plt.ylabel('Daily u10')
    plt.title('2005-2006 u10 Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, 180, 6), fontsize=6, rotation=90)

    plt.plot(days_2005[-90:], u10_vals_2005[-90:], label='2005')
    plt.plot(days_2006[0:90], u10_vals_2006[0:90], label='2006')
    plt.legend()
    plt.savefig('2005_2006_u10_Comparison.png')

    plt.close()

    plt.ylabel('Daily v10')
    plt.title('2005-2006 v10 Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, 180, 6), fontsize=6, rotation=90)

    plt.plot(days_2005[-90:], v10_vals_2005[-90:], label='2005')
    plt.plot(days_2006[0:90], v10_vals_2006[0:90], label='2006')
    plt.legend()
    plt.savefig('2005_2006_v10_Comparison.png')

    plt.close()

    plt.ylabel('Daily seaice')
    plt.title('2005-2006 seaice Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, 180, 6), fontsize=6, rotation=90)

    plt.plot(days_2005[-90:], seaice_vals_2005[-90:], label='2005')
    plt.plot(days_2006[0:90], seaice_vals_2006[0:90], label='2006')
    plt.legend()
    plt.savefig('2005_2006_seaice_Comparison.png')

    plt.close()

    plt.ylabel('Daily psfc')
    plt.title('2005-2006 psfc Comparison')
    plt.xlabel('Day')
    plt.xticks(np.arange(0, 180, 6), fontsize=6, rotation=90)

    plt.plot(days_2005[-90:], psfc_vals_2005[-90:], label='2005')
    plt.plot(days_2006[0:90], psfc_vals_2006[0:90], label='2006')
    plt.legend()
    plt.savefig('2005_2006_psfc_Comparison.png')

    plt.close()

# This method calculates the average temperature for each year of SNAP data and creates a plot
def avg_annual_temp():

    df = pd.read_csv(SNAP_COMBINED_DIRECTORY)
    df = df.set_index('ID')
    yearly_avg = []
    years = []

    for i in range(1970, 2100):
        try:
            start = date(i, 1, 1)
            end = date(i, 12, 31)
            delta = timedelta(days=1)

            daily_values = []

            while start <= end:
                month = str(start.month).zfill(2)
                day = str(start.day).zfill(2)

                daily_values.append(df.loc[f'{month}-{day}-{start.year}_17222', 'tsk'])

                start += delta

            years.append(i)
            avg_val = sum(daily_values) / len(daily_values)
            yearly_avg.append(avg_val)
        except:
            pass

    plt.ylabel('Average Temperature (C)')
    plt.title('Average Annual Temperature')
    plt.xlabel('Year')

    # Convert from K to C
    c_yearly_avg = [x - 273.1 for x in yearly_avg]
    plt.plot(years, c_yearly_avg)
    plt.savefig('avg_annual_tsk.png')

# This method plots the difference between many different transects on a daily basis
# over the course of a year
def transect_comparison():

    df = pd.read_csv(SNAP_COMBINED_DIRECTORY)
    df = df.set_index('ID')

    tsks_2005_17221 = []
    tsks_2005_17222 = []
    tsks_2005_17450 = []
    tsks_2005_17451 = []
    tsks_2005_17641 = []
    tsks_2005_17642 = []

    days_2005 = []

    start_2005 = date(2005, 1, 1)
    end_2005 = date(2005, 12, 31)
    delta = timedelta(days=1)

    while start_2005 < end_2005:
        month = str(start_2005.month).zfill(2)
        day = str(start_2005.day).zfill(2)

        tsks_2005_17221.append(df.loc[f'{month}-{day}-{start_2005.year}_17221', 'tsk'])
        tsks_2005_17222.append(df.loc[f'{month}-{day}-{start_2005.year}_17222', 'tsk'])
        tsks_2005_17450.append(df.loc[f'{month}-{day}-{start_2005.year}_17450', 'tsk'])
        tsks_2005_17451.append(df.loc[f'{month}-{day}-{start_2005.year}_17451', 'tsk'])
        tsks_2005_17641.append(df.loc[f'{month}-{day}-{start_2005.year}_17641', 'tsk'])
        tsks_2005_17642.append(df.loc[f'{month}-{day}-{start_2005.year}_17642', 'tsk'])

        start_2005 += delta

        days_2005.append(f'{month}-{day}')


    plt.ylabel('Surface Temperature (C)')
    plt.title('Transect Comparison 2005')
    plt.xlabel('Transects')

    # Convert to celcius
    c_tsks_2005_17221 = [x - 273.1 for x in tsks_2005_17221]
    c_tsks_2005_17222 = [x - 273.1 for x in tsks_2005_17222]
    c_tsks_2005_17450 = [x - 273.1 for x in tsks_2005_17450]
    c_tsks_2005_17451 = [x - 273.1 for x in tsks_2005_17451]
    c_tsks_2005_17641 = [x - 273.1 for x in tsks_2005_17641]
    c_tsks_2005_17642 = [x - 273.1 for x in tsks_2005_17642]

    plt.plot(days_2005, c_tsks_2005_17221, label='17221')
    plt.plot(days_2005, c_tsks_2005_17222, label='17222')
    plt.plot(days_2005, c_tsks_2005_17450, label='17450')
    plt.plot(days_2005, c_tsks_2005_17451, label='17451')
    plt.plot(days_2005, c_tsks_2005_17641, label='17641')
    plt.plot(days_2005, c_tsks_2005_17642, label='17642')

    plt.legend()
    plt.xticks(np.arange(0, len(days_2005), 12), fontsize=6, rotation=90)

    plt.savefig('transect_tsk_comparison_2005.png')
    plt.close()


if __name__ == '__main__':

    transect_comparison()
