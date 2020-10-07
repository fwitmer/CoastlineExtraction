import sys
import rasterio
import os
import numpy as np

'''
Create a new padded tiff with num_blank_bands added layers of zeros.

input_path          : source tiff file
num_blank_bands     : the number of bands to add
output_path         : file to save padded image to
dtype               : if unspecified defaults to 'guess', which uses the first value
                      in the source tiff's DatasetReader's dtypes attribute
'''

def tiff_add_bands(input_path, num_blank_bands, output_path, dtype='guess'):

    with rasterio.open(input_path) as source_tif:
        source_data = source_tif.read()
        source_num_bands, source_height, source_width = source_data.shape
        if dtype == 'guess':
            dtype = source_tif.dtypes[0]
        zeros = np.zeros((num_blank_bands, source_height, source_width), dtype=dtype)
        data = np.vstack((source_data, zeros))
        with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=source_height,
                    width=source_width,
                    count=data.shape[0],
                    dtype=dtype
        ) as new_tif:
            new_tif.write(data)



# run as standalone script
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'USAGE: python3 {__file__} <input file> <number of empty bands to add> <output file>')

    elif not os.path.isfile(sys.argv[1]):
        print(f'Could not locate file {sys.argv[1]}.')

    elif(os.path.isfile(sys.argv[3])):
        print(f'Output file {sys.argv[3]} already exists.')

    else:
        input_path = sys.argv[1]
        num_blank_bands = int(sys.argv[2])
        output_path = sys.argv[3]
        tiff_add_bands(input_path, num_blank_bands, output_path)
