import rasterio as rio
from rasterio import merge
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.io import MemoryFile
from rasterio.features import shapes
from rasterio.mask import mask

import numpy as np

import glob
from datetime import datetime
from datetime import timedelta
import re
import os

# creates a rasterio dataset in memory from a data array and corresponding CRS and transform
# defaults to single-band datasets with nodata value of 0
# adapted from https://medium.com/analytics-vidhya/python-for-geosciences-raster-merging-clipping-and-reprojection-with-rasterio-9f05f012b88a
def create_dataset(data, crs, transform):
    memfile = MemoryFile()
    dataset = memfile.open(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                           crs=crs, transform=transform, dtype=data.dtype, nodata=0)
    dataset.write(data, 1)

    return dataset

# takes a path for an input image and a path for a corresponding label image
# upscales the label image to match the resolution of the input image and merges them into a 5-banded image
def add_labels(input_path, label_path, output_path):
    with rio.open(label_path, 'r', driver='GTiff') as label, \
         rio.open(input_path, 'r', driver='GTiff') as input:

        # copying metadat and updating for the new band count
        input_meta = input.meta
        input_meta.update(count=5)

        # reprojecting label layer to match the CRS and resolution of input
        label_reproj, label_reproj_trans = reproject(source=rio.band(label, 1),
                                                     dst_crs = input.profile['crs'],
                                                     dst_resolution=input.res,
                                                     resampling=rio.enums.Resampling.cubic_spline)
        
        label_ds = create_dataset(label_reproj[0], input.profile['crs'], label_reproj_trans)

        # cropping reprojected labels to input image's extent
        extents, _ = next(shapes(np.zeros_like(input.read(1)), transform=input.profile['transform']))
        cropped_label, crop_transf = mask(label_ds, [extents], crop=True)

        # updating label layer to have no data where input image has no data
        cropped_label_array = cropped_label[0][:input.shape[0], :input.shape[1]]
        cropped_label_array = np.where(input.read(1) == 0, 0, cropped_label_array)

        # print(reprojected_labels[0].shape)
        with rio.open(output_path, 'w', **input_meta) as dst:
            dst.write_band(1, input.read(1))
            dst.write_band(2, input.read(2))
            dst.write_band(3, input.read(3))
            dst.write_band(4, input.read(4))
            dst.write_band(5, cropped_label_array.astype(rio.uint16))

# returns the date from a filename in YYYY-MM-DD string format
def parse_date(filename):
    date = re.search("([0-9]{4}\-[0-9]{2}-[0-9]{2})", filename)
    if date:
        return date.group(0)
    else:
        date = re.search("([0-9]{4}\_[0-9]{2})", filename)
        return date.group(0)

def match_labels(input_path, label_path):
    input_files = glob.glob(input_path + "*.tif")
    label_files = glob.glob(label_path + "*.tif")

    # preparing labels for comparison
    label_dict = {}
    label_dates = []
    for label in label_files:
        label_date = datetime.strptime(parse_date(label), "%Y_%m") + timedelta(days=14)
        label_dates.append(label_date)
        label_dict[label_date] = label
    
    # comparing each input file to the label files to find the closest match
    for input in input_files:
        input_date = datetime.strptime(parse_date(input), "%Y-%m-%d")
        date_diffs = [abs(label_date - input_date) for label_date in label_dates]
        closest_date = label_dates[date_diffs.index(min(date_diffs))]
        print("Input file:", os.path.basename(input))
        print("Matching label:", os.path.basename(label_dict[closest_date]))
        out_name = input_date.strftime("%Y-%m-%d") + "_labeled.tif"
        out_path = "data/labeled_inputs/" + out_name
        print("Merging as", out_name, "...")
        add_labels(input, label_dict[closest_date], out_path)
        print("Done.")
        print()

match_labels("data/input/", "data/labels/")