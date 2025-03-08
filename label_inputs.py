import rasterio as rio
from rasterio import merge
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.io import MemoryFile
from rasterio.features import shapes
from rasterio.mask import mask

from datetime import datetime, timedelta

import numpy as np
import glob
import re
import os

# creates a rasterio dataset in memory from a data array and corresponding CRS and transform
# defaults to single-band datasets with nodata value of 0
def create_dataset(data, crs, transform):
    memfile = MemoryFile()
    dataset = memfile.open(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                           crs=crs, transform=transform, dtype=data.dtype, nodata=0)
    dataset.write(data, 1)
    return dataset

# New helper that parses the date and returns a datetime object
def parse_date_from_filename(filename):
    """
    Parse a date from the filename.
    First, try "YYYY-MM-DD"; if that fails, try "YYYY_MM" (assuming the 1st day of the month).
    """
    regex1 = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", filename)
    if regex1:
        try:
            return datetime.strptime(regex1.group(0), "%Y-%m-%d")
        except Exception:
            pass
    regex2 = re.search(r"([0-9]{4}_[0-9]{2})", filename)
    if regex2:
        try:
            return datetime.strptime(regex2.group(0), "%Y_%m")
        except Exception:
            pass
    raise ValueError(f"Filename {filename} does not contain a recognizable date format.")

# Merges an input image with a corresponding label image into a 5-banded image
def add_labels(input_path, label_path, output_path):
    with rio.open(label_path, 'r', driver='GTiff') as label, \
         rio.open(input_path, 'r', driver='GTiff') as input_data:
        # Copy metadata and update for the new band count.
        input_depth = input_data.count
        input_meta = input_data.meta.copy()
        input_meta.update(count=5)

        # Reproject the label layer to match the CRS and resolution of the input.
        label_reproj, label_reproj_trans = reproject(source=rio.band(label, 1),
                                                     dst_crs=input_data.profile['crs'],
                                                     dst_resolution=input_data.res,
                                                     resampling=rio.enums.Resampling.cubic_spline)
        
        label_ds = create_dataset(label_reproj[0], input_data.profile['crs'], label_reproj_trans)

        # Crop reprojected labels to the input image's extent.
        extents, _ = next(shapes(np.zeros_like(input_data.read(1)), transform=input_data.profile['transform']))
        cropped_label, _ = mask(label_ds, [extents], crop=True)

        # Update the label layer to have no data where the input image has no data.
        cropped_label_array = cropped_label[0][:input_data.shape[0], :input_data.shape[1]]
        cropped_label_array = np.where(input_data.read(1) == 0, 0, cropped_label_array)

        with rio.open(output_path, 'w', **input_meta) as dst:
            dst.write_band(1, input_data.read(1))
            dst.write_band(2, input_data.read(2))
            dst.write_band(3, input_data.read(3))
            dst.write_band(4, input_data.read(input_depth))
            dst.write_band(5, cropped_label_array.astype(rio.uint16))

def match_labels(input_path, label_path):
    input_files = glob.glob(os.path.join(input_path, "*.tif"))
    label_files = glob.glob(os.path.join(label_path, "*.tif"))

    # Prepare labels for comparison by parsing dates to datetime objects.
    label_dict = {}
    for label in label_files:
        label_date = parse_date_from_filename(label) + timedelta(days=14)
        label_dict[label_date] = label
    
    sorted_label_dates = sorted(label_dict.keys())
    
    # Compare each input file to the label files to find the closest match.
    for input_file in input_files:
        print("Input file:", os.path.basename(input_file))
        input_date = parse_date_from_filename(input_file)
        closest_date = min(sorted_label_dates, key=lambda d: abs(d - input_date))
        print("Matching label:", os.path.basename(label_dict[closest_date]))
        out_name = input_date.strftime("%Y-%m-%d") + "_labeled.tif"
        out_path = os.path.join("data/labeled_inputs/", out_name)
        if os.path.exists(out_path):
            print("Labeled file already exists at:", out_path)
            print()
            continue
        print("Merging as:", out_name, end="...")
        add_labels(input_file, label_dict[closest_date], out_path)
        print("DONE")
        print()

if __name__ == "__main__":
    match_labels("data/input/", "data/labels/")