"""
Batch Alignment Script for Satellite Imagery

This script batch-aligns all satellite images in the `raw_data/` directory to match a reference
coastline raster (9-5-2016_Ortho_4Band_NDWI.tif) using GDAL warp.

Aligned images are saved to `processed_data/results_batch_align/` with a '_aligned.tif' suffix.

Key Parameters:
- Target CRS: UTM Zone 3N (EPSG:32603)
- Pixel size: 0.5 meters (high resolution)
- Extent: [598355.000000, 7326619.000000, 605849.500000, 7334628.500000]

Dependencies:
- GDAL CLI tools (`gdalwarp`)
- Python: os, sys, subprocess, custom `load_config` module

Usage:
    conda activate arosics_env
    python data_preprocessing/batch_align.py
"""


import os
import subprocess
import sys

# Add the parent directory to the path to import load_config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from load_config import load_config, get_raw_data_path, get_ground_truth_path

# Load configuration
config = load_config()

# The raw data directory path
raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['raw_data_folder'])

# The base image path (9-5-2016_Ortho_4Band_NDWI_3.125m.tif)
base_img = get_ground_truth_path(config, 4)  # Index 4 for 9-5-2016_Ortho_4Band_NDWI.tif

# Output directory
aligned_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['processed_data_folder'], 'results_batch_align')

# Create the output directory if it doesn't exist
os.makedirs(aligned_data_dir, exist_ok=True)

# Alignment parameters from your base image
# target_srs = "EPSG:32603"
# pixel_size = 5.532779396951528
# pixel_size = 0.5  # Match base image resolution (0.5m)
# te = [598472.146, 7327174.321, 605731.152, 7333144.190]  # [minX, minY, maxX, maxY]

target_srs = "EPSG:32603"
pixel_size = 3.125000 
te = [598355.000000, 7326619.000000, 605849.500000, 7334628.500000]  # [minX, minY, maxX, maxY]

# Process all .tif files in the raw_data directory
for fname in os.listdir(raw_data_dir):
    if fname.lower().endswith('.tif'):
        input_path = os.path.join(raw_data_dir, fname)
        output_path = os.path.join(
            aligned_data_dir, fname.replace('.tif', '_aligned.tif')
        )
        cmd = [
            "gdalwarp",
            "-t_srs", target_srs,
            "-tr", str(pixel_size), str(pixel_size),
            "-te", str(te[0]), str(te[1]), str(te[2]), str(te[3]),
            "-r", "bilinear",
            input_path,
            output_path
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

print("Batch alignment complete! Aligned files are in:", aligned_data_dir) 






# gdl command for to reproject it to UTM Zone 3N coordinate system.
# gdalwarp -t_srs EPSG:32603 2016_HiRes_Final_Coastline.tif 2016_HiRes_Final_Coastline_UTM3N.tif