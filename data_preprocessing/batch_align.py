
"""
Batch Alignment Script for Satellite Imagery

This script batch-aligns all satellite images in the `raw_data/` directory to match a reference

coastline raster (9-5-2016_Ortho_4Band_NDWI_3.125m.tif) using GDAL warp.


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
import argparse

# Add the parent directory to the path to import load_config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from load_config import load_config, get_raw_data_path, get_ground_truth_path

def batch_align(raw_data_dir, output_dir, target_srs="EPSG:32603", pixel_size=3.125, te=None, resampling="bilinear"):
    """
    Batch-align satellite images in raw_data_dir using gdalwarp.
    
    Args:
        raw_data_dir (str): Directory containing input .tif files.
        output_dir (str): Directory to save aligned .tif files.
        target_srs (str): Target spatial reference system (e.g., 'EPSG:32603').
        pixel_size (float): Output resolution / pixel size in meters.
        te (list): Target extent [minX, minY, maxX, maxY].
        resampling (str): Resampling method (e.g., 'bilinear', 'near', 'cubic').
    """
    if te is None:
        te = [598355.000000, 7326619.000000, 605849.500000, 7334628.500000]

    os.makedirs(output_dir, exist_ok=True)

    tif_files = [f for f in os.listdir(raw_data_dir) if f.lower().endswith('.tif')]
    if not tif_files:
        print(f"No .tif files found in {raw_data_dir}")
        return

    print(f"Aligning {len(tif_files)} images from '{raw_data_dir}' to '{output_dir}'")
    print(f"Target SRS: {target_srs} | Pixel size: {pixel_size} | Extent: {te} | Resampling: {resampling}")

    for fname in tif_files:
        input_path = os.path.join(raw_data_dir, fname)
        output_path = os.path.join(
            output_dir, fname.replace('.tif', '_aligned.tif')
        )
        cmd = [
            "gdalwarp",
            "-t_srs", target_srs,
            "-tr", str(pixel_size), str(pixel_size),
            "-te", str(te[0]), str(te[1]), str(te[2]), str(te[3]),
            "-r", resampling,
            input_path,
            output_path
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("Batch alignment complete! Aligned files are in:", output_dir)

def main():
    config = load_config()
    default_raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.get('raw_data_folder', 'raw_data'))
    default_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.get('processed_data_folder', 'processed_data'), 'results_batch_align')

    parser = argparse.ArgumentParser(description="Batch Alignment Script for Satellite Imagery using GDAL warp.")
    parser.add_argument("--raw-dir", "--input-dir", dest="raw_dir", type=str, default=default_raw_dir,
                        help=f"Path to input raw data directory (default: {default_raw_dir})")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=default_output_dir,
                        help=f"Path to output directory for aligned images (default: {default_output_dir})")
    parser.add_argument("--target-srs", dest="target_srs", type=str, default="EPSG:32603",
                        help="Target coordinate reference system (default: EPSG:32603)")
    parser.add_argument("--pixel-size", dest="pixel_size", type=float, default=3.125000,
                        help="Pixel size / spatial resolution in meters (default: 3.125)")
    parser.add_argument("--extent", "--te", dest="extent", nargs=4, type=float,
                        default=[598355.000000, 7326619.000000, 605849.500000, 7334628.500000],
                        metavar=('MINX', 'MINY', 'MAXX', 'MAXY'),
                        help="Target bounding box extent [minX minY maxX maxY] (default: 598355.0 7326619.0 605849.5 7334628.5)")
    parser.add_argument("--resampling", "-r", dest="resampling", type=str, default="bilinear",
                        help="GDAL resampling algorithm (default: bilinear)")

    args = parser.parse_args()
    batch_align(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        target_srs=args.target_srs,
        pixel_size=args.pixel_size,
        te=args.extent,
        resampling=args.resampling
    )

if __name__ == "__main__":
    main()


