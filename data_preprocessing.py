import rasterio as rio
from rasterio import windows

from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import argparse
import sys

# adapted from https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
def make_tiles(image, tile_height=512, tile_width=512, skip_no_data=False, output_dir="data/tiles"):
    with rio.open(image) as src:
        filepath, filename = os.path.split(image)
        file_base, file_extension = os.path.splitext(filename)
        meta = src.meta.copy()
        num_cols, num_rows = src.meta['width'], src.meta['height']
        overall_window = windows.Window(col_off=0, row_off=0, width=num_cols, height=num_rows)
        offsets = product(range(0, num_cols, tile_height//2), range(0, num_rows, tile_width//2))
        tiles = []
        for col_off, row_off in offsets:
            curr_window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
            curr_transform = windows.transform(curr_window, src.transform)
            tiles.append((curr_window.intersection(overall_window), curr_transform))
        for i in range(len(tiles)):
            window, transform = tiles[i]
            meta['transform'] = transform
            meta['width'] = tile_width
            meta['height'] = tile_height
            window_data = src.read(window=window)
            # optionally skip tiles with no data values
            if skip_no_data:
                if 0 in window_data[..., :-1]:
                    continue
            out_name = file_base + "_" + str(i + 1).zfill(2) + "-of-" + str(len(tiles)) + file_extension
            out_path = os.path.join(output_dir, out_name)
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            with rio.open(out_path, 'w', **meta) as dst:
                dst.write(src.read(window=window))

def _augment_and_write(bands, outpath, metadata, rotations=1):
    with rio.open(outpath, 'w', **metadata) as dst:
        for i in range(len(bands)):
            dst.write(np.rot90(bands[i], rotations), i+1)

def _flip_bands(bands):
    flipped_bands = [np.flipud(band) for band in bands]
    return flipped_bands

# takes the path to all image tiles and creates tiles that are rotated 90°, 180° and 270° as well as their flipped counterparts
# this results in 8 tiles for every input tile (including the input tile)
def augment_tiles(tile_path):
    # Normalize tile_path and ensure it's a directory path
    tile_path = os.path.normpath(tile_path)
    if not os.path.isdir(tile_path):
        raise ValueError(f"Tile path is not a valid directory: {tile_path}")
    
    # Use os.path.join for glob patterns (works cross-platform)
    files = glob.glob(os.path.join(tile_path, "*.tif"))
    files = set(files) - set(glob.glob(os.path.join(tile_path, "*rot*")))
    files = set(files) - set(glob.glob(os.path.join(tile_path, "*flip*"))) 
    for file in files:
        filename = os.path.basename(file)
        file_base, file_extension = os.path.splitext(filename)
        # generating filepaths for new tiles
        path_90 = os.path.join(tile_path, file_base + "_rot90" + file_extension)
        path_180 = os.path.join(tile_path, file_base + "_rot180" + file_extension)
        path_270 = os.path.join(tile_path, file_base + "_rot270" + file_extension)
        path_flip_name = os.path.join(tile_path, file_base + "_flip" + file_extension)
        path_flip_90 = os.path.join(tile_path, file_base + "_rot90_flip" + file_extension)
        path_flip_180 = os.path.join(tile_path, file_base + "_rot180_flip" + file_extension)
        path_flip_270 = os.path.join(tile_path, file_base + "_rot270_flip" + file_extension)
         
        with rio.open(file, driver="GTiff") as src:
            # band_1 = src.read(1)
            # band_2 = src.read(2)
            # band_3 = src.read(3)
            # band_4 = src.read(4)
            # band_5 = src.read(5)
            bands = (src.read(1), src.read(2), src.read(3), src.read(4), src.read(5))
            meta = src.meta
            
            _augment_and_write(bands, path_90, meta, 1) # 90°
            _augment_and_write(bands, path_180, meta, 2) # 180°
            _augment_and_write(bands, path_270, meta, 3) # 270°
            flipped_bands = _flip_bands(bands)
            _augment_and_write(flipped_bands, path_flip_name, meta, 0) # flipped up/down
            _augment_and_write(flipped_bands, path_flip_90, meta, 1) # flipped & 90°
            _augment_and_write(flipped_bands, path_flip_180, meta, 2) # flipped & 180°
            _augment_and_write(flipped_bands, path_flip_270, meta, 3) # flipped & 270°


        
# example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess images by creating tiles and augmenting them',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python data_preprocessing.py --input_dir data/labeled_inputs --output_dir data/tiles
  
  python data_preprocessing.py --input_dir ./images --output_dir ./tiles --augment
        '''
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/labeled_inputs',
        help='Directory containing input images to tile. Default: data/labeled_inputs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tiles',
        help='Directory for output tiles. Default: data/tiles'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Run tile augmentation after creating tiles'
    )
    parser.add_argument(
        '--tile_height',
        type=int,
        default=512,
        help='Height of tiles in pixels. Default: 512'
    )
    parser.add_argument(
        '--tile_width',
        type=int,
        default=512,
        help='Width of tiles in pixels. Default: 512'
    )
    parser.add_argument(
        '--skip_no_data',
        action='store_true',
        help='Skip tiles with no data values'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=6,
        help='Maximum number of worker threads. Default: 6'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        print("Please specify a valid directory using --input_dir argument.", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .tif files in input directory
    input_pattern = os.path.join(args.input_dir, "*.tif")
    files = glob.glob(input_pattern)
    
    if not files:
        print(f"No .tif files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(files)} files to process")
    print(f"Output directory: {args.output_dir}")
    
    from concurrent.futures import ThreadPoolExecutor
    
    # Create tiles
    def process_file(file):
        make_tiles(
            file,
            tile_height=args.tile_height,
            tile_width=args.tile_width,
            skip_no_data=args.skip_no_data,
            output_dir=args.output_dir
        )
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as p:
        p.map(process_file, files)
    
    print("Tile creation complete!")
    
    # Augment tiles if requested
    if args.augment:
        print("Starting tile augmentation...")
        augment_tiles(args.output_dir)
        print("Tile augmentation complete!")