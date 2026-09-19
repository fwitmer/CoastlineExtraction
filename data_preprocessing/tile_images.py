"""
Tile Images Script

This script splits large satellite images and their corresponding masks into smaller tiles for processing and analysis.
It is useful for handling large images that may be too memory-intensive to process as a whole.

Main Features:
- Splits images into configurable tile sizes (default: 512x512 pixels)
- Uses overlapping tiles (50% overlap) to ensure complete coverage
- Supports multiband images and single-band masks
- Processes both original images and their corresponding masks
- Ensures same tile variations for both images and masks
- Optional filtering of tiles with no-data values
- Parallel processing for improved performance

Inputs:
- Original images from results_georeference/ folder
- Masks from results_create_mask/ folder (TIFF format)
- Configurable tile dimensions (height and width)

Outputs:
- Tiled images and masks saved in results_tile_images/ folder
- Each tile is named with format: original_name_XX-of-YY.tif
- Masks are named with format: original_name_mask_XX-of-YY.tif
"""

import rasterio as rio
from rasterio import windows
import cv2
import argparse

from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import sys

# Add parent directory to path to import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_config import load_config, get_create_mask_output_folder, get_tile_images_output_folder, get_georeference_output_folder  

# adapted from https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
def make_tiles(image, output_folder, tile_height=512, tile_width=512, overlap=0.5, skip_no_data=False):
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in the range [0.0, 1.0)")

    with rio.open(image) as src:
        filepath, filename = os.path.split(image)
        file_base, file_extension = os.path.splitext(filename)
        meta = src.meta.copy()
        num_cols, num_rows = src.meta['width'], src.meta['height']
        overall_window = windows.Window(col_off=0, row_off=0, width=num_cols, height=num_rows)
        step_x = max(1, int(tile_width * (1.0 - overlap)))
        step_y = max(1, int(tile_height * (1.0 - overlap)))
        offsets = product(range(0, num_cols, step_x), range(0, num_rows, step_y))
        tiles = []
        for col_off, row_off in offsets:
            curr_window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
            curr_transform = windows.transform(curr_window, src.transform)
            tiles.append((curr_window.intersection(overall_window), curr_transform))
        for i in range(len(tiles)):
            window, transform = tiles[i]
            window_data = src.read(window=window)
            # optionally skip tiles with no data values
            if skip_no_data:
                if 0 in window_data[..., :-1] or np.all(window_data == 0):
                    continue
            meta['transform'] = transform
            meta['width'] = window_data.shape[-1]
            meta['height'] = window_data.shape[-2]
            out_name = file_base + "_" + str(i + 1).zfill(2) + "-of-" + str(len(tiles)) + file_extension
            out_path = os.path.join(output_folder, out_name)
            with rio.open(out_path, 'w', **meta) as dst:
                dst.write(window_data)

def make_tiles_tiff(image_path, output_folder, tile_height=512, tile_width=512, is_mask=False, overlap=0.5, skip_no_data=False):
    """
    Create tiles from TIFF images (for both images and masks)
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in the range [0.0, 1.0)")

    with rio.open(image_path) as src:
        filepath, filename = os.path.split(image_path)
        file_base, file_extension = os.path.splitext(filename)
        meta = src.meta.copy()
        num_cols, num_rows = src.meta['width'], src.meta['height']
        overall_window = windows.Window(col_off=0, row_off=0, width=num_cols, height=num_rows)
        
        # Calculate tile positions with overlap
        step_x = max(1, int(tile_width * (1.0 - overlap)))
        step_y = max(1, int(tile_height * (1.0 - overlap)))
        offsets = product(range(0, num_cols, step_x), range(0, num_rows, step_y))
        tiles = []
        for col_off, row_off in offsets:
            curr_window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
            curr_transform = windows.transform(curr_window, src.transform)
            tiles.append((curr_window.intersection(overall_window), curr_transform))
        
        # Save tiles
        for i in range(len(tiles)):
            window, transform = tiles[i]
            window_data = src.read(window=window)
            
            if skip_no_data:
                if np.all(window_data == 0) or (is_mask and np.sum(window_data) == 0):
                    continue
            
            meta['transform'] = transform
            meta['width'] = window_data.shape[-1]
            meta['height'] = window_data.shape[-2]
            
            # For masks, ensure single band
            if is_mask:
                meta['count'] = 1
                meta['dtype'] = 'uint8'
            
            # Create output filename
            if is_mask:
                out_name = file_base + "_mask_" + str(i + 1).zfill(2) + "-of-" + str(len(tiles)) + ".tif"
            else:
                out_name = file_base + "_" + str(i + 1).zfill(2) + "-of-" + str(len(tiles)) + ".tif"
            
            out_path = os.path.join(output_folder, out_name)
            
            with rio.open(out_path, 'w', **meta) as dst:
                if is_mask:
                    # For masks, write only the first band
                    dst.write(window_data[0:1].astype('uint8'))
                else:
                    # For images, write all bands
                    dst.write(window_data)

def tile_image_and_mask_pair(image_path, mask_path, output_folder, tile_height=512, tile_width=512, overlap=0.5, skip_no_data=False):
    """
    Create synchronized tiles from an image and its corresponding mask.
    Ensures skip_no_data filters both files identically so pairs remain in sync.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in the range [0.0, 1.0)")

    with rio.open(image_path) as src_img, rio.open(mask_path) as src_mask:
        img_base = os.path.splitext(os.path.basename(image_path))[0]
        
        img_meta = src_img.meta.copy()
        mask_meta = src_mask.meta.copy()
        mask_meta.update(count=1, dtype='uint8')

        num_cols, num_rows = src_img.width, src_img.height
        overall_window = windows.Window(col_off=0, row_off=0, width=num_cols, height=num_rows)
        
        step_x = max(1, int(tile_width * (1.0 - overlap)))
        step_y = max(1, int(tile_height * (1.0 - overlap)))
        offsets = list(product(range(0, num_cols, step_x), range(0, num_rows, step_y)))
        
        valid_tiles = []
        for col_off, row_off in offsets:
            curr_window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
            intersected = curr_window.intersection(overall_window)
            curr_transform = windows.transform(intersected, src_img.transform)
            
            img_data = src_img.read(window=intersected)
            mask_data = src_mask.read(window=intersected)
            
            if skip_no_data:
                if np.all(img_data == 0) or np.all(mask_data == 0) or (0 in img_data[..., :-1]):
                    continue
            
            valid_tiles.append((intersected, curr_transform, img_data, mask_data))
        
        total_tiles = len(valid_tiles)
        for i, (win, trans, img_data, mask_data) in enumerate(valid_tiles):
            # Image tile
            img_tile_meta = img_meta.copy()
            img_tile_meta['transform'] = trans
            img_tile_meta['width'] = img_data.shape[-1]
            img_tile_meta['height'] = img_data.shape[-2]
            img_out_name = f"{img_base}_{str(i + 1).zfill(2)}-of-{str(total_tiles)}.tif"
            img_out_path = os.path.join(output_folder, img_out_name)
            with rio.open(img_out_path, 'w', **img_tile_meta) as dst:
                dst.write(img_data)
            
            # Mask tile
            mask_tile_meta = mask_meta.copy()
            mask_tile_meta['transform'] = trans
            mask_tile_meta['width'] = mask_data.shape[-1]
            mask_tile_meta['height'] = mask_data.shape[-2]
            mask_out_name = f"{img_base}_mask_{str(i + 1).zfill(2)}-of-{str(total_tiles)}.tif"
            mask_out_path = os.path.join(output_folder, mask_out_name)
            with rio.open(mask_out_path, 'w', **mask_tile_meta) as dst:
                dst.write(mask_data[0:1].astype('uint8'))

def make_tiles_png(image_path, output_folder, tile_height=512, tile_width=512):
    """
    Create tiles from PNG images (for masks) - kept for backward compatibility
    """
    # Read the PNG image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    height, width = img.shape
    filepath, filename = os.path.split(image_path)
    file_base, file_extension = os.path.splitext(filename)
    
    # Calculate tile positions with 50% overlap
    tiles = []
    for row in range(0, height, tile_height//2):
        for col in range(0, width, tile_width//2):
            # Calculate tile bounds
            row_end = min(row + tile_height, height)
            col_end = min(col + tile_width, width)
            
            # Extract tile
            tile = img[row:row_end, col:col_end]
            
            # Skip if tile is too small
            if tile.shape[0] < tile_height//2 or tile.shape[1] < tile_width//2:
                continue
                
            tiles.append((tile, row, col))
    
    # Save tiles
    for i, (tile, row, col) in enumerate(tiles):
        out_name = file_base + "_" + str(i + 1).zfill(2) + "-of-" + str(len(tiles)) + file_extension
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, tile)

def find_corresponding_image(mask_path, georef_folder):
    """
    Find the corresponding image for a mask file.
    First tries to find in the same directory as mask, then in georef_folder.
    """
    mask_filename = os.path.basename(mask_path)
    mask_base = mask_filename.replace('_concatenated_ndwi.tif', '')
    
    # First try to find in the same directory as mask
    mask_dir = os.path.dirname(mask_path)
    potential_image = os.path.join(mask_dir, mask_base + '.tif')
    if os.path.exists(potential_image):
        return potential_image
    
    # If not found, try in georef_folder
    potential_image = os.path.join(georef_folder, mask_base + '.tif')
    if os.path.exists(potential_image):
        return potential_image
    
    return None

def process_tiling(mask_folder, georef_folder, output_folder, tile_height=512, tile_width=512, overlap=0.5, skip_no_data=False, limit=5):
    """
    Process image and mask tiling across files.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Mask folder: {mask_folder}")
    print(f"Georeference folder: {georef_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Tile Dimensions: {tile_width}x{tile_height} | Overlap: {overlap * 100}% | Skip No-data: {skip_no_data}")
    
    # Get all mask files from the create_mask output folder
    mask_files = glob.glob(os.path.join(mask_folder, "*", "*_concatenated_ndwi.tif"))
    if not mask_files:
        mask_files = glob.glob(os.path.join(mask_folder, "*_concatenated_ndwi.tif"))
    
    if limit is not None:
        mask_files = mask_files[:limit]
    
    print(f"Found {len(mask_files)} mask files to process")
    
    processed_count = 0
    for mask_path in mask_files:
        print(f"\nProcessing {os.path.basename(mask_path)}...")
        
        # Find corresponding image
        image_path = find_corresponding_image(mask_path, georef_folder)
        
        if image_path is None:
            print(f"  Warning: Could not find corresponding image for {os.path.basename(mask_path)}")
            continue
        
        print(f"  Image: {os.path.basename(image_path)}")
        print(f"  Mask: {os.path.basename(mask_path)}")
        
        try:
            tile_image_and_mask_pair(
                image_path=image_path,
                mask_path=mask_path,
                output_folder=output_folder,
                tile_height=tile_height,
                tile_width=tile_width,
                overlap=overlap,
                skip_no_data=skip_no_data
            )
            
            processed_count += 1
            print(f"  ✓ Successfully processed {os.path.basename(mask_path)}")
            
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(mask_path)}: {str(e)}")
            continue
    
    print(f"\nTiling complete! Processed {processed_count} image-mask pairs.")
    print(f"Output saved to: {output_folder}")


def main():
    config = load_config()
    default_mask_folder = get_create_mask_output_folder(config)
    default_georef_folder = get_georeference_output_folder(config)
    default_output_folder = get_tile_images_output_folder(config)

    parser = argparse.ArgumentParser(description="Split satellite images and masks into tiles.")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Tile height and width in pixels (default: 512)")
    parser.add_argument("--tile-height", type=int, default=None,
                        help="Explicit tile height in pixels (overrides --tile-size)")
    parser.add_argument("--tile-width", type=int, default=None,
                        help="Explicit tile width in pixels (overrides --tile-size)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Overlap fraction between tiles from 0.0 to <1.0 (default: 0.5)")
    parser.add_argument("--skip-nodata", dest="skip_nodata", action="store_true", default=False,
                        help="Skip tiles containing entirely nodata or empty values")
    parser.add_argument("--mask-dir", "--mask-folder", dest="mask_dir", type=str, default=default_mask_folder,
                        help=f"Path to input masks directory (default: {default_mask_folder})")
    parser.add_argument("--georef-dir", "--georef-folder", "--image-dir", dest="georef_dir", type=str, default=default_georef_folder,
                        help=f"Path to input georeferenced images directory (default: {default_georef_folder})")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=default_output_folder,
                        help=f"Path to output directory for tiles (default: {default_output_folder})")
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum number of image-mask pairs to process (default: 5)")

    args = parser.parse_args()

    tile_height = args.tile_height if args.tile_height is not None else args.tile_size
    tile_width = args.tile_width if args.tile_width is not None else args.tile_size

    process_tiling(
        mask_folder=args.mask_dir,
        georef_folder=args.georef_dir,
        output_folder=args.output_dir,
        tile_height=tile_height,
        tile_width=tile_width,
        overlap=args.overlap,
        skip_no_data=args.skip_nodata,
        limit=args.limit
    )

if __name__ == '__main__':
    main()

















# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

# """
# Tile Images Script

# This script splits large satellite images into smaller tiles for processing and analysis.
# It is useful for handling large images that may be too memory-intensive to process as a whole.

# Main Features:
# - Splits images into configurable tile sizes (default: 512x512 pixels)
# - Uses overlapping tiles (50% overlap) to ensure complete coverage
# - Supports multiband images
# - Optional filtering of tiles with no-data values
# - Parallel processing for improved performance

# Inputs:
# - Georeferenced images from results_georeference/ folder
# - Configurable tile dimensions (height and width)

# Outputs:
# - Tiled images saved in results_tile_images/ folder
# - Each tile is named with format: original_name_XX-of-YY.tif
# """

# import rasterio as rio
# from rasterio import windows

# from itertools import product
# from matplotlib import pyplot as plt
# import numpy as np
# import os
# import glob
# import sys

# # Add parent directory to path to import load_config
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from load_config import load_config, get_georeference_output_folder, get_tile_images_output_folder

# # adapted from https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
# def make_tiles(image, output_folder, tile_height=512, tile_width=512, skip_no_data=False):
#     with rio.open(image) as src:
#         filepath, filename = os.path.split(image)
#         file_base, file_extension = os.path.splitext(filename)
#         meta = src.meta.copy()
#         num_cols, num_rows = src.meta['width'], src.meta['height']
#         overall_window = windows.Window(col_off=0, row_off=0, width=num_cols, height=num_rows)
#         offsets = product(range(0, num_cols, tile_height//2), range(0, num_rows, tile_width//2))
#         tiles = []
#         for col_off, row_off in offsets:
#             curr_window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
#             curr_transform = windows.transform(curr_window, src.transform)
#             tiles.append((curr_window.intersection(overall_window), curr_transform))
#         for i in range(len(tiles)):
#             window, transform = tiles[i]
#             meta['transform'] = transform
#             meta['width'] = tile_width
#             meta['height'] = tile_height
#             window_data = src.read(window=window)
#             # optionally skip tiles with no data values
#             if skip_no_data:
#                 if 0 in window_data[..., :-1]:
#                     continue
#             out_name = file_base + "_" + str(i + 1).zfill(2) + "-of-" + str(len(tiles)) + file_extension
#             out_path = os.path.join(output_folder, out_name)
#             with rio.open(out_path, 'w', **meta) as dst:
#                 dst.write(src.read(window=window))

# # example usage
# if __name__ == '__main__':
#     # Load configuration
#     config = load_config()
    
#     # Get input and output folders from config
#     input_folder = get_georeference_output_folder(config)
#     output_folder = get_tile_images_output_folder(config)
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Get all .tif files from the georeference output folder
#     files = glob.glob(os.path.join(input_folder, "*.tif"))
    
#     print(f"Processing {len(files)} files from {input_folder}")
#     print(f"Output will be saved to {output_folder}")
    
#     from concurrent.futures import ThreadPoolExecutor
#     with ThreadPoolExecutor(max_workers=6) as p:
#         # Pass both the file and output_folder to make_tiles
#         p.map(lambda f: make_tiles(f, output_folder), files)