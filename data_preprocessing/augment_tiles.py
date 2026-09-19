"""
augment_tiles.py
----------------

This module provides data augmentation functionality for image and mask tiles.
It creates rotated (90°, 180°, 270°) and flipped versions of input tiles,
resulting in 7 augmented tiles for every input tile (8 total including original).

Main Features:
- Creates 7 augmented versions per input tile (rotations and flips)
- Processes both image tiles and their corresponding masks
- Applies same transformations to both images and masks
- Configurable maximum tile processing limit
- Automatic filtering of already augmented files
- Works with TIFF format for both images and masks

Input Files:
- Image tiles: results_tile_images/*_01-of-110.tif to *_110-of-110.tif
- Mask tiles: results_tile_images/*_concatenated_ndwi_mask_01-of-110.tif to *_concatenated_ndwi_mask_110-of-110.tif

Output Files:
- Augmented tiles saved in results_augment_tiles/ folder
- Naming convention: original_name_rot90.tif, original_name_flip.tif, etc.
- 7 augmented versions per input tile (rot90, rot180, rot270, flip, flip_rot90, flip_rot180, flip_rot270)

Functions:
    - augment_tiles(tile_path, output_path, max_tiles=None): Creates augmented versions of tiles
    - _augment_and_write_tiff(image, output_path, rotation_angle, flip=False): Helper function for TIFF augmentation
"""

import cv2
import numpy as np
import os
import glob
import sys
import argparse
import rasterio
from rasterio.transform import from_bounds

# Add parent directory to path to import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_config import load_config, get_tile_images_output_folder, get_augment_tiles_output_folder

def _augment_and_write_tiff(image_data, output_path, rotation_angle, flip=False, profile=None):
    """
    Helper function to rotate and/or flip TIFF image and write to file
    
    Args:
        image_data: Input image array (numpy array)
        output_path: Path to save the augmented image
        rotation_angle: Rotation angle (0=no rotation, 1=90°, 2=180°, 3=270°)
        flip: Whether to flip the image vertically
        profile: Rasterio profile for writing TIFF
    """
    # Handle multi-dimensional data properly
    # Rasterio reads data as (bands, height, width)
    if len(image_data.shape) == 3:
        # Multi-band image: process each band separately
        num_bands = image_data.shape[0]
        augmented_bands = []
        
        for band_idx in range(num_bands):
            band_data = image_data[band_idx]
            
            # Apply flip if requested
            if flip:
                band_data = cv2.flip(band_data, 0)  # 0 = vertical flip
            
            # Apply rotation
            if rotation_angle == 0:
                augmented_band = band_data
            elif rotation_angle == 1:  # 90 degrees
                augmented_band = cv2.rotate(band_data, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 2:  # 180 degrees
                augmented_band = cv2.rotate(band_data, cv2.ROTATE_180)
            elif rotation_angle == 3:  # 270 degrees
                augmented_band = cv2.rotate(band_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                augmented_band = band_data
            
            augmented_bands.append(augmented_band)
        
        # Stack the bands back together
        augmented_image = np.stack(augmented_bands, axis=0)
    else:
        # Single-band data (masks), process directly
        # Apply flip if requested
        if flip:
            image_data = cv2.flip(image_data, 0)  # 0 = vertical flip
        
        # Apply rotation
        if rotation_angle == 0:
            augmented_image = image_data
        elif rotation_angle == 1:  # 90 degrees
            augmented_image = cv2.rotate(image_data, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 2:  # 180 degrees
            augmented_image = cv2.rotate(image_data, cv2.ROTATE_180)
        elif rotation_angle == 3:  # 270 degrees
            augmented_image = cv2.rotate(image_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            augmented_image = image_data
    
    # Save the augmented image as TIFF
    if profile is not None:
        # Update profile with new dimensions if rotated
        if len(augmented_image.shape) == 3:
            # Multi-band data: (bands, height, width)
            if rotation_angle in [1, 3]:  # 90 or 270 degree rotation
                profile.update({
                    'height': augmented_image.shape[1],
                    'width': augmented_image.shape[2]
                })
            else:
                profile.update({
                    'height': augmented_image.shape[1],
                    'width': augmented_image.shape[2]
                })
        else:
            # Single-band data: (height, width)
            if rotation_angle in [1, 3]:  # 90 or 270 degree rotation
                profile.update({
                    'height': augmented_image.shape[0],
                    'width': augmented_image.shape[1]
                })
            else:
                profile.update({
                    'height': augmented_image.shape[0],
                    'width': augmented_image.shape[1]
                })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            if len(augmented_image.shape) == 3:
                # Multi-band image: (bands, height, width)
                for i in range(augmented_image.shape[0]):
                    dst.write(augmented_image[i], i + 1)
            else:
                # Single band image: (height, width)
                dst.write(augmented_image, 1)
    else:
        # Fallback: save as PNG if no profile available
        cv2.imwrite(output_path.replace('.tif', '.png'), augmented_image)

def augment_tiles(tile_path, output_path, max_tiles=None):
    """
    Takes the path to image and mask tiles and creates tiles that are rotated 90°, 180° and 270° 
    as well as their flipped counterparts. This results in 7 augmented tiles for every input tile 
    (8 total including the original tile). Processes both image and mask tiles with the same transformations.
    
    Args:
        tile_path: Path to the directory containing input tiles
        output_path: Path to the directory where augmented tiles will be saved
        max_tiles: Maximum number of tile pairs to process (default: None, processes all tiles)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get all image files (TIFF format: *_01-of-110.tif to *_110-of-110.tif)
    image_files = glob.glob(os.path.join(tile_path, "*_*-of-*.tif"))
    # Filter out mask files (containing "mask" in filename)
    image_files = [f for f in image_files if "mask" not in f]
    
    # Filter out already augmented files
    image_files = [f for f in image_files if not any(x in f for x in ["_rot", "_flip"])]
    
    # Sort all image files for consistent processing
    image_files = sorted(image_files)
    
    # Take only the first max_tiles if specified, otherwise process all tiles
    if max_tiles is not None:
        image_files = image_files[:max_tiles]
    
    print(f"Processing {len(image_files)} image-mask pairs for augmentation...")
    print(f"Each pair will generate 7 additional augmented versions (8 total per pair)")
    print(f"Output directory: {output_path}")
    
    successful = 0
    failed = 0
    
    for i, image_file in enumerate(image_files):
        filename = os.path.basename(image_file)
        file_base = filename.replace(".tif", "")
        
        # Find corresponding mask file
        # Convert image filename to mask filename
        # e.g., "1474782_0369619_2018-06-02_0f51_BGRN_SR_clip_aligned_GeoRegistered_01-of-110.tif" 
        # -> "1474782_0369619_2018-06-02_0f51_BGRN_SR_clip_aligned_GeoRegistered_concatenated_ndwi_mask_01-of-110.tif"
        # The mask naming pattern is: base_name_concatenated_ndwi_mask_tile_number.tif
        # So we need to insert "_concatenated_ndwi_mask" before the tile number part
        if "-of-" in file_base:
            # Split at the tile number part
            parts = file_base.split("_")
            # Find the part that contains "-of-"
            tile_part_idx = None
            for i, part in enumerate(parts):
                if "-of-" in part:
                    tile_part_idx = i
                    break
            
            if tile_part_idx is not None:
                # Insert "_concatenated_ndwi_mask" before the tile part
                parts.insert(tile_part_idx, "concatenated_ndwi_mask")
                mask_filename = "_".join(parts) + ".tif"
            else:
                # Fallback: just append
                mask_filename = file_base + "_concatenated_ndwi_mask.tif"
        else:
            # Fallback: just append
            mask_filename = file_base + "_concatenated_ndwi_mask.tif"
        
        mask_file = os.path.join(tile_path, mask_filename)
        
        if not os.path.exists(mask_file):
            print(f"Warning: No corresponding mask file found for {filename}")
            print(f"Looking for: {mask_filename}")
            continue
        
        try:
            # Read image and mask using rasterio
            with rasterio.open(image_file) as src:
                image_data = src.read()
                image_profile = src.profile.copy()
            
            with rasterio.open(mask_file) as src:
                mask_data = src.read()
                mask_profile = src.profile.copy()
            
            if image_data is None or mask_data is None:
                print(f"Error: Could not read {filename} or its mask")
                failed += 1
                continue
            
            # Generate filepaths for augmented versions
            base_name = file_base
            
            # Create augmented versions for both image and mask
            augmentations = [
                ("_rot90", 1, False),
                ("_rot180", 2, False),
                ("_rot270", 3, False),
                ("_flip", 0, True),
                ("_rot90_flip", 1, True),
                ("_rot180_flip", 2, True),
                ("_rot270_flip", 3, True)
            ]
            
            for aug_suffix, rotation_angle, flip in augmentations:
                # Augment image
                image_output_path = os.path.join(output_path, base_name + aug_suffix + ".tif")
                _augment_and_write_tiff(image_data, image_output_path, rotation_angle, flip, image_profile)
                
                # Augment mask
                # Generate mask output filename with the same pattern as input
                if "-of-" in base_name:
                    # Split at the tile number part
                    parts = base_name.split("_")
                    # Find the part that contains "-of-"
                    tile_part_idx = None
                    for i, part in enumerate(parts):
                        if "-of-" in part:
                            tile_part_idx = i
                            break
                    
                    if tile_part_idx is not None:
                        # Insert "_concatenated_ndwi_mask" before the tile part
                        parts.insert(tile_part_idx, "concatenated_ndwi_mask")
                        mask_base_name = "_".join(parts)
                    else:
                        # Fallback: just append
                        mask_base_name = base_name + "_concatenated_ndwi_mask"
                else:
                    # Fallback: just append
                    mask_base_name = base_name + "_concatenated_ndwi_mask"
                
                mask_output_path = os.path.join(output_path, mask_base_name + aug_suffix + ".tif")
                _augment_and_write_tiff(mask_data, mask_output_path, rotation_angle, flip, mask_profile)
            
            successful += 1
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} image-mask pairs")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed += 1
            continue
    
    print(f"\n=== Augmentation Summary ===")
    print(f"Successfully processed: {successful} image-mask pairs")
    print(f"Failed: {failed} image-mask pairs")
    print(f"Total augmented tiles created: {successful * 14} (7 image + 7 mask per pair)")
    print(f"Total tiles after augmentation: {successful * 16} (including originals)")

def main():
    try:
        config = load_config()
    except Exception:
        config = {}
    
    default_tiles_dir = get_tile_images_output_folder(config) if config else "processed_data/results_tile_images"
    default_output_dir = get_augment_tiles_output_folder(config) if config else "processed_data/results_augment_tiles"

    parser = argparse.ArgumentParser(description="Data augmentation for image and mask tiles (rotations and flips).")
    parser.add_argument("--input-dir", "--tiles-dir", dest="input_dir", type=str, default=default_tiles_dir,
                        help=f"Path to input tiles directory (default: {default_tiles_dir})")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=default_output_dir,
                        help=f"Path to output directory for augmented tiles (default: {default_output_dir})")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Maximum number of tile pairs to process (default: all)")

    args = parser.parse_args()

    # Check if tiles directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: {args.input_dir} directory not found!")
        print("Please run tile_images.py first to generate tiles.")
        sys.exit(1)
    
    # Check if there are any image tiles to process
    image_files = glob.glob(os.path.join(args.input_dir, "*_*-of-*.tif"))
    image_files = [f for f in image_files if "mask" not in f]
    
    if not image_files:
        print(f"No TIFF image tiles found in {args.input_dir}")
        print("Please run tile_images.py first to generate tiles.")
        sys.exit(1)
    
    print("=== Data Augmentation Workflow ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(image_files)} image tiles")
    limit_str = f"first {args.max_tiles}" if args.max_tiles else "ALL"
    print(f"Will process {limit_str} image-mask pairs")
    print("Each image-mask pair will generate 7 augmented versions.")
    print()
    
    augment_tiles(args.input_dir, args.output_dir, max_tiles=args.max_tiles)
    print("\nAugmentation complete!")

if __name__ == '__main__':
    main()













# ---------------------------------------------------------------------------------------------------------------------------------------

# """
# augment_tiles.py
# ----------------

# This module provides data augmentation functionality for image tiles.
# It creates rotated (90°, 180°, 270°) and flipped versions of input tiles,
# resulting in 7 augmented tiles for every input tile (8 total including original).

# Main Features:
# - Creates 7 augmented versions per input tile (rotations and flips)
# - Supports multiband images
# - Configurable maximum tile processing limit
# - Automatic filtering of already augmented files

# Input Files:
# - Image tiles from results_tile_images/ folder (*.tif files)
# - Each input tile should be a GeoTIFF with multiple bands

# Output Files:
# - Augmented tiles saved in results_augment_tiles/ folder
# - Naming convention: original_name_rot90.tif, original_name_flip.tif, etc.
# - 7 augmented versions per input tile (rot90, rot180, rot270, flip, flip_rot90, flip_rot180, flip_rot270)

# Functions:
#     - augment_tiles(tile_path, output_path, max_tiles=40): Creates augmented versions of tiles
#     - _augment_and_write(bands, output_path, meta, rotation_angle): Helper function for rotation
#     - _flip_bands(bands): Helper function for flipping
# """

# import rasterio as rio
# import numpy as np
# import os
# import glob
# import sys

# # Add parent directory to path to import load_config
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from load_config import load_config, get_tile_images_output_folder, get_augment_tiles_output_folder

# def _augment_and_write(bands, output_path, meta, rotation_angle):
#     """
#     Helper function to rotate bands and write to file
    
#     Args:
#         bands: Tuple of band arrays
#         output_path: Path to save the augmented image
#         meta: Rasterio metadata
#         rotation_angle: Rotation angle (0=no rotation, 1=90°, 2=180°, 3=270°)
#     """
#     rotated_bands = []
#     for band in bands:
#         if rotation_angle == 0:
#             rotated_band = band
#         elif rotation_angle == 1:  # 90 degrees
#             rotated_band = np.rot90(band, k=1)
#         elif rotation_angle == 2:  # 180 degrees
#             rotated_band = np.rot90(band, k=2)
#         elif rotation_angle == 3:  # 270 degrees
#             rotated_band = np.rot90(band, k=3)
#         rotated_bands.append(rotated_band)
    
#     # Update metadata for rotated image
#     meta_rot = meta.copy()
#     if rotation_angle in [1, 3]:  # 90 or 270 degrees
#         meta_rot['width'], meta_rot['height'] = meta['height'], meta['width']
    
#     with rio.open(output_path, 'w', **meta_rot) as dst:
#         for i, band in enumerate(rotated_bands, 1):
#             dst.write(band, i)

# def _flip_bands(bands):
#     """
#     Helper function to flip bands vertically
    
#     Args:
#         bands: Tuple of band arrays
    
#     Returns:
#         List of flipped band arrays
#     """
#     return [np.flipud(band) for band in bands]

# # def augment_tiles(tile_path, output_path, max_tiles=40):
# def augment_tiles(tile_path, output_path, max_tiles=None):
#     """
#     Takes the path to image tiles and creates tiles that are rotated 90°, 180° and 270° 
#     as well as their flipped counterparts. This results in 7 augmented tiles for every input tile 
#     (8 total including the original tile).
    
#     Args:
#         tile_path: Path to the directory containing input tiles
#         output_path: Path to the directory where augmented tiles will be saved
#         max_tiles: Maximum number of tiles to process (default: None, processes all tiles)
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_path, exist_ok=True)
    
#     files = glob.glob(os.path.join(tile_path, "*.tif"))
    
#     # Filter out already augmented files
#     files = set(files) - set(glob.glob(os.path.join(tile_path, "*rot*")))
#     files = files - set(glob.glob(os.path.join(tile_path, "*flip*")))
    
#     # Take only the first max_tiles if specified, otherwise process all
#     if max_tiles is not None:
#         files = sorted(list(files))[:max_tiles]
#     else:
#         files = sorted(list(files))
    
#     print(f"Processing {len(files)} tiles for augmentation...")
#     print(f"Each tile will generate 7 additional augmented versions (8 total per tile)")
#     print(f"Output directory: {output_path}")
    
#     successful = 0
#     failed = 0
    
#     for i, file in enumerate(files):
#         filename = os.path.basename(file)
#         file_base, file_extension = os.path.splitext(filename)
        
#         # Generate filepaths for new tiles in the output directory
#         path_90 = os.path.join(output_path, file_base + "_rot90" + file_extension)
#         path_180 = os.path.join(output_path, file_base + "_rot180" + file_extension)
#         path_270 = os.path.join(output_path, file_base + "_rot270" + file_extension)
#         path_flip_name = os.path.join(output_path, file_base + "_flip" + file_extension)
#         path_flip_90 = os.path.join(output_path, file_base + "_rot90_flip" + file_extension)
#         path_flip_180 = os.path.join(output_path, file_base + "_rot180_flip" + file_extension)
#         path_flip_270 = os.path.join(output_path, file_base + "_rot270_flip" + file_extension)
        
#         try:
#             with rio.open(file, driver="GTiff") as src:
#                 # Read all bands
#                 num_bands = src.count
#                 bands = tuple(src.read(i) for i in range(1, num_bands + 1))
#                 meta = src.meta
                
#                 # Create augmented versions
#                 _augment_and_write(bands, path_90, meta, 1)  # 90°
#                 _augment_and_write(bands, path_180, meta, 2)  # 180°
#                 _augment_and_write(bands, path_270, meta, 3)  # 270°
                
#                 flipped_bands = _flip_bands(bands)
#                 _augment_and_write(flipped_bands, path_flip_name, meta, 0)  # flipped up/down
#                 _augment_and_write(flipped_bands, path_flip_90, meta, 1)  # flipped & 90°
#                 _augment_and_write(flipped_bands, path_flip_180, meta, 2)  # flipped & 180°
#                 _augment_and_write(flipped_bands, path_flip_270, meta, 3)  # flipped & 270°
                
#                 successful += 1
                
#                 if (i + 1) % 10 == 0:
#                     print(f"Processed {i + 1}/{len(files)} tiles")
                    
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")
#             failed += 1
#             continue
    
#     print(f"\n=== Augmentation Summary ===")
#     print(f"Successfully processed: {successful} tiles")
#     print(f"Failed: {failed} tiles")
#     print(f"Total augmented tiles created: {successful * 7} (7 per input tile)")
#     print(f"Total tiles after augmentation: {successful * 8} (including originals)")

# if __name__ == '__main__':
#     # Load configuration
#     try:
#         config = load_config()
#     except FileNotFoundError:
#         print("Error: config.json not found!")
#         print("Please copy config_template.json to config.json and update the paths.")
#         exit()
    
#     # Get paths from configuration
#     tiles_dir = get_tile_images_output_folder(config)
#     output_dir = get_augment_tiles_output_folder(config)
    
#     # Check if tiles directory exists
#     if not os.path.exists(tiles_dir):
#         print(f"Error: {tiles_dir} directory not found!")
#         print("Please run tiles.py first to generate tiles.")
#         exit()
    
#     # Check if there are any tiles to process
#     tile_files = glob.glob(os.path.join(tiles_dir, "*.tif"))
#     if not tile_files:
#         print(f"No .tif files found in {tiles_dir}")
#         print("Please run tiles.py first to generate tiles.")
#         exit()
    
#     print("=== Data Augmentation Workflow ===")
#     print(f"Input directory: {tiles_dir}")
#     print(f"Output directory: {output_dir}")
#     print(f"Found {len(tile_files)} total tiles")
#     # print("Will process the first 40 tiles (or all if less than 40)")
#     print("Will process ALL tiles (no limit)")
#     print("Each tile will generate 7 augmented versions:")
#     print("  - 90° rotation")
#     print("  - 180° rotation") 
#     print("  - 270° rotation")
#     print("  - Vertical flip")
#     print("  - Vertical flip + 90° rotation")
#     print("  - Vertical flip + 180° rotation")
#     print("  - Vertical flip + 270° rotation")
#     print()
    

#     # # Run augmentation
#     # augment_tiles(tiles_dir, output_dir, max_tiles=40)
#     # Run augmentation on ALL tiles
#     augment_tiles(tiles_dir, output_dir, max_tiles=None)
    
#     print("\nAugmentation complete!")
