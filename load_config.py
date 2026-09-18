"""
This script helps you load file paths from a config file for easy access in your project.

Steps to use:
1. Edit 'config_template.json' (or copy to 'config.json') in this folder. List your TIFF and SHP files.
2. In your script, import:
      from load_config import load_config, get_image_path, get_shapefile_path
3. Load the config:
      config = load_config()
4. Get the full path to a TIFF or SHP file by index:
      image_path = get_image_path(config, 0)        # First TIFF file
      shapefile_path = get_shapefile_path(config, 0) # First SHP file
5. Change the index to select other files.

Example:
    python load_config.py
    # Prints the first image and shapefile path from the config.
"""

import glob
import json
import os

# Try loading config.json first, fall back to config_template.json if not present
template_path = os.path.join(os.path.dirname(__file__), 'config_template.json')
config_path_default = os.path.join(os.path.dirname(__file__), 'config.json')
CONFIG_PATH = config_path_default if os.path.exists(config_path_default) else template_path

def load_config(config_path=CONFIG_PATH):
    """Load the configuration JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_image_path(config, index=0):
    """Get the full path to an image file by index."""
    return os.path.join(os.path.dirname(__file__), config['image_folder'], config['image_files'][index])

def get_raw_data_path(config, index=0):
    """Get the full path to a raw data file by index."""
    return os.path.join(os.path.dirname(__file__), config['raw_data_folder'], config['raw_data_files'][index])

def get_shapefile_path(config, index=0):
    """Get the full path to a shapefile by index."""
    return os.path.join(os.path.dirname(__file__), config['shapefile_folder'], config['shapefiles'][index])

def get_ground_truth_path(config, index=0):
    """Get the full path to a ground truth file by index."""
    return os.path.join(os.path.dirname(__file__), config['ground_truth_folder'], config['ground_truth_files'][index])

def get_aligned_data_path(config, index=0):
    """Get the full path to an aligned data file by index."""
    return os.path.join(os.path.dirname(__file__), config['aligned_data_folder'], config['aligned_data_files'][index])

def get_aligned_data_folder(config):
    """Get the full path to the aligned data folder."""
    return os.path.join(os.path.dirname(__file__), config['aligned_data_folder'])

def get_georeference_output_folder(config):
    """Get the full path to the georeference output folder."""
    return os.path.join(os.path.dirname(__file__), config['georeference_output_folder'])

def get_tile_images_output_folder(config):
    """Get the full path to the tile images output folder."""
    return os.path.join(os.path.dirname(__file__), config['tile_images_output_folder'])

def get_create_mask_output_folder(config):
    """Get the full path to the create mask output folder."""
    return os.path.join(os.path.dirname(__file__), config.get('create_mask_output_folder', 'processed_data/results_create_mask'))

def get_add_mask_band_output_folder(config):
    """Get the full path to the add mask band output folder."""
    return os.path.join(os.path.dirname(__file__), config.get('add_mask_band_output_folder', 'processed_data/results_add_mask_band'))

def get_augment_tiles_output_folder(config):
    """Get the full path to the augment tiles output folder."""
    return os.path.join(os.path.dirname(__file__), config.get('augment_tiles_output_folder', 'processed_data/results_augment_tiles'))

def get_georeference_files(config, count=None):
    """Get the full paths to the georeferenced TIFF files, sorted by filename.

    Pass count to take only the first N files (all of them by default).
    """
    files = sorted(glob.glob(os.path.join(get_georeference_output_folder(config), '*.tif')))
    return files if count is None else files[:count]

def get_training_config(config):
    """Get the training parameters block (empty dict if the config has none)."""
    return config.get('training', {})

def get_model_save_path(config):
    """Get the full path to the saved model file."""
    model_path = get_training_config(config).get('model_save_path', 'training_pipeline/unet_model.pth')
    return os.path.join(os.path.dirname(__file__), model_path)


# Example:
if __name__ == "__main__":
    config = load_config()
    print("First image path:", get_image_path(config, 0))
    print("First shapefile path:", get_shapefile_path(config, 0)) 
    print("First aligned data path:", get_aligned_data_path(config, 0))
    print("Aligned data folder:", get_aligned_data_folder(config))
    print("Georeference output folder:", get_georeference_output_folder(config))
    print("Tile images output folder:", get_tile_images_output_folder(config))
    print("Create mask output folder:", get_create_mask_output_folder(config))
    print("Add mask band output folder:", get_add_mask_band_output_folder(config))
    print("Augment tiles output folder:", get_augment_tiles_output_folder(config))
    print("Georeferenced files found:", len(get_georeference_files(config)))
    print("Model save path:", get_model_save_path(config)) 
