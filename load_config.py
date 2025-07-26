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

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_template.json')

def load_config(config_path=CONFIG_PATH):
    """Load the configuration JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_image_path(config, index=0):
    """Get the full path to an image file by index."""
    return os.path.join(os.path.dirname(__file__), config['image_folder'], config['image_files'][index])

def get_shapefile_path(config, index=0):
    """Get the full path to a shapefile by index."""
    return os.path.join(os.path.dirname(__file__), config['shapefile_folder'], config['shapefiles'][index])

def get_ground_truth_path(config, index=0):
    """Get the full path to a ground truth file by index."""
    return os.path.join(os.path.dirname(__file__), config['ground_truth_folder'], config['ground_truth_files'][index])



# Example:
if __name__ == "__main__":
    config = load_config()
    print("First image path:", get_image_path(config, 0))
    print("First shapefile path:", get_shapefile_path(config, 0)) 