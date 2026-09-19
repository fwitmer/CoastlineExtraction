import os
import sys
import pytest

# Ensure root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_config import (
    load_config,
    get_image_path,
    get_shapefile_path,
    get_ground_truth_path,
    get_aligned_data_folder,
    get_georeference_output_folder,
    get_create_mask_output_folder,
    get_tile_images_output_folder,
    get_augment_tiles_output_folder,
    get_add_mask_band_output_folder,
    get_training_config,
    get_model_save_path
)

def test_load_config():
    config = load_config()
    assert isinstance(config, dict)
    assert 'image_folder' in config
    assert 'shapefile_folder' in config

def test_get_image_path():
    config = load_config()
    path = get_image_path(config, 0)
    assert isinstance(path, str)
    assert path.endswith('.tif')

def test_get_shapefile_path():
    config = load_config()
    path = get_shapefile_path(config, 0)
    assert isinstance(path, str)
    assert path.endswith('.shp')

def test_output_folders():
    config = load_config()
    assert 'results_batch_align' in get_aligned_data_folder(config)
    assert 'results_georeference' in get_georeference_output_folder(config)
    assert 'results_create_mask' in get_create_mask_output_folder(config)
    assert 'results_tile_images' in get_tile_images_output_folder(config)
    assert 'results_augment_tiles' in get_augment_tiles_output_folder(config)
    assert 'results_add_mask_band' in get_add_mask_band_output_folder(config)
