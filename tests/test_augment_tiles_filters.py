import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import cv2  # noqa: F401  (imported by augment_tiles.py)
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "data_preprocessing", "augment_tiles.py")


def load_augment_tiles_module():
    """Load augment_tiles.py by path (data_preprocessing/ is not an importable package here)."""
    # augment_tiles.py imports path helpers from load_config; stub them so this test does not
    # depend on which helpers load_config.py currently defines.
    stub = types.ModuleType("load_config")
    for name in ("load_config", "get_tile_images_output_folder", "get_augment_tiles_output_folder"):
        setattr(stub, name, lambda *args, **kwargs: None)
    spec = importlib.util.spec_from_file_location("augment_tiles_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"load_config": stub}):
        spec.loader.exec_module(module)
    return module


def write_tile(path, bands, dtype):
    profile = {
        "driver": "GTiff", "height": 4, "width": 4, "count": bands, "dtype": dtype,
        "crs": "EPSG:32603", "transform": from_origin(0, 4, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        for band in range(1, bands + 1):
            dst.write(np.arange(16, dtype=dtype).reshape(4, 4), band)


@unittest.skipIf(rasterio is None, "rasterio, numpy and opencv are required")
class AugmentTilesFilterTests(unittest.TestCase):
    def test_directory_names_do_not_affect_which_tiles_are_used(self):
        module = load_augment_tiles_module()
        # Directory names containing "_rot", "_flip" or "mask" used to make every tile get filtered out.
        for directory in ("tiles", "my_rotation_data", "tiles_flip", "masks_v2"):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as root:
                tile_dir = os.path.join(root, directory)
                out_dir = os.path.join(root, "out")
                os.makedirs(tile_dir)
                write_tile(os.path.join(tile_dir, "scene_01-of-10.tif"), 2, "uint16")
                write_tile(os.path.join(tile_dir, "scene_concatenated_ndwi_mask_01-of-10.tif"), 1, "uint8")

                module.augment_tiles(tile_dir, out_dir)

                # 7 augmented variants of the image and of its mask
                self.assertEqual(len(os.listdir(out_dir)), 14)

    def test_augmented_tiles_and_masks_in_the_input_folder_are_not_reprocessed(self):
        module = load_augment_tiles_module()
        with tempfile.TemporaryDirectory() as root:
            tile_dir = os.path.join(root, "tiles")
            out_dir = os.path.join(root, "out")
            os.makedirs(tile_dir)
            write_tile(os.path.join(tile_dir, "scene_01-of-10.tif"), 2, "uint16")
            write_tile(os.path.join(tile_dir, "scene_concatenated_ndwi_mask_01-of-10.tif"), 1, "uint8")
            # Output of an earlier run: augmented images together with their masks.
            write_tile(os.path.join(tile_dir, "scene_01-of-10_rot90.tif"), 2, "uint16")
            write_tile(os.path.join(tile_dir, "scene_concatenated_ndwi_mask_01-of-10_rot90.tif"), 1, "uint8")
            write_tile(os.path.join(tile_dir, "scene_01-of-10_flip.tif"), 2, "uint16")
            write_tile(os.path.join(tile_dir, "scene_concatenated_ndwi_mask_01-of-10_flip.tif"), 1, "uint8")

            module.augment_tiles(tile_dir, out_dir)

            self.assertEqual(len(os.listdir(out_dir)), 14)
            self.assertNotIn("scene_01-of-10_rot90_rot90.tif", os.listdir(out_dir))
