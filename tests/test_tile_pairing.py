import unittest

from training_pipeline.tile_pairing import mask_name_for_tile

# Suffixes written by data_preprocessing/augment_tiles.py (the un-augmented tile has none).
AUGMENTATION_SUFFIXES = (
    "",
    "_rot90",
    "_rot180",
    "_rot270",
    "_flip",
    "_rot90_flip",
    "_rot180_flip",
    "_rot270_flip",
)

BASES = (
    "1474782_0369619_2018-06-02_0f51_BGRN_SR_clip_aligned_GeoRegistered",
    "scene",
    "scene_a",
)


class MaskNameForTileTests(unittest.TestCase):
    def test_every_augmentation_maps_to_its_mask(self):
        for base in BASES:
            for suffix in AUGMENTATION_SUFFIXES:
                with self.subTest(base=base, suffix=suffix):
                    image_name = f"{base}_01-of-110{suffix}.tif"
                    expected = f"{base}_concatenated_ndwi_mask_01-of-110{suffix}.tif"
                    self.assertEqual(mask_name_for_tile(image_name), expected)

    def test_tile_number_with_more_digits(self):
        self.assertEqual(
            mask_name_for_tile("scene_104-of-1100_rot180_flip.tif"),
            "scene_concatenated_ndwi_mask_104-of-1100_rot180_flip.tif",
        )

    def test_names_outside_the_tile_scheme_return_none(self):
        for name in (
            "scene.tif",
            "scene_01-of-110_rot45.tif",
            "scene_01-of-110_flip_rot90.tif",
            "scene_01-of-110_rot90.png",
            "scene_01-of-110_extra.tif",
        ):
            with self.subTest(name=name):
                self.assertIsNone(mask_name_for_tile(name))

    def test_mask_names_are_not_treated_as_images(self):
        self.assertIsNone(mask_name_for_tile("scene_concatenated_ndwi_mask_01-of-110_rot90.tif"))
