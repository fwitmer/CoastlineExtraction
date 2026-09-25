import unittest

from training_pipeline.data_splits import _source_tile, grouped_train_val_indices


class GroupedSplitTests(unittest.TestCase):
    def setUp(self):
        # Each source tile has its original plus all seven offline augmentations.
        self.pairs = [
            (f"/data/tile_{tile}{suffix}.tif", f"/data/tile_{tile}_mask.tif")
            for tile in range(10)
            for suffix in ("", "_rot90", "_rot180", "_rot270", "_flip",
                           "_rot90_flip", "_rot180_flip", "_rot270_flip")
        ]

    def test_augmentations_from_one_tile_never_cross_splits(self):
        train, val = grouped_train_val_indices(self.pairs)
        train_sources = {self.pairs[index][0].split("/")[-1].split("_")[1] for index in train}
        val_sources = {self.pairs[index][0].split("/")[-1].split("_")[1] for index in val}

        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(train_sources.isdisjoint(val_sources))
        self.assertEqual(len(train) + len(val), len(self.pairs))

    def test_split_is_repeatable_and_independent_of_pair_order(self):
        first = grouped_train_val_indices(self.pairs)
        reordered_pairs = list(reversed(self.pairs))
        reordered = grouped_train_val_indices(reordered_pairs)
        first_val_sources = {_source_tile(self.pairs[index][0]) for index in first[1]}
        reordered_val_sources = {
            _source_tile(reordered_pairs[index][0]) for index in reordered[1]
        }
        self.assertEqual(first_val_sources, reordered_val_sources)

    def test_rejects_invalid_fraction_and_single_source(self):
        with self.assertRaises(ValueError):
            grouped_train_val_indices(self.pairs, train_fraction=1)
        with self.assertRaises(ValueError):
            grouped_train_val_indices(self.pairs[:8])


if __name__ == "__main__":
    unittest.main()
