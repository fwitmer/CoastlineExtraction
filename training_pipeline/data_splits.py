"""Utilities for keeping augmented samples together during dataset splits."""

import os
import random
import re


_AUGMENTATION_SUFFIX = re.compile(r"_(?:rot(?:90|180|270)(?:_flip)?|flip)$")


def _source_tile(image_path):
    """Return the source tile name shared by its original and augmented copies."""
    filename = os.path.splitext(os.path.basename(image_path))[0]
    return _AUGMENTATION_SUFFIX.sub("", filename)


def grouped_train_val_indices(image_mask_pairs, train_fraction=0.8, seed=42):
    """Split image-mask pair indices while keeping each tile's augmentations together.

    Groups are shuffled with a local seeded RNG, then a group boundary nearest to
    the requested validation size is selected. At least one source tile is kept
    in each split so validation cannot silently be empty.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    groups = {}
    for index, (image_path, _mask_path) in enumerate(image_mask_pairs):
        groups.setdefault(_source_tile(image_path), []).append(index)

    if len(groups) < 2:
        raise ValueError("At least two distinct source tiles are required for a grouped split")

    group_items = sorted(groups.items())
    random.Random(seed).shuffle(group_items)

    target_val_size = len(image_mask_pairs) * (1 - train_fraction)
    prefix_sizes = []
    cumulative_size = 0
    for _name, indices in group_items[:-1]:
        cumulative_size += len(indices)
        prefix_sizes.append(cumulative_size)
    best_prefix_length = min(
        range(1, len(group_items)),
        key=lambda length: abs(prefix_sizes[length - 1] - target_val_size),
    )
    val_groups = group_items[:best_prefix_length]

    val_group_names = {name for name, _indices in val_groups}
    train_indices = [
        index for name, indices in groups.items() if name not in val_group_names
        for index in indices
    ]
    val_indices = [index for _name, indices in val_groups for index in indices]
    return train_indices, val_indices
