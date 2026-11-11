"""
Naming rules for pairing image tiles with their NDWI mask tiles.

`tile_images.py` writes image tiles as `<base>_<NN>-of-<M>.tif` and mask tiles as
`<base>_concatenated_ndwi_mask_<NN>-of-<M>.tif`. `augment_tiles.py` appends the same
augmentation suffix to both names: `_rot90`, `_rot180`, `_rot270`, `_flip`,
`_rot90_flip`, `_rot180_flip` or `_rot270_flip`.
"""

import re

MASK_TAG = "_concatenated_ndwi_mask_"

_TILE_NAME = re.compile(
    r"^(?P<base>.+)_(?P<tile>\d+-of-\d+)"
    r"(?P<augmentation>(?:_rot(?:90|180|270))?(?:_flip)?)\.tif$"
)


def mask_name_for_tile(image_name):
    """
    Return the mask file name that belongs to an image tile file name.

    Args:
        image_name (str): File name (not a path), e.g. `<base>_01-of-110_rot90_flip.tif`.

    Returns:
        str or None: The matching mask name, e.g.
        `<base>_concatenated_ndwi_mask_01-of-110_rot90_flip.tif`, or None if the name does
        not follow the tile naming scheme (or is already a mask name).
    """
    if MASK_TAG in image_name:
        return None
    match = _TILE_NAME.match(image_name)
    if match is None:
        return None
    return f"{match['base']}{MASK_TAG}{match['tile']}{match['augmentation']}.tif"
