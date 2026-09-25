"""Mask preprocessing shared by segmentation datasets."""

from PIL import Image


def resize_binary_mask(mask, image_size):
    """Resize a PIL mask to (height, width) without creating new label values."""
    height, width = image_size
    return mask.resize((width, height), resample=Image.Resampling.NEAREST)
