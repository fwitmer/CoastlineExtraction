import rasterio as rio
from rasterio import windows

from itertools import product

from matplotlib import pyplot as plt

import os

# adapted from https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
def make_tiles(image, tile_height=512, tile_width=512):
    with rio.open(image) as src:
        filepath, filename = os.path.split(image)
        file_base, file_extension = os.path.splitext(filename)
        meta = src.meta.copy()
        num_cols, num_rows = src.meta['width'], src.meta['height']
        offsets = product(range(0, num_cols, tile_height//2), range(0, num_rows, tile_width//2))
        tiles = []
        for col_off, row_off in offsets:
            curr_window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
            curr_transform = windows.transform(curr_window, src.transform)
            tiles.append((curr_window, curr_transform))
        for i in range(len(tiles)):
            window, transform = tiles[i]
            meta['transform'] = transform
            meta['width'] = tile_width
            meta['height'] = tile_height
            out_name = file_base + "_" + str(i) + file_extension
            out_path = os.path.join(filepath, out_name)
            with rio.open(out_path, 'w', **meta) as dst:
                dst.write(src.read(window=window))

        
# example usage
make_tiles("data/merged_img.tif")