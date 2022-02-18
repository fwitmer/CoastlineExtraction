import rasterio as rio
from rasterio import windows

from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import os
import glob

# adapted from https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
def make_tiles(image, tile_height=512, tile_width=512, skip_no_data=False):
    with rio.open(image) as src:
        filepath, filename = os.path.split(image)
        file_base, file_extension = os.path.splitext(filename)
        meta = src.meta.copy()
        num_cols, num_rows = src.meta['width'], src.meta['height']
        overall_window = windows.Window(col_off=0, row_off=0, width=num_cols, height=num_rows)
        offsets = product(range(0, num_cols, tile_height//2), range(0, num_rows, tile_width//2))
        tiles = []
        for col_off, row_off in offsets:
            curr_window = windows.Window(col_off=col_off, row_off=row_off, width=tile_width, height=tile_height)
            curr_transform = windows.transform(curr_window, src.transform)
            tiles.append((curr_window.intersection(overall_window), curr_transform))
        for i in range(len(tiles)):
            window, transform = tiles[i]
            meta['transform'] = transform
            meta['width'] = tile_width
            meta['height'] = tile_height
            window_data = src.read(window=window)
            # optionally skip tiles with no data values
            if skip_no_data:
                if 0 in window_data[..., :-1]:
                    continue
            out_name = file_base + "_" + str(i + 1).zfill(2) + "-of-" + str(len(tiles)) + file_extension
            out_path = os.path.join("data/tiles/", out_name)
            with rio.open(out_path, 'w', **meta) as dst:
                dst.write(src.read(window=window))

def _augment_and_write(bands, outpath, metadata, rotations=1):
    with rio.open(outpath, 'w', **metadata) as dst:
        for i in range(len(bands)):
            dst.write(np.rot90(bands[i], rotations), i+1)

def _flip_bands(bands):
    flipped_bands = [np.flipud(band) for band in bands]
    return flipped_bands

# takes the path to all image tiles and creates tiles that are rotated 90°, 180° and 270° as well as their flipped counterparts
# this results in 8 tiles for every input tile (including the input tile)
def augment_tiles(tile_path):
    files = glob.glob(tile_path + "*.tif")
    files = set(files) - set(glob.glob(tile_path + "*rot*"))
    files = set(files) - set(glob.glob(tile_path + "*flip*")) 
    for file in files:
        filename = os.path.basename(file)
        file_base, file_extension = os.path.splitext(filename)
        # generating filepaths for new tiles
        path_90 = tile_path + file_base + "_rot90" + file_extension
        path_180 = tile_path + file_base + "_rot180" + file_extension
        path_270 = tile_path + file_base + "_rot270" + file_extension
        path_flip_name = tile_path + file_base + "_flip" + file_extension
        path_flip_90 = tile_path + file_base + "_rot90_flip" + file_extension
        path_flip_180 = tile_path + file_base + "_rot180_flip" + file_extension
        path_flip_270 = tile_path + file_base + "_rot270_flip" + file_extension
         
        with rio.open(file, driver="GTiff") as src:
            # band_1 = src.read(1)
            # band_2 = src.read(2)
            # band_3 = src.read(3)
            # band_4 = src.read(4)
            # band_5 = src.read(5)
            bands = (src.read(1), src.read(2), src.read(3), src.read(4), src.read(5))
            meta = src.meta
            
            _augment_and_write(bands, path_90, meta, 1) # 90°
            _augment_and_write(bands, path_180, meta, 2) # 180°
            _augment_and_write(bands, path_270, meta, 3) # 270°
            flipped_bands = _flip_bands(bands)
            _augment_and_write(flipped_bands, path_flip_name, meta, 0) # flipped up/down
            _augment_and_write(flipped_bands, path_flip_90, meta, 1) # flipped & 90°
            _augment_and_write(flipped_bands, path_flip_180, meta, 2) # flipped & 180°
            _augment_and_write(flipped_bands, path_flip_270, meta, 3) # flipped & 270°


        
# example usage
if __name__ == '__main__':
    files = glob.glob("data/labeled_inputs/*.tif")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as p:
        p.map(make_tiles, files)

# augment_tiles("data/tiles/")