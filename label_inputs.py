import rasterio as rio
from rasterio import merge
from rasterio.enums import Resampling
from rasterio.warp import reproject

import numpy as np


def add_labels(input_path, label_path):
    with rio.open(label_path, 'r') as label_src:
        print(label_src.shape)
        print(label_src.meta)
        labels = label_src.read(1)
        with rio.open(input_path, 'r') as input_src:
            print(input_src.shape)
            print(input_src.meta)
            input_meta = input_src.meta
            input_meta.update(count = 5)

            dst_raster = np.zeros((input_src.shape[0], input_src.shape[1]))
            reprojected_labels = reproject(labels, 
                                           dst_raster, 
                                           src_transform=label_src.transform, 
                                           dst_transform=input_src.transform, 
                                           src_crs=label_src.crs, 
                                           dst_crs=input_src.crs,
                                           resampling=Resampling.bilinear)
            print(reprojected_labels[0].shape)
            with rio.open('merged_img.tif', 'w', **input_meta) as dst:
                dst.write_band(1, input_src.read(1))
                dst.write_band(2, input_src.read(2))
                dst.write_band(3, input_src.read(3))
                dst.write_band(4, input_src.read(4))
                dst.write_band(5, reprojected_labels[0].astype(rio.uint16))


add_labels("C:\\Users\\kjcar\\Desktop\\268898_0369619_2016-10-15_0e14_BGRN_SR_clip.tif", "C:\\Users\\kjcar\\Desktop\\2016_08.tif")