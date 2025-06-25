"""
CRS (Coordinate Reference System) Checker
This script checks and displays the coordinate reference system of geospatial files.
It supports raster files (.tif, .tiff) and vector files (.shp, .geojson, .gpkg).

How to use:
1. Command line: python CoastlineExtraction/utils/check_crs.py your_file.tif
2. In Python: from CoastlineExtraction.utils.check_crs import check_crs; check_crs("your_file.tif")
"""

import sys
import os
import rasterio
import geopandas as gpd

__all__ = ['check_crs']

def check_crs(file_path, verbose=True):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.tif', '.tiff']:
            with rasterio.open(file_path) as src:
                crs = src.crs
                if verbose:
                    print(f"Raster file: {file_path}")
                    print(f"  CRS: {crs}\n")
                return crs
        elif ext in ['.shp', '.geojson', '.gpkg']:
            gdf = gpd.read_file(file_path)
            crs = gdf.crs
            if verbose:
                print(f"Vector file: {file_path}")
                print(f"  CRS: {crs}\n")
            return crs
        else:
            if verbose:
                print(f"Unsupported file type: {file_path}")
            return None
    except Exception as e:
        if verbose:
            print(f"Error reading {file_path}: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_crs.py <file1> <file2> ...")
        sys.exit(1)
    for file_path in sys.argv[1:]:
        check_crs(file_path, verbose=True)


if __name__ == "__main__":
    main() 



