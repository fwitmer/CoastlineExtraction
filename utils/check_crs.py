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

__all__ = ['check_crs', 'crs_match']

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




def crs_match(file1, file2, verbose=True):
    """
    Checks if two geospatial files have the same Coordinate Reference System (CRS).

    Args:
        file1 (str): Path to the first geospatial file (raster or vector).
        file2 (str): Path to the second geospatial file (raster or vector).
        verbose (bool): If True, prints CRS information and errors.

    Returns:
        bool: True if both files have the same CRS, False otherwise.

    Note:
        If you reproject a file in memory, you must save it to disk and pass the path to the new file.
        Do NOT pass the original file path after in-memory CRS conversion, as it will not reflect the updated CRS.
    """
    crs1 = check_crs(file1, verbose=verbose)
    crs2 = check_crs(file2, verbose=verbose)

    if crs1 is None or crs2 is None:
        print("ERROR: One or both files do not have a valid CRS.")
        return False

    if crs1 != crs2:
        print("ERROR: CRS mismatch detected.")
        return False

    if verbose:
        print("CRS match confirmed.")
    return True



def main():
    if len(sys.argv) < 2:
        print("Usage: python check_crs.py <file1> <file2> ...")
        sys.exit(1)
    for file_path in sys.argv[1:]:
        check_crs(file_path, verbose=True)


if __name__ == "__main__":
    main() 



