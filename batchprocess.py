import os
import sys
import rastertools

rootdir = os.getcwd()
# this script will handle batch processing of files including:
#       * parsing date information from filenames
#       * gathering associated metadata for each image
#       * outputting finished files into a new directory structure
files_to_be_processed = []
for root, subdirs, files in os.walk(rootdir + "/data"):
    if files:
        for f in files:
            if f.endswith(".tif") and f.find("_SR_") != -1:
                files_to_be_processed.append((root + "/" +f, f))

files_to_be_processed.sort()
for f in files_to_be_processed:
    print("Path:{} Filename:{}".format(f[0],f[1]))