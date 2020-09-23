import os
import sys
import datetime
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
                pathname = root + "/" + f
                files_to_be_processed.append((pathname, f)) # appending (path, filename)

files_to_be_processed.sort()
for f in files_to_be_processed:
    file_time = datetime.datetime.strptime(f[1][:14],'%Y%m%d_%H%M%S')
    file_year = str(file_time.year) # forcing into a string for use in pathing later
    file_month = file_time.strftime("%B") # string version of month
