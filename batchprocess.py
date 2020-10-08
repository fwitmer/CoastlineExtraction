import os
import sys
import datetime
import rastertools as rt
import rasterio
from rasterio.merge import merge

rootdir = os.getcwd()
# this script will handle batch processing of files including:
#       * parsing date information from filenames
#       * gathering associated metadata for each image
#       * outputting finished files into a new directory structure

# TODO: Add method to determine if files have already been processed to prevent reprocessing
# TODO: Add optional flag to reprocess all images
files_to_be_processed = []
for root, subdirs, files in os.walk(rootdir + "/data"):
    # skip any files found in the output directory
    if root.find("output") != -1:
        continue
    # have files, check to see if they're the ones we want
    if files:
        for f in files:
            if f.endswith(".tif") and f.find("_SR_") != -1:
                pathname = root + "/" + f
                files_to_be_processed.append((pathname, f)) # appending (path, filename)

# if we have files, check output directory structure
if files_to_be_processed:
    if not os.path.isdir("data/output"):
        os.mkdir("data/output")
    
    file_groups = {}

    files_to_be_processed.sort()
    for f in files_to_be_processed:
        if f[0].find("8bit") != -1:
            continue
        file_time = datetime.datetime.strptime(f[1][:15],'%Y%m%d_%H%M%S')
        file_year = str(file_time.year) # forcing into a string for use in pathing later
        file_month = file_time.strftime("%B") # string version of month
        file_day = file_time.day

        # grouping raster patches together for merging later
        if file_year not in file_groups.keys():
            file_groups[file_year] = {}
        if file_month not in file_groups[file_year].keys():
            file_groups[file_year][file_month] = {}
        if file_day not in file_groups[file_year][file_month].keys():
            file_groups[file_year][file_month][file_day] = []
        file_groups[file_year][file_month][file_day].append(f[0])


        # process directory structure if not already made
        if not os.path.isdir("data/output/{}".format(file_year)):
            os.mkdir("data/output/{}".format(file_year))
        if not os.path.isdir("data/output/{}/{}".format(file_year, file_month)):
            os.mkdir("data/output/{}/{}".format(file_year, file_month))

        # outfile_base = f[1][:15] + "_AnalyticMS_SR"
        
        # ndwi_outfile = "data/output/{}/{}/{}".format(file_year, file_month, outfile_base + "_NDWI.tif")
        # ndwi_class_outfile = "data/output/{}/{}/{}".format(file_year, file_month, outfile_base + "_NDWI_classified.tif")

        # ndwi = rt.calculate_ndwi(f[0], ndwi_outfile, plot=False)
        # ndwi_class = rt.ndwi_classify(f[0], ndwi_class_outfile, plot=False)
    
    for year, month_grp in file_groups.items():
        for month, day_grp in file_groups[year].items():
            for day, files in file_groups[year][month].items():
                files_to_be_merged = []
                for f in files:
                    src = rasterio.open(f)
                    files_to_be_merged.append(src)
                mosaic, out_transform = merge(files_to_be_merged)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff",
                                 "height": mosaic.shape[1],
                                 "width": mosaic.shape[2],
                                 "transform": out_transform})
                out_fp = "data/output/{year}/{month}/{year}{month}{day}_AnalyticMS_SR_merged.tif".format(year=year, month=month, day=day)
                with rasterio.open(out_fp, "w", **out_meta) as dst:
                    dst.write(mosaic)