#######################################################
# Author: Kristopher Carroll
# Last Updated: 9/14/2021
# Description:
#   Provides functionality to use Google Earth Engine's Python API to extract monthly
#   surface water labels from the Global Surface Water Dataset (GSW). These monthly
#   water labels are filtered by date range, clipped to the region of interest (ROI),
#   and validated to have less than the specified number of NoData pixels.
# Inputs:
#   start_date - string in the format 'yyyy-MM-dd'
#   end_date - string in the format 'yyyy-MM-dd'
#   roi - ee.Geometry.Polygon representing the ROI
#   nodata_threshold - integer representing the maximum allowable number of NoData pixels
# Outputs:
#   filtered, clipped and cleaned ee.ImageCollection matching the input requirements
######################################################
import ee
from geetools import batch
ee.Initialize()

def get_gsw_monthly(start_date, end_date, roi, nodata_threshold):
    gsw_images = ee.ImageCollection('JRC/GSW1_3/MonthlyHistory') \
                   .filterDate(start_date, end_date) \
                   .filterBounds(roi)
    def clip_images(image):
        return image.clip(roi)
    clipped_images = gsw_images.map(clip_images)

    def count_nodata(image):
        count = image.lt(1) \
                     .reduceRegion(ee.Reducer.sum(), roi) \
                     .values() \
                     .get(0)
        return image.set('nodata_count', count)
    
    valid_images = clipped_images.map(count_nodata) \
                                 .filter(ee.Filter.lt('nodata_count', nodata_threshold))
    return valid_images

# wrapper for the batch.Download.ImageCollection.toDrive function found here:
# https://github.com/gee-community/gee_tools
def export_images(image_collection, folder, region):
    batch.Export.imagecollection.toDrive(image_collection, folder, scale=30, dataType='uint8', region=region, verbose=True)

# Example code
start_date = '2015-01-01'
end_date = '2015-12-31'
roi = ee.Geometry.Polygon([[[-162.8235626220703, 66.05622435812153],
                            [-162.674560546875, 66.05622435812153],
                            [-162.674560546875, 66.10883816429516],
                            [-162.8235626220703, 66.10883816429516],
                            [-162.8235626220703, 66.05622435812153]]])
nodata_threshold = 1000

results = get_gsw_monthly(start_date, end_date, roi, nodata_threshold)
dates = ee.List(results.aggregate_array('system:time_start')) \
          .map(lambda time_start:ee.Date(time_start).format('yyyy-MM')) \
          .getInfo()
print(len(dates), "images returned at the following dates:")
print(dates)

# Exporting resulting images to Google Drive
'''
folder = 'GSW_Monthly_Labels'
export_images(results, folder, roi)
'''