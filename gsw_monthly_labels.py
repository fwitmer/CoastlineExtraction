import ee
ee.Initialize()

def get_gsw_monthly(start_date, end_date, roi):
    gsw_images = ee.ImageCollection('JRC/GSW1_3/MonthlyHistory') \
                   .filterDate(start_date, end_date)             \
                   .filterBounds(roi)
    def clip_images(image):
        return image.clip(roi)
    clipped_images = gsw_images.map(clip_images)

    def count_nodata(image):
        count = image.lt(1)                               \
                     .reduceRegion(ee.Reducer.sum(), roi) \
                     .values()                            \
                     .get(0)
        return image.set('nodata_count', count)
    
    valid_images = clipped_images.map(count_nodata) \
                                 .filter(ee.Filter.lt('nodata_count', 1000))
    return valid_images

start_date = '2016-01-01'
end_date = '2021-08-31'
roi = ee.Geometry.Polygon([[[-162.8235626220703, 66.05622435812153],
                            [-162.674560546875, 66.05622435812153],
                            [-162.674560546875, 66.10883816429516],
                            [-162.8235626220703, 66.10883816429516],
                            [-162.8235626220703, 66.05622435812153]]])

results = get_gsw_monthly(start_date, end_date, roi)
dates = ee.List(results.aggregate_array('system:time_start')) \
          .map(lambda time_start:ee.Date(time_start).format('yyyy-MM')) \
          .getInfo()
print(dates)
