import rasterio
import numpy as np
from xml.dom import minidom
filepath = "Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/"
img_filename = "20160908_212941_0e0f_1B_AnalyticMS.tif"
xml_filename = "20160908_212941_0e0f_1B_AnalyticMS_metadata.xml"

img = filepath + img_filename
xml_file = filepath + xml_filename

# loading bands in radiance
with rasterio.open(img) as src:
    blue_band_radiance = src.read(1)
    green_band_radiance = src.read(2)
    red_band_radiance = src.read(3)
    nir_band_radiance = src.read(4)

# parsing the XML metadata to determine coefficients
xmldoc = minidom.parse(xml_file)
nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

coeffs = {}
for node in nodes:
    band_num = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
    if band_num in ['1', '2', '3', '4']:
        i = int(band_num)
        value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
        coeffs[i] = float(value)

print("Conversion coefficients:", coeffs)