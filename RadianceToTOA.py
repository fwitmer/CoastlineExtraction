import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
import numpy as np
from xml.dom import minidom


filepath = "Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/"
img_filename = "20160908_212941_0e0f_1B_AnalyticMS.tif"
xml_filename = "20160908_212941_0e0f_1B_AnalyticMS_metadata.xml"

img = filepath + img_filename
xml_file = filepath + xml_filename

# loading bands in radiance
print("Opening", img_filename, "to read in band data:", end=" ")
with rasterio.open(img) as src:
    print("DONE")
    print("\tReading BLUE:", end=" ")
    blue_band_radiance = src.read(1)   # band 1 - blue
    print("DONE")

    print("\tReading GREEN:", end=" ")
    green_band_radiance = src.read(2)  # band 2 - green
    print("DONE")

    print("\tReading RED:", end=" ")
    red_band_radiance = src.read(3)    # band 3 - red
    print("DONE")

    print("\tReading NIR:", end=" ")
    nir_band_radiance = src.read(4)    # band 4 - near-infrared
    print("DONE")
print("\tBlue band:")
print("\t\tMIN: {} MAX: {}".format(np.amin(blue_band_radiance), np.amax(blue_band_radiance)))
print("\tGreen band:")
print("\t\tMIN: {} MAX: {}".format(np.amin(green_band_radiance), np.amax(green_band_radiance)))
print("\tRed band:")
print("\t\tMIN: {} MAX: {}".format(np.amin(red_band_radiance), np.amax(red_band_radiance)))
print("\tNIR band:")
print("\t\tMIN: {} MAX: {}".format(np.amin(nir_band_radiance), np.amax(nir_band_radiance)))
print()

# parsing the XML metadata to determine coefficients
print("Parsing", xml_filename, "for reflectance coefficients:", end=" ")
xmldoc = minidom.parse(xml_file)
nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
coeffs = {}
for node in nodes:
    band_num = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
    if band_num in ['1', '2', '3', '4']:
        i = int(band_num)
        value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
        coeffs[i] = float(value)
print("DONE")

print("\tReflectance coefficients:")
print("\t\tBLUE: {}".format(coeffs[1]))
print("\t\tGREEN: {}".format(coeffs[2]))
print("\t\tRED: {}".format(coeffs[3]))
print("\t\tNIR: {}".format(coeffs[4]))
print()

# converting Digital Number (DN) to TOA reflectance
print("Converting to top-of-atmosphere (TOA) reflectance:", end=" ")
blue_band_reflectance = blue_band_radiance * coeffs[1]
green_band_reflectance = green_band_radiance * coeffs[2]
red_band_reflectance = red_band_radiance * coeffs[3]
nir_band_reflectance = nir_band_radiance * coeffs[4]
print("DONE")

print("\tBlue band:")
print("\t\tMIN: {} MAX: {}".format(np.amin(blue_band_reflectance), np.amax(blue_band_reflectance)))
print("\tGreen band:")
print("\t\tMIN: {} MAX: {}".format(np.amin(green_band_reflectance), np.amax(green_band_reflectance)))
print("\tRed band:")
print("\t\tMIN: {} MAX: {}".format(np.amin(red_band_reflectance), np.amax(red_band_reflectance)))
print("\tNIR band:")
print("\t\tMIN:{} MAX: {}".format(np.amin(nir_band_reflectance), np.amax(nir_band_reflectance)))
