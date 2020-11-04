import warnings
import os
import rasterio
import cv2
from rasterio.plot import show_hist
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from xml.dom import minidom
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def radiance_to_toa(rasterfile, xmlfile, outfile=None, plot=False, verbose=False):
    raster_filepath = os.path.dirname(rasterfile) + "/"
    raster_filename = os.path.basename(rasterfile)
    xml_filepath = os.path.dirname(xmlfile) + "/"
    xml_filename = os.path.basename(xmlfile)

    img = raster_filepath + raster_filename
    xml_file = xml_filepath + xml_filename

    # the following code for converting to TOA reflectance was largely adapted from a tutorial on Planet
    # source: https://developers.planet.com/tutorials/convert-planetscope-imagery-from-radiance-to-reflectance/
    # loading bands in radiance
    print("Opening", raster_filename, "to read in band data:", end=" ")
    with rasterio.open(img, driver="GTiff") as src:
        kwargs = src.meta
        kwargs.update(dtype=rasterio.float64, count=4)
        print("DONE")
        if verbose: print("\tReading BLUE:", end=" ")
        blue_band_radiance = src.read(1)   # band 1 - blue
        if verbose: print("DONE")

        if verbose: print("\tReading GREEN:", end=" ")
        green_band_radiance = src.read(2)  # band 2 - green
        if verbose: print("DONE")

        if verbose: print("\tReading RED:", end=" ")
        red_band_radiance = src.read(3)    # band 3 - red
        if verbose: print("DONE")

        if verbose: print("\tReading NIR:", end=" ")
        nir_band_radiance = src.read(4)    # band 4 - near-infrared
        if verbose: print("DONE")
    if verbose:
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

    if verbose:
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

    if verbose:
        print("\tBlue band:")
        print("\t\tMIN: {} MAX: {}".format(np.amin(blue_band_reflectance), np.amax(blue_band_reflectance)))
        print("\tGreen band:")
        print("\t\tMIN: {} MAX: {}".format(np.amin(green_band_reflectance), np.amax(green_band_reflectance)))
        print("\tRed band:")
        print("\t\tMIN: {} MAX: {}".format(np.amin(red_band_reflectance), np.amax(red_band_reflectance)))
        print("\tNIR band:")
        print("\t\tMIN: {} MAX: {}".format(np.amin(nir_band_reflectance), np.amax(nir_band_reflectance)))
    print()

    # writing the TOA reflectance image to disk
    if outfile:
        out_filename = outfile
    else:
        out_filename = raster_filename.split(sep=".")[0] + "_TOAreflectance.tif"
    print("Saving TOA reflectance as", out_filename, ":", end=" ")

    with rasterio.open(raster_filepath + out_filename, 'w', **kwargs) as dst:
        dst.write_band(1, blue_band_reflectance.astype(rasterio.float64))
        dst.write_band(2, green_band_reflectance.astype(rasterio.float64))
        dst.write_band(3, red_band_reflectance.astype(rasterio.float64))
        dst.write_band(4, nir_band_reflectance.astype(rasterio.float64))
    print("DONE")
    print()

    if plot:
        labels = ["Blue Band Reflectance", "Green Band Reflectance", "Red Band Reflectance", "NIR Band Reflectance"]
        bands = [blue_band_reflectance, green_band_reflectance, red_band_reflectance, nir_band_reflectance]
        plot_raster(bands, labels)

    return raster_filepath + out_filename


def calculate_ndwi(rasterfile, outfile=None, plot=False):
    raster_filepath = os.path.dirname(rasterfile) + "/"
    raster_filename = os.path.basename(rasterfile)

    img = raster_filepath + raster_filename

    print("Opening", raster_filename, "to read in band data:", end=" ")
    with rasterio.open(img, driver="GTiff") as src:
        kwargs = src.meta
        kwargs.update(dtype=rasterio.float32, count=1)

        green_band = src.read(2).astype(rasterio.float64)  # band 2 - green
        nir_band = src.read(4).astype(rasterio.float64)    # band 4 - NIR
        print("DONE\n")

    print("Calculating NDWI:", end=" ")
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (green_band - nir_band) / (green_band + nir_band)
    print("DONE\n")
    if outfile:
        out_filename = outfile
    else:
        out_filename = raster_filepath + raster_filename.split(sep=".")[0] + "_NDWI.tif"
    print("Saving calculated NDWI image as", out_filename, ":", end=" ")
    with rasterio.open(out_filename, 'w', **kwargs) as dst:
        dst.write_band(1, ndwi.astype(rasterio.float32))
        print("DONE\n")

    if plot:
        bands = [ndwi]
        labels = ["NDWI (Normalized Difference Water Index"]
        plot_raster(bands, labels)

        show_hist(ndwi, bins=100, stacked=False, alpha=0.3, histtype='stepfilled', title="NDWI Values")
    return out_filename


def ndwi_classify(rasterfile, outfile=None, plot=False):
    threshold = get_otsu_threshold(rasterfile, normalized=True)
    raster_filepath = os.path.dirname(rasterfile) + "/"
    raster_filename = os.path.basename(rasterfile)

    img = raster_filepath + raster_filename
    print("Opening", raster_filename, "for NDWI water classification:", end=" ")
    with rasterio.open(img, driver="GTiff") as src:
        kwargs = src.meta
        kwargs.update(dtype=rasterio.int8, count=1)

        ndwi = src.read(1)
        print("DONE\n")
    # TODO: update this with dynamically calculated thresholds
    print("Classifying water based on NDWI threshold of ({}):".format(threshold), end=" ")
    classified_raster = np.where(ndwi >= threshold,  # if pixel value >= threshold, new raster value is 1 else 0
                                 1,
                                 0)
    print("DONE\n")
    if outfile:
        out_filename = outfile
    else:
        out_filename = raster_filepath + raster_filename.split(sep=".")[0] + "_classified.tif"
    print("Saving classified raster as", out_filename, ":", end=" ")
    with rasterio.open(out_filename, 'w', **kwargs) as dst:
        dst.nodata = 9001
        dst.write_band(1, classified_raster.astype(rasterio.int8))
    print("DONE\n")

    if plot:
        bands = [classified_raster]
        labels = ["Water Classification Map"]
        plot_raster(bands, labels)

    return raster_filepath + out_filename

# TODO: Convert to handle output error on server with no display
def plot_raster(bands, labels):

    for band, label in zip(bands, labels):
        print("Generating", label, "plot:", end=" ")
        min_val = np.nanmin(band)
        max_val = np.nanmax(band)
        mid = np.nanmean(band)
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        cax = ax.imshow(band, cmap='Greys_r', clim=(min_val, max_val),
                        norm=MidpointNormalize(midpoint=mid, vmin=min_val, vmax=max_val))
        ax.axis('off')
        ax.set_title(label, fontsize=18, fontweight='bold')
        cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)

        plt.show()
        print("DONE")
    print()


def get_otsu_threshold(path, reduce_noise = False, normalized = False):
    with rasterio.open(path, driver="GTiff") as src:
        kwargs = src.meta
        kwargs.update(dtype=rasterio.uint8, count=1)
        ndwi = src.read(1)
    ndwi_8_bit = (ndwi * 127) + 127
    out_filename = path.split(sep=".")[0] + "_8bit.tif"
    with rasterio.open(out_filename, mode='w', **kwargs) as dst:
        dst.write_band(1, ndwi_8_bit.astype(rasterio.uint8))

    image = cv2.imread(out_filename, 0) # 0 is grayscale mode
    if reduce_noise:
        image = cv2.GaussianBlur(image, (5,5), 0)

    otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    otsu_threshold_float = float((otsu_threshold - 127) / 127) # returning otsu threshold back to -1 to 1 range

    return otsu_threshold_float

raster = "data/test/20160909_merged.tif"
ndwi = calculate_ndwi(raster, plot=True)
ndwi_class = ndwi_classify(ndwi, plot=True)

# raster = "data/Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/20160908_212941_0e0f_1B_AnalyticMS.tif"

# xml = "data/Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/20160908_212941_0e0f_1B_AnalyticMS_metadata.xml"

# ref_raster = radiance_to_toa(raster, xml, plot=True)

# ndwi_raster = calculate_ndwi(ref_raster, plot=True)

# classified_raster = ndwi_classify(ndwi_raster, plot=True)

# get_otsu_threshold("/home/kjcarroll/git/CoastlineExtraction/data/output/2016/October/20161014_213436_AnalyticMS_SR_NDWI.tif")