import warnings
import os
import rasterio
import cv2
from rasterio.plot import show_hist
from rasterio.fill import fillnodata
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from xml.dom import minidom
from skimage.filters import threshold_yen
from skimage import feature
from arosics import COREG
import time
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  active_contour,
                                  checkerboard_level_set)
from scipy.interpolate import make_interp_spline, BSpline, splprep, splev
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

        nir_num = src.count  # adjusting NIR band to 4 or 5 band images

        green_band = src.read(2).astype(rasterio.float32)  # band 2 - green
        nir_band = src.read(nir_num).astype(rasterio.float32)    # band 4 - NIR
        print("DONE\n")

    print("Calculating NDWI:", end=" ")
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (green_band - nir_band) / (green_band + nir_band)
    print("DONE\n")
    ndwi = cv2.GaussianBlur(ndwi, (17, 17), 0)
    if outfile:
        out_filename = outfile
    else:
        out_filename = raster_filepath + raster_filename.split(sep=".")[0] + "_NDWI.tif"
    print("Saving calculated NDWI image as", out_filename, ":", end=" ")
    with rasterio.open(out_filename, 'w', **kwargs) as dst:
        dst.nodata = 0
        dst.write_band(1, ndwi.astype(rasterio.float32))
        print("DONE\n")

    if plot:
        bands = [ndwi]
        labels = ["NDWI (Normalized Difference Water Index)"]
        plot_raster(bands, labels)

        show_hist(ndwi, bins=100, stacked=False, alpha=0.3, histtype='stepfilled', title="NDWI Values")
    return out_filename


def ndwi_classify(rasterfile, outfile=None, plot=False):
    raster_filepath = os.path.dirname(rasterfile) + "/"
    raster_filename = os.path.basename(rasterfile)

    img = raster_filepath + raster_filename
    print("Opening", raster_filename, "for NDWI water classification:", end=" ")
    with rasterio.open(img, driver="GTiff") as src:
        kwargs = src.meta
        kwargs.update(dtype=rasterio.uint8, count=1)

        ndwi = src.read(1).astype(rasterio.uint8)
        print("DONE\n")
    if plot:
        plt.imshow(ndwi, cmap='gray')
        plt.show()
    k_means = get_k_means(ndwi, plot=plot)
    area_of_uncertainty = 0
    lowest_count = (k_means == 0).sum()
    print("K-means Summary:")
    for i in [0, 1, 2]:
        ratio = float((k_means == i).sum() / (k_means.shape[0] * k_means.shape[1]))
        if (k_means == i).sum() < lowest_count:
            area_of_uncertainty = i
            lowest_count = (k_means == i).sum()
        print("\tClass {}: {}%".format(i, ratio * 100))
    # k_out = raster_filepath + raster_filename.split(sep=".")[0] + "_KMeans.tif"
    # with rasterio.open(k_out, 'w', **kwargs) as dst:
    #     dst.nodata = 255
    #     dst.write_band(1, k_means.astype(rasterio.uint8))
    ndwi_classified = np.zeros(ndwi.shape).astype(np.bool)
    print("NDWI Classified Shape:", ndwi_classified.shape)
    print("K-Means Shape:", k_means.shape)
    first_window = True
    water_val = 0
    for (x, y, window) in sliding_window(k_means, 50, (100, 100)):
        if first_window:
            first_window = False
            largest_count = (window == 0).sum()
            for num in [0,1,2]:
                count = (window == num).sum()
                if count > largest_count:
                    largest_count = count
                    water_val = num
        water_ratio = (window == water_val).sum() / (window.shape[0] * window.shape[1])
        if (window == area_of_uncertainty).sum() and (window == water_val).sum():
            cropped_ndwi = ndwi[y:y + window.shape[0], x:x + window.shape[1]]
            otsu_threshold, image_result = cv2.threshold(cropped_ndwi, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )
            classified_window = np.where(cropped_ndwi >= otsu_threshold,
                                         1,
                                         0)
            ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] = \
                (ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] | classified_window)
        elif water_ratio > 0.9:
            ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] = \
                (ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] | np.ones(window.shape).astype(
                    np.bool))

        # if water_ratio == 0:
        #     ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] = \
        #         (ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] | np.ones(window.shape).astype(
        #             np.bool))
        # if water_ratio > 0.9:
        #     ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] = \
        #         (ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] | np.zeros(window.shape).astype(
        #             np.bool))
        # elif water_ratio < 0.005:
        #     ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] = \
        #         (ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] | np.ones(window.shape).astype(
        #             np.bool))
        # else:
        #     cropped_ndwi = ndwi[y:y + window.shape[0], x:x + window.shape[1]]
        #     otsu_threshold, image_result = cv2.threshold(cropped_ndwi, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )
        #     classified_window = np.where(cropped_ndwi >= otsu_threshold,
        #                                  0,
        #                                  1)
        #     ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] = \
        #         (ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] | classified_window)
        # water_ratio = float((window == lowest_num).sum()) / (window.shape[0] * window.shape[1])
        # if water_ratio > 0.95:
        #     if water_ratio >= 0.995:
        #         ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] = \
        #             (ndwi_classified[y:y + window.shape[0], x:x + window.shape[1]] | np.ones(window.shape).astype(
        #                 np.bool))
        #     continue
        # if water_ratio < 0.05:
        #     ndwi_classified[y:y+window.shape[0], x:x + window.shape[1]] = \
        #         (ndwi_classified[y:y+window.shape[0], x:x + window.shape[1]] | np.zeros(window.shape).astype(np.bool))
        #     continue
        # cropped_ndwi = ndwi[y:y + window.shape[0], x:x + window.shape[1]]
        # otsu_threshold, image_result = cv2.threshold(cropped_ndwi, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )
        # classified_window = np.where(cropped_ndwi >= otsu_threshold,
        #                              1,
        #                              0)
        # ndwi_classified[y:y+window.shape[0], x:x + window.shape[1]] = \
        #     (ndwi_classified[y:y+window.shape[0], x:x + window.shape[1]] | classified_window)
    if plot:
        plt.imshow(ndwi_classified, cmap='gray')
        plt.show()
    print("DONE\n")
    transformed_classification = morph_transform(ndwi_classified.astype(np.uint8), 15, 15)
    if plot:
        plt.imshow(transformed_classification, cmap='gray')
        plt.show()
    if outfile:
        out_filename = outfile
    else:
        out_filename = raster_filepath + raster_filename.split(sep=".")[0] + "_classified.tif"
    print("Saving classified raster as", out_filename, ":", end=" ")
    with rasterio.open(out_filename, 'w', **kwargs) as dst:
        dst.nodata = 255
        dst.write_band(1, transformed_classification.astype(rasterio.uint8))
    print("DONE\n")

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
    ndwi_8_bit = np.floor((ndwi + 1) * 128)
    ndwi_8_bit = np.where(ndwi_8_bit > 255, 255, ndwi_8_bit)
    out_filename = path.split(sep=".")[0] + "_8bit.tif"
    with rasterio.open(out_filename, mode='w', **kwargs) as dst:
        dst.write_band(1, ndwi_8_bit.astype(rasterio.uint8))

    image = cv2.imread(out_filename, 0) # 0 is grayscale mode
    if reduce_noise:
        image = cv2.GaussianBlur(image, (5,5), 0)

    otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    otsu_threshold_float = float((otsu_threshold / 128.0) - 1) # returning otsu threshold back to -1 to 1 range

    return otsu_threshold_float


def get_yen_threshold(path):
    with rasterio.open(path, driver="GTiff") as src:
        image = src.read(1)
    threshold = threshold_yen(image)
    return (threshold - 127) / 128


def get_edges(img):
    src = cv2.imread(img, 0)
    plt.imshow(src, cmap='gray')
    plt.show()

    src_blur = cv2.GaussianBlur(src, (15,15), 0)
    canny = cv2.Canny(src_blur, 30, 80, L2gradient=None)
    plt.imshow(canny)
    plt.show()

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    drawing = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
    cv2.drawContours(drawing, contours, -1, 1, 3)
    plt.imshow(drawing, cmap='gray')
    plt.show()


def get_contours(img, outfile=None, plot=False):
    raster_filepath = os.path.dirname(img) + "/"
    raster_filename = os.path.basename(img)
    with rasterio.open(img, driver="GTiff") as src:
        input = src.read(1).astype(rasterio.uint8)
    # src = cv2.imread(img, 0)
    if plot:
        plt.imshow(input, cmap='gray')
        plt.show()
    # src_blur = cv2.GaussianBlur(src, (17,17), 0)
    contours, hierarchy = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    drawing = np.zeros((input.shape[0], input.shape[1]), dtype=np.uint8)
    cv2.drawContours(drawing, contours, -1, 1, 3)
    if plot:
        plt.imshow(drawing, cmap='gray')
        plt.show()

    # preparing to write output coastline
    if outfile:
        out_filename = outfile
    else:
        out_filename = raster_filepath + raster_filename.split(sep=".")[0] + "_coastline.tif"
    with rasterio.open(img, driver="GTiff") as src:
        kwargs = src.meta
        kwargs.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(out_filename, 'w', **kwargs) as dst:
        dst.nodata = 255
        dst.write_band(1, drawing.astype(rasterio.uint8))


def get_k_means(img, num_means=3, plot=False):
    try:
        src = cv2.imread(img, cv2.IMREAD_ANYDEPTH)
        if plot:
            plt.imshow(src, cmap='gray')
            plt.show()
        # converting to 2D array of pixel values per
        # https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/
        pix_vals = src.reshape((-1, 1))
        pix_vals = np.float32(pix_vals)
        image_shape = src.shape
    except:
        pix_vals = img.reshape((-1, 1))
        pix_vals = np.float32(pix_vals)
        image_shape = img.shape

    retval, labels, centers = cv2.kmeans(pix_vals, num_means, None,
                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1.0),
                                         attempts=30,
                                         flags=cv2.KMEANS_RANDOM_CENTERS)
    reshaped_labels = labels.reshape(image_shape)
    if plot:
        plt.imshow(reshaped_labels, cmap='gray')
        plt.show()

    return reshaped_labels.astype(np.uint8)


def sliding_window(image, step, window_size):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield(x, y, image[y:y + window_size[0], x:x + window_size[1]])


# Function to Geo-Reference target_image based on base_image (It is recommended
# To use the HiRes September 2016 Image as base_image
def georeference(base_image, target_image, outfile=None):

    # Get image paths
    base_filepath = os.path.dirname(base_image) + '/'
    base_filename = os.path.basename(base_image)

    target_filepath = os.path.dirname(target_image) + '/'
    target_filename = os.path.basename(target_image)

    im_reference = base_filepath + base_filename
    im_target = target_filepath + target_filename

    # Specify correct output filepath
    if outfile:
        path_out = outfile
    else:
        path_out = target_filename.split(sep=".")[0] + "_GeoRegistered.tif"

    # Coregister imagery
    # wp and ws Set as bounding box around Deering Airstrip
    CR = COREG(im_reference, im_target, wp=(600578.602641986, 7328849.357436092), ws=(965, 1089.7365), path_out=path_out)

    # Calculate spatial shifts
    CR.calculate_spatial_shifts()

    # Correct shifts
    CR.correct_shifts()

    print('Saving Georegistered image as', path_out, ":", end=" ")

    return target_filepath + path_out


def morph_transform(fname, kwidth, kheight, outname=None):
    '''
    Perform opening/closing as described in the Paravolidakis paper.

    IN:
        fname: raster image to transform
        kwidth: kernel width
        kheight: kernel height
        outname: name of file to write transformed image to
    OUT:
        No return value, results written to file
    '''
    try:
        dat = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
    except:
        dat = fname
    kernel = np.ones((kheight, kwidth), np.uint8)
    opened = cv2.morphologyEx(dat, cv2.MORPH_OPEN, kernel)
    opened_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    if outname:
        return cv2.imwrite(outname, opened_closed)
    else:
        return opened_closed

# This function should only be passed files with values as unsigned integers
# still looking into how to interpolate signed float values
def fill_nodata(file_to_fill, mask_file = None, plot=False):
    raster_filepath = os.path.dirname(file_to_fill) + "/"
    raster_filename = os.path.basename(file_to_fill)
    if mask_file:
        with rasterio.open(mask_file) as src:
            masks = src.read_masks()
            count = src.count
            mask = masks[0] & masks[1]

    else:
        mask = None
    if plot:
        plt.imshow(mask, cmap='gray')
        plt.show()
    with rasterio.open(file_to_fill) as src:
        nodata = src.nodata
        kwargs = src.meta
        kwargs.update(dtype=rasterio.float32, count=1)
        ndwi = src.read(1)
        if plot:
            plt.imshow(ndwi, cmap='gray')
            plt.show()
        filled = fillnodata(ndwi, (ndwi != nodata), max_search_distance=300)
    if plot:
        plt.imshow(filled, cmap='gray')
        plt.show()

    out_filename = raster_filepath + raster_filename.split(sep=".")[0] + "_filled.tif"

    with rasterio.open(out_filename, 'w', **kwargs) as dst:
        dst.nodata = 0
        dst.write_band(1, filled.astype(rasterio.float32))

def get_snake(file, plot=False):
    def store_evolution_in(lst):
        def _store(x):
            lst.append(np.copy(x))

        return _store
    evolution = []
    callback = store_evolution_in(evolution)
    raster_filepath = os.path.dirname(file) + "/"
    raster_filename = os.path.basename(file)
    with rasterio.open(file, driver='GTiff') as src:
        kwargs = src.meta
        kwargs.update(count=1, dtype=rasterio.uint8)
        input = src.read(1).astype(rasterio.uint8)
    if plot:
        plt.imshow(input, cmap='gray')
        plt.show()
    init_lvl_set = checkerboard_level_set(input.shape)
    lvl_set = morphological_chan_vese(input, 100, init_level_set=init_lvl_set,iter_callback=callback, smoothing=1)
    if plot:
        plt.imshow(lvl_set, cmap='gray')
        plt.show()
    noise_reduced = morph_transform(lvl_set.astype(rasterio.uint8), 9, 9)
    if plot:
        plt.imshow(noise_reduced)
        plt.show()
    out_filename = raster_filepath + raster_filename.split(sep=".")[0] + "_chan_vese.tif"
    with rasterio.open(out_filename, 'w', **kwargs) as dst:
        dst.write_band(1, noise_reduced.astype(rasterio.uint8))
    return lvl_set


# raster = "data/test/20161015_merged.tif"
# ndwi = calculate_ndwi(raster, plot=True)
# ndwi_class = ndwi_classify(ndwi, plot=True)

# raster = "data/9-5-2016_Ortho/9-5-2016_Ortho_4Band.tif"
# ndwi = calculate_ndwi(raster, plot=True)
# ndwi_class = ndwi_classify(ndwi, plot=True)

# raster = "data/Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/20160908_212941_0e0f_1B_AnalyticMS.tif"

# xml = "data/Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/20160908_212941_0e0f_1B_AnalyticMS_metadata.xml"

# ref_raster = radiance_to_toa(raster, xml, plot=True)

# ndwi_raster = calculate_ndwi(ref_raster, plot=True)

# classified_raster = ndwi_classify(ndwi_raster, plot=True)

# get_otsu_threshold("/home/kjcarroll/git/CoastlineExtraction/data/output/2016/October/20161014_213436_AnalyticMS_SR_NDWI.tif")
# get_edges("data/test/20161015_merged_NDWI_8bit.tif")
# get_contours("data/test/20161015_merged_NDWI_classified.tif")
#
# get_edges("data/9-5-2016_Ortho/9-5-2016_Ortho_4Band_NDWI_8bit.tif")
# get_contours("data/9-5-2016_Ortho/9-5-2016_Ortho_4Band_NDWI_classified.tif")

# get_k_means("data/test/20161015_merged_NDWI_8bit.tif")
# calculate_ndwi("data/test/20161015_merged.tif")


# fill_nodata("data/test/20160909_merged_NDWI.tif", "data/test/20160909_merged.tif", plot=True)
# get_otsu_threshold("data/test/20160909_merged_NDWI_filled.tif")
# ndwi_classify("data/test/20160909_merged_NDWI_filled_8bit.tif", plot=True)
# get_contours("data/test/20160909_merged_NDWI_filled_8bit_classified.tif", plot=True)
# fill_nodata("data/9-5-2016_Ortho/9-5-2016_Ortho_4Band_NDWI_filled.tif", "data/9-5-2016_Ortho/9-5-2016_Ortho_4Band.tif",
#             plot=True)

# ndwi_classify("data/9-5-2016_Ortho/9-5-2016_Ortho_4Band_NDWI_8bit.tif", plot=True)
# ndwi_classify("data/test/20160909_merged_NDWI_8bit.tif")
# get_contours("data/test/20161015_merged_NDWI_filled_8bit_classified.tif")
# get_snake("data/test/20161015_merged_NDWI_filled.tif", "data/test/20161015_merged_NDWI_filled_coastline.tif")
# get_snake("data/test/20161015_merged_NDWI_filled.tif", "data/test/20161015_merged_NDWI_filled_8bit_classified.tif")

# outfile = "data/test/OrthoTiles/20160904_NDWI.tif"
# calculate_ndwi("data/test/OrthoTiles/369619_2016-09-04_RE2_3A_Analytic_SR_clip_Georegistered.tif", outfile=outfile)
# get_otsu_threshold("data/test/OrthoTiles/20160904_NDWI.tif")
# fill_nodata("data/test/OrthoTiles/20160904_NDWI_8bit.tif",
#             "data/test/OrthoTiles/369619_2016-09-04_RE2_3A_Analytic_SR_clip.tif")
# ndwi_classify("data/test/OrthoTiles/20160904_NDWI_8bit_filled.tif")
# get_contours("data/test/OrthoTiles/20160904_NDWI_8bit_filled_classified.tif", plot=True)
# get_edges("data/test/OrthoTiles/20160904_NDWI_8bit_filled_classified.tif")
# get_snake("data/test/OrthoTIles/20160904_NDWI_filled.tif", "data/test/OrthoTiles/20160904_NDWI_filled_8bit_classified.tif")
# fill_nodata("data/test/OrthoTiles/20160904_NDWI_8bit_KMeans.tif", "data/test/OrthoTiles/369619_2016-09-04_RE2_3A_Analytic_SR_clip.tif", plot=True)
#
# outfile = "data/test/OrthoTiles/20160906_NDWI.tif"
# calculate_ndwi("data/test/OrthoTiles/369619_2016-09-06_RE5_3A_Analytic_SR_clip.tif", outfile=outfile)
# get_otsu_threshold("data/test/OrthoTiles/20160906_NDWI.tif")
# fill_nodata("data/test/OrthoTiles/20160906_NDWI_8bit.tif",
#             "data/test/OrthoTiles/369619_2016-09-06_RE5_3A_Analytic_SR_clip.tif")
# ndwi_classify("data/test/OrthoTiles/20160906_NDWI_8bit_filled.tif", plot=True)
# get_contours("data/test/OrthoTiles/20160906_NDWI_8bit_filled_classified.tif", plot=True)
# get_snake("data/test/OrthoTiles/20160906_NDWI_filled.tif", "data/test/OrthoTiles/20160906_NDWI_filled_8bit_classified.tif")

# calculate_ndwi("data/test/20161015_merged.tif")
# fill_nodata("data/test/20161015_merged_NDWI.tif", "data/test/20161015_merged_NDWI.tif")
# get_otsu_threshold("data/test/20161015_merged_NDWI_filled.tif")
# ndwi_classify("data/test/20161015_merged_NDWI_filled_8bit.tif", plot=True)
# get_contours("data/test/20161015_merged_NDWI_filled_8bit_coastline.tif", plot=True)

# with rasterio.open("data/test/OrthoTiles/20160906_NDWI_8bit.tif", driver='GTiff') as src:
#     ndwi = src.read(1).astype(rasterio.uint8)
#     nodata = src.nodata
#
#
# with rasterio.open("data/test/OrthoTiles/369619_2016-09-06_RE5_3A_Analytic_SR_clip.tif", driver='GTiff') as src:
#     masks = src.read_masks()
#     mask = (masks[1] & masks[4])
#     plt.imshow(mask, cmap='gray')
#     plt.show()
#     blue = src.read(1)
#     print(blue.dtype)
#     plt.imshow(blue, cmap='gray')
#     plt.show()
#     blue_filled = fillnodata(blue, mask, max_search_distance=300)
#     plt.imshow(blue_filled, cmap='gray')
#     plt.show()
#     filled_ndwi = fillnodata(ndwi, (ndwi != nodata), max_search_distance=100)
#     plt.imshow(filled_ndwi, cmap='gray')
#     plt.show()

# get_contours("data/test/DeepWaterMap/September_6_2016_REOrthoTile_Explorer_dwm.tif", plot=True)

# def store_evolution_in(lst):
#     def _store(x):
#         lst.append(np.copy(x))
#
#     return _store
#
# with rasterio.open("data/test/DeepWaterMap/September_4_2016_REOrthoTile_Explorer_dwm.tif", driver='GTiff') as src:
#     input = src.read(1).astype(rasterio.uint8)
# plt.imshow(input, cmap='gray')
# plt.show()
# init_lvl_set = checkerboard_level_set(input.shape)
# evolution = []
# callback = store_evolution_in(evolution)
# lvl_set = morphological_chan_vese(input, 100, init_level_set=init_lvl_set, smoothing=1, iter_callback=callback)
# plt.imshow(lvl_set, cmap='gray')
# plt.show()
# open_closed = morph_transform(lvl_set.astype(np.uint8), 9, 9)
# plt.imshow(open_closed, cmap='gray')
# plt.show()

# get_snake("data/test/OrthoTiles/20160904_NDWI_8bit_filled.tif", plot=True)
# get_contours("data/test/OrthoTiles/20160904_NDWI_8bit_filled_chan_vese.tif", plot=True)

# get_snake("data/test/OrthoTiles/20160906_NDWI_8bit_filled.tif", plot=True)
# get_contours("data/test/OrthoTiles/20160906_NDWI_8bit_filled_chan_vese.tif", plot=True)


# get_snake("data/test/DeepWaterMap/September_6_2016_REOrthoTile_Explorer_dwm.tif", plot=True)
# get_contours("data/test/DeepWaterMap/September_6_2016_REOrthoTile_Explorer_dwm_chan_vese.tif", plot=True)
#
# get_snake("data/test/DeepWaterMap/September_4_2016_REOrthoTile_Explorer_dwm.tif", plot=True)
# get_contours("data/test/DeepWaterMap/September_4_2016_REOrthoTile_Explorer_dwm_chan_vese.tif", plot=True)