import warnings
import os
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
import numpy as np
from xml.dom import minidom

def DN_to_TOA(rasterfile, xmlfile, plot = False, verbose = False):
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
    scale = 100000
    out_filename = raster_filename.split(sep=".")[0] + "_TOAreflectance.tif"
    print("Saving TOA reflectance as", out_filename, ":", end=" ")
    scaled_blue = blue_band_reflectance * scale
    scaled_green = green_band_reflectance * scale
    scaled_red = red_band_reflectance * scale
    scaled_nir = nir_band_reflectance * scale
    with rasterio.open(raster_filepath + out_filename, 'w', **kwargs) as dst:
        dst.write_band(1, blue_band_reflectance.astype(rasterio.float64))
        dst.write_band(2, green_band_reflectance.astype(rasterio.float64))
        dst.write_band(3, red_band_reflectance.astype(rasterio.float64))
        dst.write_band(4, nir_band_reflectance.astype(rasterio.float64))
    print("DONE")
    print()

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

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

        for refl_band, color in zip([blue_band_reflectance, green_band_reflectance, red_band_reflectance, nir_band_reflectance],
                        ["Blue", "Green", "Red", "NIR"]):
            print("Generating", color, "band plot:", end=" ")
            min_val = np.nanmin(refl_band)
            max_val = np.nanmax(refl_band)
            mid = np.nanmean(refl_band)

            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            cax = ax.imshow(refl_band, cmap='Greys', clim=(min_val, max_val),
                        norm=MidpointNormalize(midpoint=mid, vmin=min_val, vmax=max_val))

            ax.axis('off')
            ax.set_title(color + " Reflectance", fontsize=18, fontweight='bold')

            cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)

            plt.show()
            print("DONE")
        print()

def calculate_ndwi(rasterfile):
    raster_filepath = os.path.dirname(rasterfile) + "/"
    raster_filename = os.path.basename(rasterfile)

    img = raster_filepath + raster_filename

    print("Opening", raster_filename, "to read in band data:", end=" ")
    with rasterio.open(img, driver="GTiff") as src:
        kwargs = src.meta
        kwargs.update(dtype=rasterio.float64, count=1)

        green_band = src.read(2)  # band 2 - green
        nir_band = src.read(4)    # band 4 - NIR
        print("DONE")

    print("Calculating NDWI:", end=" ")
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.where(
        (green_band + nir_band) == 0.,
        0,
        (green_band - nir_band) / (green_band + nir_band))
    print("DONE")
    out_filename = raster_filename.split(sep=".")[0] + "_NDWI.tif"
    print("Saving TOA reflectance as", out_filename, ":", end=" ")
    with rasterio.open(raster_filepath + out_filename, 'w', **kwargs) as dst:
        dst.write_band(1, ndwi.astype(float))
        print("DONE")

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

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

    for refl_band, color in zip(
            [ndwi], ["NDWI"]):
        print("Generating", color, "band plot:", end=" ")
        min_val = np.nanmin(refl_band)
        max_val = np.nanmax(refl_band)
        mid = np.nanmean(refl_band)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        cax = ax.imshow(refl_band, cmap='Greys', clim=(min_val, max_val),
                        norm=MidpointNormalize(midpoint=mid, vmin=min_val, vmax=max_val))

        ax.axis('off')
        ax.set_title(color + " Reflectance", fontsize=18, fontweight='bold')

        cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.65)

        plt.show()
        print("DONE")
    print()


raster = "Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/20160908_212941_0e0f_1B_AnalyticMS.tif"

xml = "Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/20160908_212941_0e0f_1B_AnalyticMS_metadata.xml"

DN_to_TOA(raster, xml)
# DN_to_TOA(raster, xml, plot=True, verbose=True)
ref_raster = "Unortho Deering Images With RPCs 1-30/files/PSScene4Band/20160908_212941_0e0f/basic_analytic/20160908_212941_0e0f_1B_AnalyticMS_TOAreflectance.tif"
calculate_ndwi(ref_raster)
