# import libraries
import rasterio as rio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.io import MemoryFile

import shapely
from shapely.geometry import Polygon, shape, box
import geopandas as gpd

import sys
import os

from matplotlib import pyplot as plt
import numpy as np
import cv2

# Add import for check_crs
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.check_crs import check_crs, crs_match

# Add import for spatial analysis
from utils.spatial_analysis import log_spatial_info

# Add import for config loading
from load_config import load_config, get_image_path, get_shapefile_path, get_ground_truth_path, get_georeference_files, get_create_mask_output_folder

# Add import for shapefile generation
from utils.shapefile_generator import coastline_shp_from_raster, save_and_process, save_concatenated_ndwi_with_shapefile



# Gaussian blur parameters
KSIZE_BLUR = (9, 9)  # Kernel size (must be positive and odd)
SIGMA_X = 6      # Standard deviation in X direction
SIGMA_Y = 6      # Standard deviation in Y direction

MAJORITY_THRESHOLD = 0.05  # 5% of windows need to classify as water (lower threshold for better detection)


def get_ndwi_label(image_path, points_path, ksize=100, blurring=True, out_dir="result_ndwi_labels"):
    """
    This function performs NDWI calculation and classification with optional Gaussian blurring.

    Steps:
    1. Read green and NIR bands to calculate NDWI.
    2. Apply Gaussian blurring if required.
    3. Match points CRS with raster.
    4. Create a buffer around each point (number of buffers depends on the number of points in the shapefile).
    5. Mask the NDWI image with the specified buffer (sliding window).
        5.1 out_image: same shape as the original image with the specified buffer.
        5.2 out_image_clipped: clipped to the buffer.
            If the entire buffer is inside the NDWI image, its shape should be (201, 201).
            If all or part of the sliding window is outside the NDWI image, the shape will be smaller.
    6. Skip sliding windows that are not entirely inside the NDWI image.
    7. Calculate threshold based on the clipped image.
    8. Apply OR operations between `out_image` to form the final labeled image.
    9. Note that the NDWI threshold value may be less than 128.
       Therefore, it is crucial to substitute no data with -1.
    10. Apply majority rule on the number of windows to segment pixels as water.
    11. Concatenate the remaining sliding window images (unlabeled parts) from NDWI classified.
    """




    # Check CRS of input files
    check_crs(image_path, verbose=True)
    check_crs(points_path, verbose=True)



    
    # Establish the NDWI calculation and copy metadata
    with rio.open(image_path, driver='GTiff') as src_raster:
        green = src_raster.read(2).astype(np.float32)  # Get the green band
        nir_num = src_raster.count  # Adjusting NIR band to 4 or 5 band images
        nir = src_raster.read(nir_num).astype(np.float32)  # Get NIR band
        
        np.seterr(divide='ignore', invalid='ignore')
        
        ndwi = (green - nir) / (green + nir + 1e-8)  # NDWI equation
        # ndwi[np.isnan(ndwi)] = 0  # Sets any NaN values in the NDWI array to 0. (Dividing by zero => NaN pixels)
        ndwi_profile = src_raster.profile  # Copies the image profile (metadata).
        
        # Apply Gaussian blur
        if blurring:
            print("Gaussian Filtering Applied")
            ndwi = cv2.GaussianBlur(ndwi, KSIZE_BLUR, SIGMA_X, SIGMA_Y)

        # Blank label layer
        label = np.zeros((src_raster.height, src_raster.width)).astype(np.uint8)
        # Buffer matrix: each element represents the number of times a sliding window moves over a specific pixel.
        buffer_numbers = np.zeros((src_raster.height, src_raster.width)).astype(np.uint8)
        # Water count matrix: each pixel value represents the number of times pixels are labelled as water. 
        water_count = np.zeros((src_raster.height, src_raster.width)).astype(np.uint8)
        src_CRS = src_raster.crs
        
        # Getting pixel size for correct calculation of buffer.
        # This value expresses spatial resolution.
        pixel_size = abs(src_raster.transform[0])


        
        raster_bounds = src_raster.bounds # Fix: get bounds inside the 'with' block
        
    # Preparing points for creating label masks
    points_shp = gpd.read_file(points_path)
    points_geom = points_shp.geometry
    
    # Check if CRS needs to be converted
    if points_shp.crs != src_CRS:
        print(f"Converting points CRS from {points_shp.crs} to {src_CRS}")
        points_geom = points_geom.to_crs(src_CRS)  # Convert CRS to match the raster
    else:
        print(f"Points CRS already matches raster CRS: {src_CRS}")

    # Save the reprojected points to a new file for CRS checking
    os.makedirs(out_dir, exist_ok=True)
    reprojected_points_path = os.path.join(out_dir, "reprojected_points.shp")
    gpd.GeoDataFrame(geometry=points_geom).to_file(reprojected_points_path)

    # Use crs_match to check CRS of the raster and the reprojected vector file
    # IMPORTANT: Pass the path to the reprojected file, NOT the original file, to crs_match.
    if not crs_match(image_path, reprojected_points_path):
        raise ValueError("CRS mismatch after conversion! Check your input files and CRS conversion steps.")

    # Log spatial info once, outside the main loop
    log_spatial_info(raster_bounds, points_geom)

    otsu_thresholds_clipped = []  # Creating a holder for Otsu's threshold values for clipped images
    skipped = 0  # Counter for skipped windows (less than 201*201)
    
    # Processing each point found
    for multipoint in points_geom:
        for point in multipoint.geoms:
            # Create a buffer around the point
            buffer = point.buffer(ksize * pixel_size, cap_style=3)
            buffer_series = gpd.GeoSeries(buffer)

            # Writing NDWI to an in-memory dataset to use for masking
            ndwi_profile.update(count=1, nodata=0, dtype=rio.float32)
            with MemoryFile() as memfile:
                with memfile.open(**ndwi_profile) as mem_data:
                    mem_data.write_band(1, ndwi)
                with memfile.open() as dataset:
                    
                    # Added .intersection() to check if the buffer intersects with the raster's bounds before masking
                    # Get raster bounds as a shapely geometry
                    raster_bounds_geom = box(*dataset.bounds)
                    
                    # Check if the point's buffer intersects with the raster's bounds before masking
                    if buffer.intersects(raster_bounds_geom):
                        out_image, out_transform = mask(dataset, shapes=[buffer], nodata=-1, crop=False)
                        out_image = out_image[0]
                        out_image = ((out_image + 1) * 127.5).astype(np.uint8)
                        out_image = out_image.astype(np.uint8)
                        
                        out_image_clipped, out_transform_clipped = mask(dataset, shapes=[buffer], nodata=-1, crop=True)
                        out_image_clipped = out_image_clipped[0]
                        out_image_clipped = (out_image_clipped * 127) + 128
                        out_image_clipped = out_image_clipped.astype(np.uint8)
                        
                        # Create mask array: 1 for pixels within the sliding window, 0 elsewhere
                        mask_array = np.zeros_like(out_image, dtype=np.uint8)
                        mask_array[out_image != -1] = 1  # Set to 1 where we have valid data (not nodata)

                        # Skip buffering windows that are partly or wholly out of the NDWI image
                        if out_image_clipped.shape[0] < 200 or out_image_clipped.shape[1] < 200:
                            skipped += 1
                            continue

                        else:
                            # Calculate Otsu's threshold based on the clipped image
                            threshold_clipped, image_result_clipped = cv2.threshold(out_image_clipped, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            
                            # Debug: Print threshold values for first few windows
                            if len(otsu_thresholds_clipped) < 5:
                                print(f"  Window {len(otsu_thresholds_clipped)+1}: threshold={threshold_clipped}")
                            
                            # Only use valid thresholds (not exactly 0.0 or 1.0 which indicate failure)
                            if threshold_clipped != 0.0 and threshold_clipped != 1.0:
                                otsu_thresholds_clipped.append(threshold_clipped)
                                threshold_window = np.where(out_image >= threshold_clipped, 1, 0).astype(np.uint8)
                            else:
                                # Skip this window if threshold is invalid
                                skipped += 1
                                continue
                            
                        
                            label = label | threshold_window.astype(np.uint8)  # Labelled image with sliding windows 
                            
                            water_count = water_count + threshold_window
                            buffer_numbers = buffer_numbers + mask_array

                    else:
                        # If the buffer does not intersect, skip to the next point
                        skipped += 1
                        continue

    # Labelled images based on majority sliding windows
    label_majority = np.where(water_count > (buffer_numbers * MAJORITY_THRESHOLD), 1, 0)
    
    # Labelled image based on mean threshold (one threshold)
    if otsu_thresholds_clipped:
        mean_threshold = np.mean(otsu_thresholds_clipped) + 10
    else:
        mean_threshold = 128  # Default threshold if no valid thresholds
    
    ndwi_8bit = ((ndwi * 127) + 128).astype(np.uint8)
    ndwi_classified = np.where(ndwi_8bit >= mean_threshold, 1, 0)
    
    print(f"Mean threshold used: {mean_threshold}")
    print(f"NDWI 8-bit range: {ndwi_8bit.min()} to {ndwi_8bit.max()}")
    print(f"Water pixels in NDWI classified: {np.sum(ndwi_classified)}")
    
    # Start with global NDWI classification as the base
    ndwi_concatenated = ndwi_classified.copy()
    
    # For areas with sliding windows, use the sliding window result where it detects water
    # This preserves the global classification but enhances it with local precision
    sliding_windows = np.where(buffer_numbers > 0, 1, 0)
    water_areas = np.where(label == 1, 1, 0)  # Areas where sliding window detected water
    ndwi_concatenated = np.where((sliding_windows == 1) & (water_areas == 1), 1, ndwi_concatenated)

    print(f"Green min: {green.min():.2f}, Green max: {green.max():.2f}")
    print(f"NIR min: {nir.min():.2f}, NIR max: {nir.max():.2f}")
    print(f"NDWI min: {ndwi.min():.2f}, NDWI max: {ndwi.max():.2f}")  # From -1 to +1
        
    print(f"Total number of valid thresholds: {len(otsu_thresholds_clipped)}")
    print(f"Number of skipped windows: {skipped}")
    
    print(f"Actual thresholds (8-bit unsigned): {otsu_thresholds_clipped}")
    print(f"Average threshold value (8-bit unsigned): {np.mean(otsu_thresholds_clipped)}")
    print(f"Average threshold value (-1 to 1 NDWI range): {(np.mean(otsu_thresholds_clipped) - 128) / 127}")
    
    print(f"Label min: {np.nanmin(label)} , max: {np.nanmax(label)}")
    print(f"Label majority min: {np.nanmin(label_majority)} , max: {np.nanmax(label_majority)}")
    print(f"NDWI classified min: {np.nanmin(ndwi_classified)} , max: {np.nanmax(ndwi_classified)}")
    print(f"NDWI concatenated min: {np.nanmin(ndwi_concatenated)} , max: {np.nanmax(ndwi_concatenated)}")
    print(f"Buffer numbers min: {np.nanmin(buffer_numbers)} , max: {np.nanmax(buffer_numbers)}")
    print(f"Water count min: {np.nanmin(water_count)} , max: {np.nanmax(water_count)}")
    
    # Additional debugging for water detection
    print(f"Total water pixels in label: {np.sum(label)}")
    print(f"Total water pixels in label_majority: {np.sum(label_majority)}")
    print(f"Total water pixels in ndwi_classified: {np.sum(ndwi_classified)}")
    print(f"Total water pixels in ndwi_concatenated: {np.sum(ndwi_concatenated)}")
    
    # Check if we have any valid water detection
    if np.sum(ndwi_concatenated) == 0:
        print("WARNING: No water pixels detected! This might indicate:")
        print("1. NDWI threshold is too high")
        print("2. No valid transect points in the image area")
        print("3. Image quality issues")
    
    # Debug majority calculation
    if np.nanmax(buffer_numbers) > 0:
        max_water_ratio = np.nanmax(water_count) / np.nanmax(buffer_numbers)
        print(f"Maximum water ratio: {max_water_ratio:.3f} (water_count/buffer_numbers)")
        print(f"Majority threshold: {MAJORITY_THRESHOLD}")
        print(f"Pixels that would pass majority: {np.sum(water_count > (buffer_numbers * MAJORITY_THRESHOLD))}")
    
    # Save concatenated NDWI as TIFF and generate shapefile
    try:
        save_concatenated_ndwi_with_shapefile(ndwi_concatenated, ndwi_profile, image_path, out_dir)
    except Exception as e:
        print(f"Warning: Could not generate shapefile: {e}")
        # Still save the TIFF even if shapefile generation fails
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        tiff_path = os.path.join(out_dir, f"{base_name}_concatenated_ndwi.tif")
        with rio.open(tiff_path, 'w', **ndwi_profile) as dst:
            dst.write(ndwi_concatenated.astype(rio.uint8), 1)
    
    save_ndwi_plots(ndwi, ndwi_classified, label, label_majority, ndwi_concatenated, out_dir)



def save_ndwi_plots(ndwi, ndwi_classified, label, label_majority, ndwi_concatenated, out_dir="result_ndwi_labels"):
    """
    Saves and visualizes the results of NDWI classification and labeling.

    This function generates and saves plots for different NDWI-based classification outputs:
      - The raw NDWI image.
      - The NDWI image classified using a mean threshold.
      - The NDWI image labeled using a sliding window approach.
      - The NDWI image labeled using a majority rule on sliding windows.
      - A concatenated image combining sliding window and mean threshold results.
    All plots are saved as PNG files in the specified output directory.

    Args:
        ndwi: The computed NDWI image (2D numpy array).
        ndwi_classified: Binary NDWI image classified by mean threshold (2D numpy array).
        label: Binary NDWI image labeled by sliding window (2D numpy array).
        label_majority: Binary NDWI image labeled by majority rule (2D numpy array).
        ndwi_concatenated: Combined NDWI classification result (2D numpy array).
        out_dir: Directory to save the output plots (default: 'result_ndwi_labels').
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.imshow(ndwi)
    plt.title("NDWI Image")
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "ndwi_image.png"))
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(18, 14))

    axs[0, 0].imshow(ndwi_classified)
    axs[0, 0].set_title('NDWI Classified with mean threshold')
    axs[0, 0].axis('off')

    # Plot labelled image (based on sliding windows)
    axs[0, 1].imshow(label)
    axs[0, 1].set_title('NDWI Classified with sliding window')
    axs[0, 1].axis('off')

    # Plot labelled image (based on majority sliding windows)
    axs[1, 0].imshow(label_majority)
    axs[1, 0].set_title('NDWI Classified with majority sliding window')
    axs[1, 0].axis('off')
    
    # Plot NDWI classified concatenated between majority sliding windows with one mean threshold
    axs[1, 1].imshow(ndwi_concatenated)
    axs[1, 1].set_title('NDWI Concatenated')
    axs[1, 1].axis('off')

    # Save NDWI concatenated as a separate PNG
    plt.figure(figsize=(10, 8))
    plt.imshow(ndwi_concatenated)
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "ndwi_concatenated.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ndwi_outputs_summary.png"))
    plt.close()




boundary = {'type': 'Polygon',
            'coordinates': [[[-162.8235626220703, 66.05622435812153], 
                             [-162.674560546875, 66.05622435812153], 
                             [-162.674560546875, 66.10883816429516],
                             [-162.8235626220703, 66.10883816429516], 
                             [-162.8235626220703, 66.05622435812153]]]}


#  To Run script , you need only to change image and points path to yours.
config = load_config()

# Get the first 5 files from results_georeference
image_paths = get_georeference_files(config, 5)
points_path = get_ground_truth_path(config, 0)  # Deering_transect_points_2016_fw_UTM3N.shp

# Get output folder from config
output_folder = get_create_mask_output_folder(config)

print(f"Processing {len(image_paths)} images from results_georeference...")
print(f"Points path: {points_path}")
print(f"Output folder: {output_folder}")

# Process each image
for i, image_path in enumerate(image_paths):
    print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
    try:
        # Create output directory for this specific image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_folder, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Process the image
        get_ndwi_label(image_path, points_path, out_dir=image_output_dir)
        print(f"Successfully processed: {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {str(e)}")

print(f"\nAll processing complete. Results saved to: {output_folder}")
