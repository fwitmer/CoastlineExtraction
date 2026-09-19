"""
Georeference Script

This script aligns a set of satellite images to a reference base image using spatial correction.
It is useful when images are slightly misaligned after reprojection.

Main Features:

- Uses the AROSICS library for fine-tuning image alignment (coregistration)
- Works with already aligned images and applies small adjustments if needed
- Automatically saves the output to a results folder

Inputs:
- Base Image: 9-5-2016_Ortho_4Band_NDWI_3.125m.tif (well-aligned reference image)
- Target Images: Satellite images to be aligned (from aligned_data/)

Outputs:
- Georeferenced images saved in results_georeference/ folder
"""


# Input :
# Base Image :2016_HiRes_Final_Coastline_UTM3N.tif (from config)
# Target Image for Geo reference  : aligned_data
# Output : results_georeference 


import os
import shutil
import glob
import argparse
import rasterio
import sys

# Add parent directory to path to import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_config import load_config, get_ground_truth_path, get_aligned_data_folder, get_georeference_output_folder


def check_image_extent(image_path):
    """
    Check basic info of a raster image like bounds, size, and CRS.

    This helps to understand the image's spatial coverage and if it's usable 
    for alignment.

    Args:
        image_path (str): Path to the image file.

    Returns:
        BoundingBox: A tuple showing (left, bottom, right, top) coordinates.
    """

    try:
        with rasterio.open(image_path) as src:
            bounds = src.bounds
            print(f"Image: {os.path.basename(image_path)}")
            print(f"  Bounds: {bounds}")
            print(f"  Width: {src.width}, Height: {src.height}")
            print(f"  CRS: {src.crs}")
            return bounds
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def georeference(base_image, target_image, outfile=None, force=False, output_dir=None):
    """
    Georeference a target image by aligning it to a base image using AROSICS.

    This function checks for small misalignments and fixes them. If no issues
    are found, it simply copies the original image to the output folder. You
    can also choose to force a retry with more aggressive settings.

    Args:
        base_image (str): Path to the reference image.
        target_image (str): Path to the image that needs alignment.
        outfile (str, optional): Custom name for the output file.
        force (bool, default=False): If True, retries with stronger correction if first attempt fails.
        output_dir (str, optional): Directory to save the result. Defaults to results_georeference/.

    Returns:
        str: Path to the output image file.
    """

    
    # Check if files exist
    if not os.path.isfile(base_image):
        raise FileNotFoundError(f"Base image not found: {base_image}")
    if not os.path.isfile(target_image):
        raise FileNotFoundError(f"Target image not found: {target_image}")

    # Use provided output directory or create results_georeference directory in the main project directory
    if output_dir:
        results_dir = output_dir
    else:
        project_dir = os.path.dirname(base_image)  # Get the main project directory
        results_dir = os.path.join(project_dir, 'results_georeference')
    
    os.makedirs(results_dir, exist_ok=True)

    target_filename = os.path.basename(target_image)

    # Specify correct output filepath
    if outfile:
        path_out = os.path.join(results_dir, outfile)
    else:
        out_name = target_filename.split(sep=".")[0] + "_GeoRegistered.tif"
        path_out = os.path.join(results_dir, out_name)

    # For pre-aligned images, use more sensitive parameters
    try:
        print(f"Attempting fine coregistration for {target_filename}...")
        
        # Use very sensitive parameters for pre-aligned images
        CR = COREG(base_image, target_image, path_out=path_out, 
                   max_shift=2,  # Very small max shift for fine-tuning
                   window_size=(128, 128),  # Smaller windows for more precise detection
                   grid_res=50,  # Fine grid resolution
                   min_corr=0.3,  # Lower correlation threshold
                   fmt_out='GTiff')  # Ensure GTiff output
        
        print(f"Calculating spatial shifts for {target_filename}...")
        CR.calculate_spatial_shifts()
        
        # Check if any significant shifts were detected
        shifts = CR.shifts
        if shifts and any(abs(shift) > 0.1 for shift in shifts.values()):
            print(f"Significant shifts detected: {shifts}")
            print(f"Correcting shifts for {target_filename}...")
            CR.correct_shifts()
            print(f'Successfully saved fine-tuned image as {path_out}')
        else:
            print(f"No significant shifts detected for {target_filename}. Image already well-aligned.")
            # Copy the original aligned image to results if no correction needed
            shutil.copy2(target_image, path_out)
            print(f'Copied original aligned image to {path_out}')
        
        return path_out
        
    except Exception as e:
        print(f"Error during fine coregistration {target_filename}: {str(e)}")
        
        if not force:
            print(f"Since images are pre-aligned, copying original to results: {path_out}")
            shutil.copy2(target_image, path_out)
            return path_out
        else:
            # Try with even more aggressive parameters if force=True
            try:
                print(f"Retrying with aggressive parameters for {target_filename}...")
                CR = COREG(base_image, target_image, path_out=path_out,
                          max_shift=5,  # Increased max shift
                          window_size=(64, 64),  # Very small windows
                          grid_res=25,  # Very fine grid
                          min_corr=0.1)  # Very low correlation threshold
                
                CR.calculate_spatial_shifts()
                CR.correct_shifts()
                
                print(f'Successfully saved Georegistered image as {path_out}')
                return path_out
                
            except Exception as e2:
                print(f"All attempts failed for {target_filename}: {str(e2)}")
                raise e2





def main():
    config = load_config()
    default_base_img = get_ground_truth_path(config, 2)
    default_aligned_dir = get_aligned_data_folder(config)
    default_output_dir = get_georeference_output_folder(config)

    parser = argparse.ArgumentParser(description="Fine coregistration of pre-aligned satellite images using AROSICS.")
    parser.add_argument("--base-image", dest="base_image", type=str, default=default_base_img,
                        help=f"Path to reference base image (default: {default_base_img})")
    parser.add_argument("--aligned-dir", "--input-dir", dest="aligned_dir", type=str, default=default_aligned_dir,
                        help=f"Path to input aligned images directory (default: {default_aligned_dir})")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=default_output_dir,
                        help=f"Path to output directory for georeferenced images (default: {default_output_dir})")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force full coregistration warping")

    args = parser.parse_args()

    print("=== Georeferencing Workflow ===")
    print("Note: This script performs fine coregistration on pre-aligned images.")
    print(f"Base image: {args.base_image}")
    print(f"Aligned data directory: {args.aligned_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Check if base image exists
    if not os.path.exists(args.base_image):
        print(f"Error: Base image not found: {args.base_image}")
        print("Please ensure the ground truth files are properly configured.")
        sys.exit(1)
    
    # Check if aligned data directory exists
    if not os.path.exists(args.aligned_dir):
        print(f"Error: Aligned data directory not found: {args.aligned_dir}")
        print("Please run batch_align.py first to create aligned images.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created/verified: {args.output_dir}")
    
    # Check base image extent
    print("=== Base Image Information ===")
    check_image_extent(args.base_image)
    
    # Get all .tif files in the aligned data folder
    target_images = glob.glob(os.path.join(args.aligned_dir, "*.tif"))
    print(f"\nFound {len(target_images)} pre-aligned target images to process")
    
    if len(target_images) == 0:
        print("No target images found in aligned data directory.")
        print("Please ensure batch_align.py has been run successfully.")
        sys.exit(1)
    
    # Check first few target images for debugging
    print("\n=== Sample Target Images Information ===")
    for target_img in target_images[:3]:
        check_image_extent(target_img)
        print()
    
    print("=== Starting Fine Coregistration Process ===")
    successful = 0
    failed = 0
    
    for target_img in target_images:
        print(f"\nProcessing: {os.path.basename(target_img)}")
        try:
            output = georeference(args.base_image, target_img, force=args.force, output_dir=args.output_dir)
            print("Processing complete. Output:", output)
            successful += 1
        except Exception as e:
            print("Error during processing:", e)
            failed += 1
            continue
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total images: {len(target_images)}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()




# Note : reprojected the base image using the command-       
# base_img = r"C:\Users\91730\Desktop\Plante_Sentinal_Fusion_Exp\2016_HiRes_Final_Coastline_UTM3N.tif"     

# gdalinfo 2016_HiRes_Final_Coastline_UTM3N.tif
# gdalinfo "C:\Users\91730\Desktop\Plante_Sentinal_Fusion_Exp\sample_data\369619_2018-06-15_RE1_3A_Analytic_SR_clip.tif"