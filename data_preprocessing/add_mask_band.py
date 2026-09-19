import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import argparse
from pathlib import Path

# Add import for config loading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_config import load_config, get_georeference_files, get_create_mask_output_folder, get_add_mask_band_output_folder


def add_mask_band(image_path, mask_path, output_dir):
    """
    Add a mask band to a geotiff image and save mask separately.
    
    Args:
        image_path (str): Path to the input geotiff image
        mask_path (str): Path to the mask geotiff file
        output_dir (str): Directory to save output files
    
    Returns:
        tuple: (stacked_image, mask_array, output_paths)
    """
    # Open Planet imagery
    with rasterio.open(image_path) as src:
        image = src.read()   # shape: (bands, H, W)
        profile = src.profile.copy()

    # Open mask (assumed single band, same size as image)
    with rasterio.open(mask_path) as msk:
        mask = msk.read(1)   # shape: (H, W)
        mask_profile = msk.profile.copy()

    # Verify that mask and image have the same dimensions
    if mask.shape != (image.shape[1], image.shape[2]):
        raise ValueError(f"Mask dimensions {mask.shape} don't match image dimensions {(image.shape[1], image.shape[2])}")

    # Get base name for output files
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save mask separately
    mask_output_path = os.path.join(output_dir, f"{image_basename}_mask.tif")
    mask_profile.update(count=1, dtype='uint8')
    with rasterio.open(mask_output_path, "w", **mask_profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
    
    # Stack image + mask as new band
    stacked = np.vstack([image, mask[np.newaxis, ...]])  # (bands+1, H, W)

    # Update metadata: number of bands
    profile.update(count=stacked.shape[0])

    # Save stacked image
    stacked_output_path = os.path.join(output_dir, f"{image_basename}_with_mask.tif")
    with rasterio.open(stacked_output_path, "w", **profile) as dst:
        dst.write(stacked)

    print(f"Saved mask separately → {mask_output_path}")
    print(f"Saved stacked GeoTIFF with {stacked.shape[0]} bands → {stacked_output_path}")
    
    output_paths = {
        'mask': mask_output_path,
        'stacked': stacked_output_path
    }
    
    return stacked, mask, output_paths

def find_corresponding_mask(image_path, mask_base_dir):
    """
    Find the corresponding mask file for a given image.
    
    Args:
        image_path (str): Path to the input image
        mask_base_dir (str): Base directory containing mask files
    
    Returns:
        str: Path to the corresponding mask file, or None if not found
    """
    # Get the base name of the image (without extension)
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Look for the corresponding mask file
    mask_pattern = os.path.join(mask_base_dir, image_basename, f"{image_basename}_concatenated_ndwi.tif")
    
    if os.path.exists(mask_pattern):
        return mask_pattern
    
    # Alternative search pattern
    mask_pattern_alt = os.path.join(mask_base_dir, f"{image_basename}_concatenated_ndwi.tif")
    if os.path.exists(mask_pattern_alt):
        return mask_pattern_alt
    
    return None



def visualize_bands(image_with_mask, basename, output_dir):
    """
    Visualize R, G, B, NIR, and Mask bands properly.
    
    Args:
        image_with_mask (numpy.ndarray): Stacked image with mask band
        basename (str): Base name for saving the visualization
        output_dir (str): Directory to save the visualization
    """
    # Create figure with 5 subplots (R, G, B, NIR, Mask)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Plot individual bands (all in grayscale since they are single bands)
    band_titles = ["Red Band", "Green Band", "Blue Band", "NIR Band", "Mask"]
    
    for i in range(5):
        ax = axes[i]
        band = image_with_mask[i]
        
        if i < 4:  # Image bands (R, G, B, NIR)
            # Normalize the band for better visualization
            norm_band = (band - band.min()) / (band.max() - band.min())
            ax.imshow(norm_band, cmap="gray")
        else:  # Mask band
            ax.imshow(band, cmap="gray")
        
        ax.set_title(band_titles[i], fontsize=12, fontweight='bold')
        ax.axis("off")
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, f"{basename}_bands_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved band visualization → {viz_path}")

def create_individual_visualizations(processed_data, output_dir):
    """
    Create individual visualizations for each processed image.
    
    Args:
        processed_data (list): List of processed image data
        output_dir (str): Directory to save visualizations
    """
    print(f"\nCreating visualizations for {len(processed_data)} images...")
    
    for data in processed_data:
        print(f"  Creating visualization for: {data['basename']}")
        visualize_bands(data['stacked'], data['basename'], output_dir)


def main():
    config = load_config()
    default_mask_dir = get_create_mask_output_folder(config)
    default_output_dir = get_add_mask_band_output_folder(config)

    parser = argparse.ArgumentParser(description="Add water mask band to georeferenced satellite images (5-band stack).")
    parser.add_argument("--image-files", nargs="*", default=None,
                        help="Specific image file paths to process")
    parser.add_argument("--mask-dir", dest="mask_dir", type=str, default=default_mask_dir,
                        help=f"Path to input masks directory (default: {default_mask_dir})")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=default_output_dir,
                        help=f"Path to output directory for stacked GeoTIFFs (default: {default_output_dir})")
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum number of images to process from georeference folder (default: 5)")
    parser.add_argument("--no-viz", dest="visualize", action="store_false", default=True,
                        help="Skip generating PNG visual inspection plots")

    args = parser.parse_args()

    if args.image_files:
        image_files = args.image_files
    else:
        image_files = get_georeference_files(config, limit=args.limit)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting batch processing to add mask bands...")
    print(f"Input images: {len(image_files)} files")
    print(f"Mask files directory: {args.mask_dir}")
    print(f"Output directory: {args.output_dir}")
    
    processed_data = []
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        mask_path = find_corresponding_mask(image_path, args.mask_dir)
        if mask_path is None:
            print(f"  Warning: No mask found for {os.path.basename(image_path)}")
            continue
        
        print(f"  Found mask: {os.path.basename(mask_path)}")
        try:
            stacked, mask, output_paths = add_mask_band(image_path, mask_path, args.output_dir)
            processed_data.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'stacked': stacked,
                'mask': mask,
                'output_paths': output_paths,
                'basename': os.path.splitext(os.path.basename(image_path))[0]
            })
            print(f"  Successfully processed: {os.path.basename(output_paths['stacked'])}")
        except Exception as e:
            print(f"  Error processing {os.path.basename(image_path)}: {str(e)}")
            continue
    
    print(f"\nBatch processing complete. {len(processed_data)} files processed successfully.")
    
    if processed_data and args.visualize:
        create_individual_visualizations(processed_data, args.output_dir)
        print(f"\nAll processing complete! Check the output directory for:")
        print(f"  - *_mask.tif files (separate mask files)")
        print(f"  - *_with_mask.tif files (stacked images with mask band)")
        print(f"  - *_bands_visualization.png files (band visualizations)")
    elif not processed_data:
        print("No files were processed successfully.")

if __name__ == "__main__":
    main()

