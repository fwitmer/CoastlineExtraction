import os
import glob
import random
import time
import numpy as np
import rasterio as rio
import cv2
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

def stretch_band_lut(band):
    """Percentile-based linear stretching using a fast 1D Lookup Table (LUT) for 16-bit to 8-bit mapping."""
    valid_mask = band > 0
    valid_pixels = band[valid_mask]
    if len(valid_pixels) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    
    # Subsample for percentile (every 100th pixel is sufficient and extremely fast)
    subsampled = valid_pixels[::100]
    if len(subsampled) == 0:
        subsampled = valid_pixels
        
    p2, p98 = np.percentile(subsampled, (2, 98))
    if p98 == p2:
        return np.zeros_like(band, dtype=np.uint8)
        
    # Build 16-bit to 8-bit lookup table
    lut = np.clip((np.arange(65536).astype(np.float32) - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    lut[0] = 0  # Map background (no-data) to 0
    
    return lut[band]

def main():
    # Resolve sibling directory "new_data" relative to the script's repository location
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_data_base = os.path.abspath(os.path.join(repo_root, "..", "new_data"))
    years = ["2021", "2024", "2025"]
    
    # 1. Collect all full scenes across all years
    scenes = []
    for year in years:
        tif_paths = glob.glob(os.path.join(new_data_base, year, "tif", "*.tif"))
        for p in tif_paths:
            scenes.append((year, p))
            
    print(f"Total original scenes available: {len(scenes)}")
    
    # Select 10 representative scenes with seed for reproducibility
    random.seed(42)
    selected_scenes = random.sample(scenes, 10)
    
    # Clear and recreate the samples directory
    samples_base_dir = os.path.join(new_data_base, "samples")
    if os.path.exists(samples_base_dir):
        import shutil
        shutil.rmtree(samples_base_dir)
    os.makedirs(samples_base_dir, exist_ok=True)
    
    print(f"Generating optimized full scene visualizations and NDWI water masks in: {samples_base_dir}")
    
    samples_info = []

    for idx, (year, tif_path) in enumerate(selected_scenes):
        t_scene_start = time.time()
        sample_num = idx + 1
        sample_dir = os.path.join(samples_base_dir, f"sample_{sample_num}")
        os.makedirs(sample_dir, exist_ok=True)
        
        filename = os.path.basename(tif_path)
        print(f"\nProcessing Scene {sample_num}/10: {filename} (Year: {year})")
        
        # Get matching full UDM mask path
        parts = filename.split("_")
        item_id = "_".join(parts[0:4])
        mask_filename = f"{item_id}_3B_udm2_clip.tif"
        mask_path = os.path.join(new_data_base, year, "udm_masks", mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"  Warning: Mask path not found: {mask_path}")
            continue

        local_tif = f"/tmp/sample_{sample_num}_image.tif"
        local_mask = f"/tmp/sample_{sample_num}_mask.tif"

        try:
            # Copy to local /tmp to optimize I/O on NFS
            shutil_start = time.time()
            import shutil
            shutil.copy2(tif_path, local_tif)
            shutil.copy2(mask_path, local_mask)
            
            # 2. Open file and read downsampled bands to avoid CPU/Memory thrashing
            with rio.open(local_tif) as src:
                h, w = src.height, src.width
                scale = 1.0
                if max(h, w) > 1500:
                    scale = 1500.0 / max(h, w)
                new_h = int(round(h * scale))
                new_w = int(round(w * scale))
                
                # Downsample multi-band simultaneously
                img_data = src.read(
                    [1, 2, 3, 4],
                    out_shape=(4, new_h, new_w),
                    resampling=rio.enums.Resampling.bilinear
                )
                
            blue, green, red, nir = img_data[0], img_data[1], img_data[2], img_data[3]
            
            # Generate stretched BGR for visualization
            blue_s = stretch_band_lut(blue)
            green_s = stretch_band_lut(green)
            red_s = stretch_band_lut(red)
            img_bgr = np.dstack((blue_s, green_s, red_s))
            image_jpg_path = os.path.join(sample_dir, "image.jpg")
            cv2.imwrite(image_jpg_path, img_bgr)
            
            # 3. Read downsampled mask
            with rio.open(local_mask) as src:
                mask_data = src.read(
                    [1, 2, 3, 4, 5, 6],
                    out_shape=(6, new_h, new_w),
                    resampling=rio.enums.Resampling.nearest
                )
                
            udm_clear = mask_data[0] == 1
            udm_snow = mask_data[1] == 1
            udm_shadow = mask_data[2] == 1
            udm_lhaze = mask_data[3] == 1
            udm_hhaze = mask_data[4] == 1
            udm_cloud = mask_data[5] == 1
            
            valid_area = blue > 0
            total_valid_pixels = np.sum(valid_area)
            
            # Config 7 - Visible Brightness Index
            brightness = (blue.astype(np.float32) + green.astype(np.float32) + red.astype(np.float32)) / 3.0 / 10000.0
            
            # Otsu thresholding pool from valid, clear, non-cloud areas
            otsu_pool_pixels = brightness[valid_area & ~udm_snow & ~udm_cloud]
            if len(otsu_pool_pixels) > 1000:
                otsu_pool_pixels = otsu_pool_pixels[::5] # Subsample pool further to make skimage threshold_otsu instant
                
            if len(otsu_pool_pixels) > 100:
                t_otsu = threshold_otsu(otsu_pool_pixels)
                adjusted_threshold = t_otsu * 0.95
            else:
                t_otsu = 0.0
                adjusted_threshold = 0.0
                
            # Perform vectorized quality mask classification
            if adjusted_threshold >= 0.20:
                otsu_clouds = (brightness > adjusted_threshold) & valid_area & ~udm_snow & ~udm_cloud
            else:
                otsu_clouds = np.zeros_like(valid_area, dtype=bool)
                
            custom_shadow = (brightness < 0.03) & valid_area & ~udm_cloud & ~otsu_clouds
            custom_haze = (brightness > 0.14) & valid_area & ~udm_snow & ~udm_shadow & ~udm_cloud & ~otsu_clouds
            
            final_cloud = udm_cloud | otsu_clouds | custom_haze
            final_shadow = udm_shadow | custom_shadow
            final_haze = udm_lhaze | udm_hhaze
            final_snow = udm_snow
            final_clear = valid_area & ~final_cloud & ~final_shadow & ~final_snow & ~final_haze
            
            # Calculate pixel stats
            count_clear = np.sum(final_clear)
            count_cloud = np.sum(final_cloud)
            count_haze = np.sum(final_haze)
            count_snow = np.sum(final_snow)
            count_shadow = np.sum(final_shadow)
            
            # Create color mask
            mask_color = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            mask_color[valid_area] = [180, 180, 180] # background (Gray)
            
            # Colors (BGR):
            mask_color[final_clear & valid_area] = [80, 180, 80]     # Clear (Light Green)
            mask_color[final_shadow & valid_area] = [40, 40, 40]     # Shadow (Dark Gray)
            mask_color[final_haze & valid_area] = [100, 200, 255]    # Haze (Yellow/Orange)
            mask_color[final_snow & valid_area] = [230, 230, 100]    # Snow/Ice (Cyan)
            mask_color[final_cloud & valid_area] = [80, 80, 255]     # Heavy Cloud/Otsu/Custom (Red)
            
            mask_jpg_path = os.path.join(sample_dir, "mask_visualization.jpg")
            cv2.imwrite(mask_jpg_path, mask_color)
            
            # 4. NDWI calculation & Otsu thresholding (Match local conventions with CV2)
            green_f = green.astype(np.float32)
            nir_f = nir.astype(np.float32)
            denom = green_f + nir_f
            
            ndwi = np.zeros_like(green_f)
            valid_denom = denom > 0
            ndwi[valid_denom] = (green_f[valid_denom] - nir_f[valid_denom]) / denom[valid_denom]
            ndwi[np.isnan(ndwi)] = 0
            
            # Scale Gaussian blur kernel size based on downsampling scale (default kernel in ndwi_labels.py is (9, 9))
            blur_k = int(round(9 * scale))
            if blur_k % 2 == 0:
                blur_k += 1
            if blur_k < 3:
                blur_k = 3
                
            ndwi_blurred = cv2.GaussianBlur(ndwi, (blur_k, blur_k), 0)
            
            # Scale to 8-bit using existing codebase convention: ndwi_8bit = ((ndwi * 127) + 128).astype(np.uint8)
            ndwi_8bit = np.floor((ndwi_blurred * 127.0) + 128.0)
            ndwi_8bit = np.clip(ndwi_8bit, 0, 255).astype(np.uint8)
            ndwi_8bit[~valid_area] = 0
            
            # Dynamic Otsu thresholding ONLY on valid area to avoid black-border bias
            valid_ndwi_pixels = ndwi_8bit[valid_area]
            if len(valid_ndwi_pixels) > 100:
                ndwi_t_otsu, _ = cv2.threshold(valid_ndwi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ndwi_water_mask = (ndwi_8bit >= ndwi_t_otsu) & valid_area
                # Convert threshold back to original [-1.0, 1.0] float representation
                ndwi_t_otsu_float = float((ndwi_t_otsu - 128.0) / 127.0)
            else:
                ndwi_t_otsu = 128
                ndwi_t_otsu_float = 0.0
                ndwi_water_mask = np.zeros_like(valid_area, dtype=bool)
                
            # Render NDWI Land/Water color mask:
            # Water = Blue (BGR: [240, 100, 0]), Land = Greenish/Brown (BGR: [50, 130, 50]), Background = Black ([0, 0, 0])
            ndwi_color = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            ndwi_color[valid_area] = [50, 130, 50]       # Land (Greenish/Brown)
            ndwi_color[ndwi_water_mask] = [240, 100, 0]  # Water (Blue)
            
            ndwi_jpg_path = os.path.join(sample_dir, "ndwi_water_mask.jpg")
            cv2.imwrite(ndwi_jpg_path, ndwi_color)
            
            # Compute final statistics
            if total_valid_pixels > 0:
                cloud_fraction = count_cloud / total_valid_pixels
                snow_fraction = count_snow / total_valid_pixels
                shadow_fraction = count_shadow / total_valid_pixels
                haze_fraction = count_haze / total_valid_pixels
                clear_fraction = count_clear / total_valid_pixels
                water_fraction = np.sum(ndwi_water_mask) / total_valid_pixels
            else:
                cloud_fraction = snow_fraction = shadow_fraction = haze_fraction = clear_fraction = water_fraction = 0.0
                
            # Write details text file
            details_path = os.path.join(sample_dir, "details.txt")
            with open(details_path, "w") as f:
                f.write(f"Sample Number: {sample_num}\n")
                f.write(f"Year: {year}\n")
                f.write(f"Scene ID: {item_id}\n")
                f.write(f"Original Imagery Scene: {filename}\n")
                f.write(f"Original Mask Scene: {mask_filename}\n")
                f.write(f"Processed Resolution: {new_w}x{new_h} (Scaled from {w}x{h})\n")
                f.write(f"Algorithm Configuration: Config 7 (Balanced Sensitive)\n")
                f.write(f"Calculated Cloud Otsu Threshold: {t_otsu:.4f}\n")
                f.write(f"Adjusted Cloud Otsu Threshold: {adjusted_threshold:.4f}\n")
                f.write(f"Calculated NDWI Otsu Threshold (8-bit): {ndwi_t_otsu}\n")
                f.write(f"Calculated NDWI Otsu Threshold (Float): {ndwi_t_otsu_float:.4f}\n\n")
                f.write(f"--- Area Fractions ---\n")
                f.write(f"Clear Area Fraction: {clear_fraction*100:.2f}%\n")
                f.write(f"Cloud/Haze Cover Fraction: {cloud_fraction*100:.2f}%\n")
                f.write(f"Snow/Ice Cover Fraction: {snow_fraction*100:.2f}%\n")
                f.write(f"Haze Cover Fraction: {haze_fraction*100:.2f}%\n")
                f.write(f"Shadow Cover Fraction: {shadow_fraction*100:.2f}%\n")
                f.write(f"Water Area Fraction (NDWI): {water_fraction*100:.2f}%\n")
                
            print(f"  ✓ Saved all visualizations to {sample_dir} in {time.time() - t_scene_start:.2f}s")
            
            samples_info.append({
                "sample_num": sample_num,
                "year": year,
                "scene_id": item_id,
                "image_path": image_jpg_path,
                "mask_path": mask_jpg_path,
                "ndwi_path": ndwi_jpg_path,
                "clear": clear_fraction,
                "cloud": cloud_fraction,
                "snow": snow_fraction,
                "shadow": shadow_fraction,
                "water": water_fraction,
                "ndwi_thresh": ndwi_t_otsu_float
            })

        except Exception as e:
            print(f"Error processing scene {filename}: {e}")
        finally:
            # Clean up local temporary TIFFs
            if os.path.exists(local_tif):
                os.remove(local_tif)
            if os.path.exists(local_mask):
                os.remove(local_mask)

    print("\nAll full-scene images and masks computed successfully. Creating PDF documents...")
    
    # 5. Create PDF 1: Image & Configuration 7 Mask Side-by-Side
    pdf1_path = os.path.join(new_data_base, "config7_side_by_side_samples.pdf")
    print(f"Saving PDF 1 to: {pdf1_path}")
    
    mask_patches = [
        mpatches.Patch(color='#50B450', label='Clear (Light Green)'),
        mpatches.Patch(color='#282828', label='Shadow (Dark Gray)'),
        mpatches.Patch(color='#FFC864', label='Light Haze (Yellow/Orange)'),
        mpatches.Patch(color='#64E6E6', label='Snow/Ice (Cyan)'),
        mpatches.Patch(color='#FF5050', label='Heavy Cloud/Otsu/Custom (Red)')
    ]
    
    with PdfPages(pdf1_path) as pdf:
        for info in samples_info:
            fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), dpi=120) # Sized and set dpi for fast rendering
            
            # Read images fast
            img = cv2.cvtColor(cv2.imread(info["image_path"]), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(info["mask_path"]), cv2.COLOR_BGR2RGB)
            
            # Plot True Color
            axes[0].imshow(img)
            axes[0].set_title("True Color Image (Stretched RGB)", fontsize=10, pad=5)
            axes[0].axis("off")
            
            # Plot Mask
            axes[1].imshow(mask)
            axes[1].set_title("Otsu-Refined Custom Quality Mask (Config 7)", fontsize=10, pad=5)
            axes[1].axis("off")
            
            # Text annotation details
            stats_text = (
                f"Year: {info['year']}   |   Scene ID: {info['scene_id']}\n"
                f"Clear: {info['clear']*100:.1f}%  |  Cloud: {info['cloud']*100:.1f}%  |  "
                f"Snow: {info['snow']*100:.1f}%  |  Shadow: {info['shadow']*100:.1f}%"
            )
            fig.text(0.5, 0.12, stats_text, ha='center', fontsize=9, fontweight='semibold',
                     bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Add Legend
            fig.legend(handles=mask_patches, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.03))
            
            # Super title
            fig.suptitle(f"Sample {info['sample_num']}: Quality Control Visual Analysis (Deering, AK)", 
                         fontsize=12, fontweight='bold', y=0.96)
            
            plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.95, wspace=0.1)
            pdf.savefig(fig)
            plt.close(fig)
            
    print("  ✓ PDF 1 Created!")

    # 6. Create PDF 2: Image, Configuration 7 Mask, and NDWI Land/Water Mask Side-by-Side-by-Side
    pdf2_path = os.path.join(new_data_base, "config7_ndwi_three_way_samples.pdf")
    print(f"Saving PDF 2 to: {pdf2_path}")
    
    ndwi_patches = [
        mpatches.Patch(color='#0064F0', label='Water (Blue)'),
        mpatches.Patch(color='#328232', label='Land (Greenish-Brown)'),
        mpatches.Patch(color='#000000', label='No-data (Black)')
    ]
    
    with PdfPages(pdf2_path) as pdf:
        for info in samples_info:
            fig, axes = plt.subplots(1, 3, figsize=(14, 8.5), dpi=120)
            
            # Read images fast
            img = cv2.cvtColor(cv2.imread(info["image_path"]), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(info["mask_path"]), cv2.COLOR_BGR2RGB)
            ndwi_mask = cv2.cvtColor(cv2.imread(info["ndwi_path"]), cv2.COLOR_BGR2RGB)
            
            # Plot True Color
            axes[0].imshow(img)
            axes[0].set_title("True Color Stretched RGB", fontsize=9, pad=5)
            axes[0].axis("off")
            
            # Plot Quality Mask
            axes[1].imshow(mask)
            axes[1].set_title("Quality Mask (Config 7)", fontsize=9, pad=5)
            axes[1].axis("off")
            
            # Plot NDWI land-water
            axes[2].imshow(ndwi_mask)
            axes[2].set_title("NDWI Land/Water Mask (Otsu)", fontsize=9, pad=5)
            axes[2].axis("off")
            
            # Text details
            stats_text = (
                f"Year: {info['year']}   |   Scene ID: {info['scene_id']}\n"
                f"Cloud: {info['cloud']*100:.1f}%  |  Snow: {info['snow']*100:.1f}%  |  "
                f"Water (NDWI): {info['water']*100:.1f}%  |  NDWI Otsu Threshold: {info['ndwi_thresh']:.3f}"
            )
            fig.text(0.5, 0.12, stats_text, ha='center', fontsize=9, fontweight='semibold',
                     bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Add legends
            legend_ax1 = fig.legend(handles=mask_patches, loc='lower left', ncol=3, fontsize=8, bbox_to_anchor=(0.08, 0.02))
            legend_ax2 = fig.legend(handles=ndwi_patches, loc='lower right', ncol=3, fontsize=8, bbox_to_anchor=(0.92, 0.02))
            
            # Super title
            fig.suptitle(f"Sample {info['sample_num']}: Land/Water Segmentation vs. Quality Control (Deering, AK)", 
                         fontsize=12, fontweight='bold', y=0.96)
            
            plt.subplots_adjust(top=0.88, bottom=0.18, left=0.04, right=0.96, wspace=0.12)
            pdf.savefig(fig)
            plt.close(fig)
            
    print("  ✓ PDF 2 Created!")
    print(f"\nAll tasks finished successfully!\nPDFs located at:\n1. {pdf1_path}\n2. {pdf2_path}")

if __name__ == "__main__":
    main()
