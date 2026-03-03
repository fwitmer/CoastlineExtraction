"""
visualize_ndwi.py

Visualizes NDWI label quality for the CoastlineExtraction pipeline.
Generates a 4-panel output: Original RGB | NDWI Heatmap | Binary Mask | Overlay

Usage:
    # Single image
    python visualize_ndwi.py sample_data/PlanetLabs/image.tif

    # Entire directory
    python visualize_ndwi.py sample_data/PlanetLabs/

    # Custom output directory
    python visualize_ndwi.py sample_data/PlanetLabs/ --output_dir results/ndwi_viz
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import cv2
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from load_config import load_config


# ── helpers ──────────────────────────────────────────────────────────────────

# Default Gaussian blur parameters — matches ndwi_labels.py
KSIZE_BLUR = (9, 9)
SIGMA_X = 6
SIGMA_Y = 6

# Majority threshold — matches ndwi_labels.py MAJORITY_THRESHOLD
MAJORITY_THRESHOLD = 0.55


def load_bands(image_path):
    """
    Load Green (band 2) and NIR (last band) from a PlanetLabs GeoTIFF.
    Also loads RGB (bands 3,2,1) for display.
    Band order: 1=Blue, 2=Green, 3=Red, 4=NIR (PlanetScope)
                1=Blue, 2=Green, 3=Red, 4=RedEdge, 5=NIR (RapidEye)

    Raises:
        ValueError: If the image has fewer than 3 bands.
    """
    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError(
                f"{image_path} has only {src.count} band(s). "
                "Expected at least 3 (Blue, Green, Red) + NIR."
            )
        nir_idx = src.count          # NIR is always the last band
        blue  = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red   = src.read(3).astype(np.float32)
        nir   = src.read(nir_idx).astype(np.float32)
    return blue, green, red, nir


def compute_ndwi(green, nir, blur_kernel=None, blur_sigma_x=None, blur_sigma_y=None):
    """
    NDWI = (Green - NIR) / (Green + NIR)
    Returns values in [-1, 1]; positive → water, negative → land.

    Gaussian blur parameters default to matching ndwi_labels.py:
    kernel (9,9), sigmaX=6, sigmaY=6.

    Args:
        green: Green band array (float32).
        nir: NIR band array (float32).
        blur_kernel: Tuple (kH, kW) for GaussianBlur kernel size.
        blur_sigma_x: Sigma X for GaussianBlur.
        blur_sigma_y: Sigma Y for GaussianBlur.
    """
    if blur_kernel is None:
        blur_kernel = KSIZE_BLUR
    if blur_sigma_x is None:
        blur_sigma_x = SIGMA_X
    if blur_sigma_y is None:
        blur_sigma_y = SIGMA_Y

    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (green - nir) / (green + nir)
    ndwi = np.nan_to_num(ndwi, nan=0.0)
    # Gaussian blur — matches ndwi_labels.py defaults
    ndwi = cv2.GaussianBlur(ndwi, blur_kernel, blur_sigma_x, blur_sigma_y)
    return ndwi


def sliding_window_otsu(ndwi, window_size=64):
    """
    Simplified sliding-window Otsu thresholding (pixel-grid, majority voting).

    NOTE: This is a simplified approximation of the geometry-based sliding
    window in ndwi_labels.py, which uses shapefile transect points as window
    centres and rasterio.mask for extraction. Results will differ slightly.
    This version is intended for quick visual inspection of NDWI label quality,
    not for generating training labels.

    Args:
        ndwi: 2-D NDWI array (float, range [-1, 1]).
        window_size: Side length of each square window in pixels.

    Returns:
        Binary mask (uint8): 1 = water, 0 = land.
    """
    h, w = ndwi.shape
    vote_map  = np.zeros((h, w), dtype=np.int32)
    count_map = np.zeros((h, w), dtype=np.int32)

    # Convert NDWI to uint8 for cv2.threshold
    ndwi_8bit = np.clip((ndwi + 1) * 127.5, 0, 255).astype(np.uint8)

    step = window_size // 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            window = ndwi_8bit[y:y + window_size, x:x + window_size]
            if window.size == 0:
                continue
            # Skip windows that are mostly nodata / uniform
            if window.std() < 1e-3:
                continue
            thresh, _ = cv2.threshold(
                window, 0, 1,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            classified = (window >= thresh).astype(np.int32)
            vote_map[y:y + window_size, x:x + window_size]  += classified
            count_map[y:y + window_size, x:x + window_size] += 1

    # Majority vote → 1 = water, 0 = land
    # Threshold matches ndwi_labels.py MAJORITY_THRESHOLD (0.55)
    safe_count = np.where(count_map == 0, 1, count_map)
    binary_mask = (vote_map / safe_count >= MAJORITY_THRESHOLD).astype(np.uint8)
    return binary_mask


def normalize_for_display(arr):
    """Stretch array to [0, 1] for display."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def build_rgb(red, green, blue):
    """Stack and percentile-stretch to a display-ready RGB image."""
    rgb = np.stack([red, green, blue], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb


def print_ndwi_stats(ndwi, binary_mask, image_name):
    """Print statistics to help tune thresholds."""
    water_pct = binary_mask.mean() * 100
    print(f"\n── NDWI Stats: {image_name} ──")
    print(f"  Min  : {ndwi.min():.4f}")
    print(f"  Max  : {ndwi.max():.4f}")
    print(f"  Mean : {ndwi.mean():.4f}")
    print(f"  Std  : {ndwi.std():.4f}")
    print(f"  Water pixels : {water_pct:.1f}%")
    print(f"  Land  pixels : {100 - water_pct:.1f}%")


# ── main visualisation ────────────────────────────────────────────────────────

def visualize_single(image_path, output_dir, window_size):
    """
    Generate and save a 4-panel NDWI visualisation for one image.
    Panels: Original RGB | NDWI Heatmap | Binary Mask | Overlay
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load bands
    blue, green, red, nir = load_bands(image_path)

    # 2. Compute NDWI + binary mask
    ndwi        = compute_ndwi(green, nir)
    binary_mask = sliding_window_otsu(ndwi, window_size=window_size)

    # 3. Stats
    print_ndwi_stats(ndwi, binary_mask, image_name)

    # 4. Build display arrays
    rgb          = build_rgb(red, green, blue)
    ndwi_display = normalize_for_display(ndwi)

    # Colour overlay: water=blue, land=tan
    overlay = rgb.copy()
    overlay[binary_mask == 1] = [0.20, 0.45, 0.85]   # blue  → water
    overlay[binary_mask == 0] = overlay[binary_mask == 0] * 0.6 + np.array([0.6, 0.5, 0.3]) * 0.4

    # 5. Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"NDWI Label Quality — {image_name}", fontsize=13, fontweight='bold')

    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis('off')

    im = axes[1].imshow(ndwi_display, cmap='RdYlBu', vmin=0, vmax=1)
    axes[1].set_title("NDWI Heatmap\n(blue=water, red=land)")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Binary Mask\n(white=water, black=land)")
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis('off')
    water_patch = mpatches.Patch(color=[0.20, 0.45, 0.85], label='Water')
    axes[3].legend(handles=[water_patch], loc='lower right', fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{image_name}_ndwi_viz.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


def visualize_directory(input_dir, output_dir, window_size):
    """Process all .tif files in a directory."""
    tif_files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.lower().endswith('.tif') and 'udm' not in f.lower()
    ]
    if not tif_files:
        print(f"No .tif files found in {input_dir}")
        return

    print(f"Found {len(tif_files)} image(s) in {input_dir}")
    for path in tif_files:
        print(f"\nProcessing: {os.path.basename(path)}")
        try:
            visualize_single(path, output_dir, window_size)
        except Exception as e:
            print(f"  ERROR: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize NDWI label quality for CoastlineExtraction pipeline."
    )
    parser.add_argument(
        "input",
        help="Path to a single .tif image or a directory of .tif images."
    )
    parser.add_argument(
        "--output_dir",
        default="ndwi_visualizations",
        help="Directory to save PNG outputs (default: ndwi_visualizations/)."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Sliding window size for Otsu thresholding (default: from config or 64)."
    )
    parser.add_argument(
        "--blur_kernel",
        type=int,
        default=None,
        help="Gaussian blur kernel size, e.g. 9 for (9,9) (default: from config or 9)."
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=None,
        help="Gaussian blur sigma (default: from config or 6)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load NDWI parameters from config, with CLI overrides
    try:
        config = load_config()
        ndwi_cfg = config.get("ndwi", {})
    except Exception:
        ndwi_cfg = {}

    window_size = args.window_size or ndwi_cfg.get("window_size", 64)

    blur_k = args.blur_kernel or ndwi_cfg.get("blur_kernel", 9)
    blur_kernel = (blur_k, blur_k)

    blur_sigma = args.blur_sigma
    if blur_sigma is None:
        blur_sigma = ndwi_cfg.get("blur_sigma", 6)
    blur_sigma_x = blur_sigma
    blur_sigma_y = blur_sigma

    # Stash in module-level defaults so compute_ndwi picks them up
    global KSIZE_BLUR, SIGMA_X, SIGMA_Y
    KSIZE_BLUR = blur_kernel
    SIGMA_X = int(blur_sigma_x)
    SIGMA_Y = int(blur_sigma_y)

    print(f"Using window_size={window_size}, blur_kernel={blur_kernel}, "
          f"blur_sigma=({blur_sigma_x}, {blur_sigma_y})")

    if os.path.isdir(args.input):
        visualize_directory(args.input, args.output_dir, window_size)
    elif os.path.isfile(args.input):
        visualize_single(args.input, args.output_dir, window_size)
    else:
        print(f"ERROR: '{args.input}' is not a valid file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()