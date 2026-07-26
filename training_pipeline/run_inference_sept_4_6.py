import os
import sys
import numpy as np
import geopandas as gpd
import rasterio as rio
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import skimage.measure
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, box
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Path resolution relative to repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
if script_dir not in sys.path:
    sys.path.append(script_dir)

from train_and_eval_pipeline import UNet, AttentionUNet

UTM_ZONE_3N = 'EPSG:32603'

# ----------------------------
# Helper & Inference Functions
# ----------------------------
def stretch_rgb_image(image_data, dataset_mask=None):
    """
    Applies a 2%-98% percentile contrast stretch on valid non-zero satellite pixels
    so the RGB visualization matches bright QGIS rendering.
    """
    scaled = np.zeros_like(image_data, dtype=np.uint8)
    for c in range(3):
        channel = image_data[:, :, c].astype(np.float32)
        valid = channel[dataset_mask] if dataset_mask is not None else channel[channel > 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 > p2:
                channel = np.clip((channel - p2) / (p98 - p2) * 255.0, 0, 255)
        scaled[:, :, c] = channel.astype(np.uint8)
    return scaled

def extract_model_coastline(model, image_path, transform, device):
    model.eval()
    with rio.open(image_path) as src:
        image_data = src.read([3, 2, 1])
        dataset_mask = src.dataset_mask() > 0
        h_orig, w_orig = image_data.shape[1], image_data.shape[2]
        
    scaled_data = (np.clip(image_data.astype(np.float32) / 10000.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    scaled_data = np.transpose(scaled_data, (1, 2, 0))
        
    image = Image.fromarray(scaled_data)
    transform_resize = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform_resize(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
        pred_mask = (output > 0.5).astype("uint8")
        
    # Resize back to original dimensions & mask out NoData regions
    pred_mask_resized = Image.fromarray(pred_mask * 255).resize((w_orig, h_orig), Image.NEAREST)
    pred_mask_np = (np.array(pred_mask_resized) > 0) & dataset_mask
    
    # Eroded dataset mask (points must be strictly inside satellite coverage, excluding artificial image borders)
    eroded_mask = ndimage.binary_erosion(dataset_mask, iterations=4)
    
    contours = skimage.measure.find_contours(pred_mask_np.astype(np.float32), 0.5)
    
    lines = []
    for contour in contours:
        rows = np.clip(np.round(contour[:, 0]).astype(int), 0, h_orig - 1)
        cols = np.clip(np.round(contour[:, 1]).astype(int), 0, w_orig - 1)
        
        in_bounds = eroded_mask[rows, cols]
        split_indices = np.where(~in_bounds)[0]
        segments = np.split(contour, split_indices)
        for seg in segments:
            seg_clean = seg[eroded_mask[np.clip(np.round(seg[:, 0]).astype(int), 0, h_orig - 1),
                                        np.clip(np.round(seg[:, 1]).astype(int), 0, w_orig - 1)]]
            if len(seg_clean) >= 2:
                xs, ys = rio.transform.xy(transform, seg_clean[:, 0], seg_clean[:, 1])
                lines.append(LineString(list(zip(xs, ys))))
    return lines, pred_mask_np

def plot_water_land_prediction(t_path, pred_mask_np, pred_lines, plot_out_path):
    """
    Plots a 3-panel visualization of:
    1. Stretched RGB Satellite Image (matching QGIS brightness)
    2. Binary Classification (Land vs Water vs NoData)
    3. Stretched RGB Image with Translucent Water Mask Overlay & Clean Coastline Boundary
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    tile_name = os.path.basename(t_path)
    
    with rio.open(t_path) as src:
        image_data = src.read([3, 2, 1])
        image_data = np.transpose(image_data, (1, 2, 0))
        dataset_mask = src.dataset_mask() > 0
        tile_bounds = src.bounds
        extent = [tile_bounds.left, tile_bounds.right, tile_bounds.bottom, tile_bounds.top]

    rgb_stretched = stretch_rgb_image(image_data, dataset_mask)

    # Panel 1: RGB Satellite Image
    axes[0].imshow(rgb_stretched, extent=extent)
    axes[0].set_title("RGB Satellite Image (QGIS Stretch)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Binary Classification (Land=0, Water=1, NoData=2)
    class_map = np.zeros(pred_mask_np.shape, dtype=np.uint8)
    class_map[pred_mask_np] = 1
    class_map[~dataset_mask] = 2

    cmap_water_land = mcolors.ListedColormap(['#8B5A2B', '#1E88E5', '#FFFFFF'])  # Land (Brown), Water (Blue), NoData (White)
    axes[1].imshow(class_map, extent=extent, cmap=cmap_water_land)
    axes[1].set_title("Binary Classification (Water vs Land)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    legend_elements = [
        Patch(facecolor='#1E88E5', label='Water (Predicted)'),
        Patch(facecolor='#8B5A2B', label='Land (Predicted)'),
        Patch(facecolor='#FFFFFF', edgecolor='gray', label='NoData (Background)')
    ]
    axes[1].legend(handles=legend_elements, loc="upper right")

    # Panel 3: Satellite Overlay with Water Mask & Coastline Contour
    axes[2].imshow(rgb_stretched, extent=extent)
    water_overlay = np.zeros((*pred_mask_np.shape, 4), dtype=np.float32)
    water_overlay[pred_mask_np] = [0.12, 0.53, 0.90, 0.45]  # Semi-transparent blue for water
    axes[2].imshow(water_overlay, extent=extent)
    
    for idx, line in enumerate(pred_lines):
        lbl = "Extracted Coastline" if idx == 0 else ""
        axes[2].plot(*line.xy, color="cyan", linewidth=2.0, linestyle="--", label=lbl)
    if pred_lines:
        axes[2].legend(loc="upper right")
        
    axes[2].set_title("Water Mask & Clean Coastline Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f"Model Binary Water/Land Prediction Analysis - {tile_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(plot_out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved binary water/land visualization plot to: {plot_out_path}")

def calculate_rmse_on_transects(predicted_lines, trans_deering, ref_distances, usgs_distances, hires_distances):
    empty_regional = {1: float('nan'), 2: float('nan'), 3: float('nan'), 4: float('nan'), 5: float('nan')}
    if not predicted_lines:
        return float('nan'), float('nan'), float('nan'), empty_regional
        
    combined_lines = MultiLineString(predicted_lines)
    errors_planet = []
    errors_usgs = []
    errors_hires = []
    regional_errors = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    for idx, row in trans_deering.iterrows():
        oid = int(row['TransOrder'])
        t_geom = row.geometry
        
        region_id = 5
        if oid >= 17443:
            region_id = 1  # Western Region
        elif oid >= 17394:
            region_id = 2  # Northern Region
        elif oid >= 17370:
            region_id = 3  # Central Region
        elif oid >= 17337:
            region_id = 4  # Town Region
            
        pt_int = combined_lines.intersection(t_geom)
        dist = None
        if not pt_int.is_empty:
            if isinstance(pt_int, Point):
                dist = t_geom.project(pt_int)
            elif isinstance(pt_int, MultiPoint):
                dist = min([t_geom.project(pt) for pt in pt_int.geoms])
            elif hasattr(pt_int, 'geoms'):
                pts = [pt for pt in pt_int.geoms if isinstance(pt, Point)]
                if pts:
                    dist = min([t_geom.project(pt) for pt in pts])
                    
        if dist is not None:
            if oid in ref_distances:
                err_p = dist - ref_distances[oid]
                errors_planet.append(err_p)
                regional_errors[region_id].append(err_p)
            if oid in usgs_distances:
                err_u = dist - usgs_distances[oid]
                errors_usgs.append(err_u)
            if oid in hires_distances:
                err_h = dist - hires_distances[oid]
                errors_hires.append(err_h)
                
    rmse_planet = np.sqrt(np.mean(np.square(errors_planet))) if errors_planet else float('nan')
    rmse_usgs = np.sqrt(np.mean(np.square(errors_usgs))) if errors_usgs else float('nan')
    rmse_hires = np.sqrt(np.mean(np.square(errors_hires))) if errors_hires else float('nan')
    
    regional_rmse = {}
    for r_id, errs in regional_errors.items():
        regional_rmse[r_id] = np.sqrt(np.mean(np.square(errs))) if errs else float('nan')
    
    return rmse_planet, rmse_usgs, rmse_hires, regional_rmse

def plot_predictions_comparison(t_path, u_lines, planet_union, hires_union, usgs_union, transform, plot_out_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"U-Net Predicted vs. Ground Truth Coastline\n(Tile: {os.path.basename(t_path)})", fontsize=14, fontweight="bold")
    
    with rio.open(t_path) as src:
        image_data = src.read([3, 2, 1])
        image_data = np.transpose(image_data, (1, 2, 0))
        dataset_mask = src.dataset_mask() > 0
        tile_bounds = src.bounds
        extent = [tile_bounds.left, tile_bounds.right, tile_bounds.bottom, tile_bounds.top]
        
    rgb_stretched = stretch_rgb_image(image_data, dataset_mask)
    ax.imshow(rgb_stretched, extent=extent, alpha=0.95)
    tile_box = box(*tile_bounds)
    
    p_cropped = planet_union.intersection(tile_box)
    h_cropped = hires_union.intersection(tile_box)
    u_cropped = usgs_union.intersection(tile_box)
    
    # Plot Ground Truths
    def plot_geom(geom, color, linewidth, label, linestyle="-"):
        if geom.is_empty:
            return
        if isinstance(geom, LineString):
            ax.plot(*geom.xy, color=color, linewidth=linewidth, label=label, linestyle=linestyle)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                ax.plot(*line.xy, color=color, linewidth=linewidth, label=label, linestyle=linestyle)
        elif hasattr(geom, "geoms"):
            for sub_geom in geom.geoms:
                plot_geom(sub_geom, color, linewidth, label, linestyle)
                
    plot_geom(p_cropped, color="orange", linewidth=2.5, label="Planet Labs Reference")
    plot_geom(h_cropped, color="red", linewidth=2.5, label="Manual Hi-Res GT")
    plot_geom(u_cropped, color="green", linewidth=2.5, label="USGS Coastline")
    
    # Plot predicted coastline
    for idx, line in enumerate(u_lines):
        lbl = "Predicted Coastline (U-Net)" if idx == 0 else ""
        ax.plot(*line.xy, color="cyan", linewidth=2.0, linestyle="--", label=lbl)
        
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l:
            by_label[l] = h
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    plt.tight_layout()
    plt.savefig(plot_out_path, dpi=300)
    plt.close()
    print(f"Saved visualization plot to: {plot_out_path}")

def main():
    print("==================================================")
    print("RUNNING MODEL INFERENCE ON SEPT 4 & 6")
    print("==================================================")
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = os.path.join(repo_root, "output_models")
    attn_model_path = os.path.join(output_dir, "best_deep_attn_unet_8epochs.pth")
    unet_model_path = os.path.join(output_dir, "best_deep_unet_8epochs.pth")
    legacy_model_path = os.path.join(output_dir, "best_unet_8epochs.pth")
    
    if os.path.exists(attn_model_path):
        model_path = attn_model_path
        model = AttentionUNet(n_channels=3, n_classes=1)
        model_type = "Attention U-Net"
    elif os.path.exists(unet_model_path):
        model_path = unet_model_path
        model = UNet(n_channels=3, n_classes=1)
        model_type = "Standard U-Net"
    elif os.path.exists(legacy_model_path):
        model_path = legacy_model_path
        model = UNet(n_channels=3, n_classes=1)
        model_type = "Legacy Standard U-Net"
    else:
        print(f"Error: No model weights found at {attn_model_path} or {unet_model_path}.")
        sys.exit(1)
            
    print(f"Loading [{model_type}] weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    p_transects = os.path.join(repo_root, "USGS_Coastlines", "WestChukchi_exposed_STepr_rates", "WestChukchi_exposed_STepr_rates.shp")
    p_usgs = os.path.join(repo_root, "USGS_Coastlines", "Deering_shorelines_2016.shp")
    p_hires = os.path.join(repo_root, "ground_truth", "2016_HiRes_Final_Coastline.shp")
    
    planet_candidates = [
        os.path.join(repo_root, "existing_data", "DigitizedCoastlines", "PlanetCoastline_gt", "09_09_2016", "9_9_16_PlanetCoastline.shp"),
        os.path.join(repo_root, "..", "existing_data", "DigitizedCoastlines", "PlanetCoastline_gt", "09_09_2016", "9_9_16_PlanetCoastline.shp")
    ]
    p_planet = next((p for p in planet_candidates if os.path.exists(p)), planet_candidates[0])
    
    print("Loading transects and reference coastlines...")
    trans_deering = gpd.read_file(p_transects).to_crs(UTM_ZONE_3N)
    trans_deering = trans_deering[trans_deering['BaselineID'] == 117].sort_values('TransOrder')
    usgs = gpd.read_file(p_usgs).to_crs(UTM_ZONE_3N)
    hires = gpd.read_file(p_hires).to_crs(UTM_ZONE_3N)
    planet = gpd.read_file(p_planet).to_crs(UTM_ZONE_3N)
    
    usgs_union = usgs.union_all() if hasattr(usgs, 'union_all') else usgs.unary_union
    hires_union = hires.union_all() if hasattr(hires, 'union_all') else hires.unary_union
    planet_union = planet.union_all() if hasattr(planet, 'union_all') else planet.unary_union
    
    # Precompute reference intersection distances
    planet_distances = {}
    usgs_distances = {}
    hires_distances = {}
    for idx, row in trans_deering.iterrows():
        oid = int(row['TransOrder'])
        t_geom = row.geometry
        
        # Planet Labs Reference
        p_int = t_geom.intersection(planet_union)
        if not p_int.is_empty:
            if isinstance(p_int, Point):
                planet_distances[oid] = t_geom.project(p_int)
            elif isinstance(p_int, MultiPoint):
                planet_distances[oid] = min([t_geom.project(pt) for pt in p_int.geoms])
            elif hasattr(p_int, 'geoms'):
                pts = [pt for pt in p_int.geoms if isinstance(pt, Point)]
                if pts:
                    planet_distances[oid] = min([t_geom.project(pt) for pt in pts])
                    
        # USGS
        u_int = t_geom.intersection(usgs_union)
        if not u_int.is_empty:
            if isinstance(u_int, Point):
                usgs_distances[oid] = t_geom.project(u_int)
            elif isinstance(u_int, MultiPoint):
                usgs_distances[oid] = min([t_geom.project(pt) for pt in u_int.geoms])
            elif hasattr(u_int, 'geoms'):
                pts = [pt for pt in u_int.geoms if isinstance(pt, Point)]
                if pts:
                    usgs_distances[oid] = min([t_geom.project(pt) for pt in pts])
                    
        # HiRes
        h_int = t_geom.intersection(hires_union)
        if not h_int.is_empty:
            if isinstance(h_int, Point):
                hires_distances[oid] = t_geom.project(h_int)
            elif isinstance(h_int, MultiPoint):
                hires_distances[oid] = min([t_geom.project(pt) for pt in h_int.geoms])
            elif hasattr(h_int, 'geoms'):
                pts = [pt for pt in h_int.geoms if isinstance(pt, Point)]
                if pts:
                    hires_distances[oid] = min([t_geom.project(pt) for pt in pts])
                    
    # Test tile paths
    test_tiles = [
        os.path.join(repo_root, "test_data_4_6_sept", "sept_4", "files", "369619_2016-09-04_RE2_3A_Analytic_SR_clip.tif"),
        os.path.join(repo_root, "test_data_4_6_sept", "sept_6", "files", "369619_2016-09-06_RE5_3A_Analytic_SR_clip.tif")
    ]
    
    vis_dir = os.path.join(repo_root, "inference_outputs")
    os.makedirs(vis_dir, exist_ok=True)
    
    for t_path in test_tiles:
        print(f"\nProcessing test tile: {os.path.basename(t_path)}")
        if not os.path.exists(t_path):
            print(f"Error: Tile {t_path} not found.")
            continue
            
        with rio.open(t_path) as src:
            transform = src.transform
            
        pred_lines, pred_mask_np = extract_model_coastline(model, t_path, transform, device)
        
        # Save predicted coastline contours as shapefile
        tile_name = os.path.splitext(os.path.basename(t_path))[0]
        shp_out_path = os.path.join(vis_dir, f"{tile_name}_model_predicted_coastline.shp")
        if pred_lines:
            gdf_pred = gpd.GeoDataFrame(geometry=pred_lines, crs=UTM_ZONE_3N)
            gdf_pred.to_file(shp_out_path)
            print(f"Saved predicted coastline shapefile to: {shp_out_path}")
        else:
            print("No coastline contours detected for this tile.")
            
        # Calculate RMSE scores
        rmse_p, rmse_u, rmse_h, regional_rmse = calculate_rmse_on_transects(pred_lines, trans_deering, planet_distances, usgs_distances, hires_distances)
        
        region_names = {
            1: "Western Region",
            2: "Northern Region",
            3: "Central Region",
            4: "Town Region",
            5: "East Region"
        }
        
        print(f"\nRESULTS FOR TILE: {os.path.basename(t_path)}")
        print(f"  Overall RMSE vs Planet Labs Ref: {rmse_p:.2f} m")
        print(f"  Overall RMSE vs USGS Coastlines: {rmse_u:.2f} m")
        print(f"  Overall RMSE vs Manual Hi-Res GT: {rmse_h:.2f} m")
        print("  Regional RMSE Breakdown (vs Planet Labs Ref):")
        for r_id in range(1, 6):
            r_val = regional_rmse.get(r_id, float('nan'))
            r_str = f"{r_val:.2f} m" if not np.isnan(r_val) else "N/A (No transect intersections)"
            print(f"    - {region_names[r_id]} (R{r_id}): {r_str}")
        
        # Save visualization plots
        plot_out_path = os.path.join(vis_dir, f"{tile_name}_model_predicted_vs_gt_comparison.png")
        plot_predictions_comparison(t_path, pred_lines, planet_union, hires_union, usgs_union, transform, plot_out_path)
        
        # Save binary water/land prediction visualization plot
        water_land_out_path = os.path.join(vis_dir, f"{tile_name}_model_water_land_prediction.png")
        plot_water_land_prediction(t_path, pred_mask_np, pred_lines, water_land_out_path)

if __name__ == "__main__":
    main()
