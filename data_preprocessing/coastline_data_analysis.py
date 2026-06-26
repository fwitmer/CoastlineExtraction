import os
import sys
import glob
import json
import shutil
import logging
import time
import numpy as np
import rasterio as rio
import rasterio.features
import cv2
import skimage.measure
import geopandas as gpd
import shapely
from shapely.geometry import box, LineString, Point, MultiLineString, MultiPoint
from shapely.ops import split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Resolve repository root and sibling directories dynamically
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
new_data_base = os.path.abspath(os.path.join(repo_root, "..", "new_data"))

# Set up logging
log_path = os.path.join(new_data_base, "coastline_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CoastlinePipeline")

# EPSG projections
UTM_ZONE_3N = 'EPSG:32603'

def plot_geometry_on_tile(ax, geom, transform, color, linewidth, label):
    """
    Plots a shapely geometry (LineString or MultiLineString) on a matplotlib axis
    by mapping coordinates to raster pixel space.
    """
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        coords = np.array(geom.coords)
        rows, cols = rio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
        ax.plot(cols, rows, color=color, linewidth=linewidth, label=label)
    elif isinstance(geom, MultiLineString):
        first = True
        for line in geom.geoms:
            coords = np.array(line.coords)
            rows, cols = rio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
            ax.plot(cols, rows, color=color, linewidth=linewidth, label=label if first else None)
            first = False

def calculate_tile_quality(rgbn_tile, udm2_tile, floor_threshold=0.20, otsu_mult=0.95, 
                           additional_shadow_thresh=0.03, additional_haze_thresh=0.14, 
                           max_allowed_cloud=0.15):
    """
    Evaluates a tile's quality using Configuration 7 parameters.
    """
    valid_mask = rgbn_tile[0] > 0
    total_valid_pixels = np.sum(valid_mask)
    if total_valid_pixels == 0:
        return {"status": "REJECT", "reason": "No-data tile", "cloud_fraction": 1.0}

    udm_lhaze = udm2_tile[3] == 1
    udm_hhaze = udm2_tile[4] == 1
    udm_cloud = udm2_tile[5] == 1
    udm_native_cloud = udm_cloud | udm_lhaze | udm_hhaze

    udm_cloud_fraction = np.sum(udm_native_cloud & valid_mask) / total_valid_pixels
    if udm_cloud_fraction > 0.20:
        return {"status": "REJECT", "reason": "High native UDM cloud cover", "cloud_fraction": udm_cloud_fraction}

    brightness = np.mean(rgbn_tile[0:3], axis=0) / 10000.0
    udm_snow = udm2_tile[1] == 1
    otsu_search_pool = valid_mask & ~udm_snow & ~udm_cloud

    if np.sum(otsu_search_pool) > 100:
        try:
            t_otsu = skimage.measure.label(otsu_search_pool) # dummy to ensure package imported
            t_otsu = skimage.filters.threshold_otsu(brightness[otsu_search_pool])
            adjusted_threshold = t_otsu * otsu_mult
            if adjusted_threshold >= floor_threshold:
                otsu_clouds = (brightness > adjusted_threshold) & otsu_search_pool
            else:
                otsu_clouds = np.zeros_like(valid_mask, dtype=bool)
        except Exception:
            otsu_clouds = np.zeros_like(valid_mask, dtype=bool)
    else:
        otsu_clouds = np.zeros_like(valid_mask, dtype=bool)

    if additional_haze_thresh is not None:
        custom_haze = (brightness > additional_haze_thresh) & valid_mask & ~udm_snow & ~udm_cloud & ~otsu_clouds
    else:
        custom_haze = np.zeros_like(valid_mask, dtype=bool)

    final_cloud_mask = udm_cloud | otsu_clouds | custom_haze
    final_cloud_fraction = np.sum(final_cloud_mask & valid_mask) / total_valid_pixels

    decision = "KEEP" if final_cloud_fraction <= max_allowed_cloud else "REJECT"
    reason = "KEEP" if decision == "KEEP" else "Excessive final cloud cover (Otsu-refined)"

    return {"status": decision, "reason": reason, "cloud_fraction": float(final_cloud_fraction)}

def extract_tile_coastline(rgbn_tile, transform):
    """
    Extracts coastline contour from a tile using NDWI and Otsu thresholding, 
    and returns georeferenced LineString geometries.
    """
    green = rgbn_tile[1].astype(np.float32)
    nir = rgbn_tile[3].astype(np.float32)
    denom = green + nir
    
    ndwi = np.zeros_like(green)
    valid_denom = denom > 0
    ndwi[valid_denom] = (green[valid_denom] - nir[valid_denom]) / denom[valid_denom]
    
    ndwi_blurred = cv2.GaussianBlur(ndwi, (9, 9), 0)
    ndwi_8bit = np.floor((ndwi_blurred * 127.0) + 128.0)
    ndwi_8bit = np.clip(ndwi_8bit, 0, 255).astype(np.uint8)
    
    valid_mask = rgbn_tile[0] > 0
    valid_ndwi_pixels = ndwi_8bit[valid_mask]
    
    if len(valid_ndwi_pixels) > 100:
        t_otsu, _ = cv2.threshold(valid_ndwi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        water_mask = (ndwi_8bit >= t_otsu) & valid_mask
    else:
        water_mask = np.zeros_like(valid_mask, dtype=bool)
        
    contours = skimage.measure.find_contours(water_mask.astype(np.float32), 0.5)
    
    lines = []
    for contour in contours:
        # contour is in (row, col) format
        col = contour[:, 1]
        row = contour[:, 0]
        xs, ys = rio.transform.xy(transform, row, col)
        if len(xs) >= 2:
            lines.append(LineString(list(zip(xs, ys))))
            
    return lines, water_mask

def extend_line(line, distance=5000.0):
    """
    Extends both ends of a LineString geometry by a large distance.
    This guarantees that the splitter line completely intersects a tile bounding box.
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return line
        
    p1, p2 = np.array(coords[0]), np.array(coords[1])
    dir_start = p1 - p2
    len_start = np.linalg.norm(dir_start)
    if len_start > 0:
        p_start = p1 + (dir_start / len_start) * distance
    else:
        p_start = p1
        
    p_last2, p_last = np.array(coords[-2]), np.array(coords[-1])
    dir_end = p_last - p_last2
    len_end = np.linalg.norm(dir_end)
    if len_end > 0:
        p_end = p_last + (dir_end / len_end) * distance
    else:
        p_end = p_last
        
    new_coords = [tuple(p_start)] + coords[1:-1] + [tuple(p_end)]
    return LineString(new_coords)

def main():
    logger.info("==================================================")
    logger.info("STARTING DATA ANALYSIS AND CLEANING PIPELINE")
    logger.info("==================================================")
    
    years = ["2021", "2022", "2023", "2024", "2025"]
    
    # 1. Load reference shapefiles and reproject them to UTM Zone 3N dynamically
    p_transects = os.path.join(repo_root, "USGS_Coastlines", "WestChukchi_exposed_STepr_rates", "WestChukchi_exposed_STepr_rates.shp")
    p_hires = os.path.join(repo_root, "ground_truth", "2016_HiRes_Final_Coastline.shp")
    p_planet = os.path.join(repo_root, "..", "existing_data", "DigitizedCoastlines", "PlanetCoastline_gt", "09_09_2016", "9_9_16_PlanetCoastline.shp")
    
    logger.info("Loading reference shapefiles...")
    trans = gpd.read_file(p_transects).to_crs(UTM_ZONE_3N)
    trans_deering = trans[trans['BaselineID'] == 117].sort_values('TransOrder')
    hires = gpd.read_file(p_hires).to_crs(UTM_ZONE_3N)
    planet = gpd.read_file(p_planet).to_crs(UTM_ZONE_3N)
    
    hires_union = hires.union_all() if hasattr(hires, 'union_all') else hires.unary_union
    planet_union = planet.union_all() if hasattr(planet, 'union_all') else planet.unary_union
    
    logger.info(f"Loaded {len(trans_deering)} Deering transect lines.")
    
    # Precompute reference intersection distances along each transect
    ref_distances = {} # TransOrder -> planet distance
    usgs_distances = {} # TransOrder -> usgs distance
    
    for idx, row in trans_deering.iterrows():
        oid = int(row['TransOrder'])
        t_geom = row.geometry
        
        # Intersect with Planet reference coastline
        p_int = t_geom.intersection(planet_union)
        if not p_int.is_empty:
            if isinstance(p_int, Point):
                ref_distances[oid] = t_geom.project(p_int)
            elif isinstance(p_int, MultiPoint):
                # Pick the point closest to the baseline start
                ref_distances[oid] = min([t_geom.project(pt) for pt in p_int.geoms])
            elif hasattr(p_int, 'geoms'):
                pts = [pt for pt in p_int.geoms if isinstance(pt, Point)]
                if pts:
                    ref_distances[oid] = min([t_geom.project(pt) for pt in pts])
                    
        # Intersect with USGS ground truth coastline
        u_int = t_geom.intersection(hires_union)
        if not u_int.is_empty:
            if isinstance(u_int, Point):
                usgs_distances[oid] = t_geom.project(u_int)
            elif isinstance(u_int, MultiPoint):
                usgs_distances[oid] = min([t_geom.project(pt) for pt in u_int.geoms])
            elif hasattr(u_int, 'geoms'):
                pts = [pt for pt in u_int.geoms if isinstance(pt, Point)]
                if pts:
                    usgs_distances[oid] = min([t_geom.project(pt) for pt in pts])

    logger.info(f"Precomputed {len(ref_distances)} Planet reference intersections.")
    logger.info(f"Precomputed {len(usgs_distances)} USGS ground truth intersections.")

    # 2. Filtering bad tiles and grouping files by year-month
    logger.info("Step 1: Filtering bad tiles and grouping by year-month...")
    accepted_tiles = [] # List of dicts with tile details
    
    for year in years:
        tif_dir = os.path.join(new_data_base, year, "tiles", "tif")
        udm_dir = os.path.join(new_data_base, year, "tiles", "udm_masks")
        img_paths = glob.glob(os.path.join(tif_dir, "*.tif"))
        logger.info(f"Scanning {len(img_paths)} raw tiles for Year {year}...")
        
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            parts = filename.split("_")
            if "3B" in parts:
                index_3b = parts.index("3B")
                item_id = "_".join(parts[:index_3b])
            else:
                item_id = "_".join(parts[0:4])
            suffix = parts[-1]
            mask_filename = f"{item_id}_3B_udm2_clip_mask_{suffix}"
            mask_path = os.path.join(udm_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                continue
                
            try:
                with rio.open(img_path) as src:
                    rgbn_tile = src.read()
                    transform = src.transform
                    profile = src.profile
                    bounds = src.bounds
                with rio.open(mask_path) as src:
                    udm2_tile = src.read()
                    
                quality = calculate_tile_quality(rgbn_tile, udm2_tile)
                if quality["status"] == "KEEP":
                    # Parse year and month from filename (e.g., 20210611_...)
                    date_part = parts[0]
                    ym = f"{date_part[:4]}_{date_part[4:6]}"
                    
                    accepted_tiles.append({
                        "year_month": ym,
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "filename": filename,
                        "mask_filename": mask_filename,
                        "transform": transform,
                        "profile": profile,
                        "bounds": bounds,
                        "rgbn": rgbn_tile
                    })
            except Exception as e:
                logger.error(f"Error filtering tile {filename}: {e}")
                
    logger.info(f"Step 1 Complete! Kept {len(accepted_tiles)} accepted tiles out of {len(accepted_tiles)} processed.")

    # 3. Extract coastline contours for all accepted tiles
    logger.info("Extracting coastline contours for all accepted tiles...")
    for tile in accepted_tiles:
        lines, water_mask = extract_tile_coastline(tile["rgbn"], tile["transform"])
        tile["coastline_lines"] = lines
        tile["individual_water_mask"] = water_mask

    # Group accepted tiles by year_month
    ym_groups = {}
    for tile in accepted_tiles:
        ym = tile["year_month"]
        if ym not in ym_groups:
            ym_groups[ym] = []
        ym_groups[ym].append(tile)
        
    logger.info(f"Grouped accepted tiles into {len(ym_groups)} Year-Month groups: {sorted(ym_groups.keys())}")

    # 4. Compute monthly intersections, average distances, and variances per transect
    logger.info("Step 2 & 3: Computing monthly rolling averages and RMSE values...")
    monthly_transect_distances = {} # ym -> {oid -> list of distances}
    monthly_average_distances = {} # ym -> {oid -> mean_distance}
    monthly_variance_distances = {} # ym -> {oid -> variance_distance}
    
    for ym, tiles_in_month in ym_groups.items():
        monthly_transect_distances[ym] = {}
        for tile in tiles_in_month:
            lines = tile["coastline_lines"]
            if not lines:
                continue
            tile_box = box(*tile["bounds"])
            # Filter transects that intersect the tile box
            intersecting_transects = trans_deering[trans_deering.intersects(tile_box)]
            
            # Combine all line segments of the tile coastline into a single MultiLineString
            combined_lines = MultiLineString(lines)
            
            for idx, row in intersecting_transects.iterrows():
                oid = int(row['TransOrder'])
                t_geom = row.geometry
                
                pt_int = combined_lines.intersection(t_geom)
                if not pt_int.is_empty:
                    if isinstance(pt_int, Point):
                        dist = t_geom.project(pt_int)
                        if oid not in monthly_transect_distances[ym]:
                            monthly_transect_distances[ym][oid] = []
                        monthly_transect_distances[ym][oid].append(dist)
                    elif isinstance(pt_int, MultiPoint):
                        dist = min([t_geom.project(pt) for pt in pt_int.geoms])
                        if oid not in monthly_transect_distances[ym]:
                            monthly_transect_distances[ym][oid] = []
                        monthly_transect_distances[ym][oid].append(dist)
                    elif hasattr(pt_int, 'geoms'):
                        pts = [pt for pt in pt_int.geoms if isinstance(pt, Point)]
                        if pts:
                            dist = min([t_geom.project(pt) for pt in pts])
                            if oid not in monthly_transect_distances[ym]:
                                monthly_transect_distances[ym][oid] = []
                            monthly_transect_distances[ym][oid].append(dist)
                            
        # Compute mean and variance for this month's active transects
        monthly_average_distances[ym] = {}
        monthly_variance_distances[ym] = {}
        for oid, dists in monthly_transect_distances[ym].items():
            monthly_average_distances[ym][oid] = float(np.mean(dists))
            if len(dists) > 1:
                monthly_variance_distances[ym][oid] = float(np.var(dists))
            else:
                monthly_variance_distances[ym][oid] = 0.0

    # 5. Compute RMSE against Planet Reference and USGS Ground Truth
    rmse_planet_report = {}
    rmse_usgs_report = {}
    
    for ym in sorted(ym_groups.keys()):
        # Planet Reference RMSE
        errors_planet = []
        for oid, avg_dist in monthly_average_distances[ym].items():
            if oid in ref_distances:
                errors_planet.append(avg_dist - ref_distances[oid])
        if errors_planet:
            rmse_p = np.sqrt(np.mean(np.square(errors_planet)))
            rmse_planet_report[ym] = rmse_p
        else:
            rmse_planet_report[ym] = float('nan')
            
        # USGS Ground Truth RMSE
        errors_usgs = []
        for oid, avg_dist in monthly_average_distances[ym].items():
            if oid in usgs_distances:
                errors_usgs.append(avg_dist - usgs_distances[oid])
        if errors_usgs:
            rmse_u = np.sqrt(np.mean(np.square(errors_usgs)))
            rmse_usgs_report[ym] = rmse_u
        else:
            rmse_usgs_report[ym] = float('nan')
            
        logger.info(f"Year-Month: {ym} | Planet Reference RMSE: {rmse_planet_report[ym]:.2f} m | USGS Ground Truth RMSE: {rmse_usgs_report[ym]:.2f} m")

    # Save RMSE statistics report as JSON
    rmse_stats_path = os.path.join(new_data_base, "monthly_rmse_statistics.json")
    with open(rmse_stats_path, "w") as f:
        json.dump({
            "planet_reference_rmse": rmse_planet_report,
            "usgs_ground_truth_rmse": rmse_usgs_report
        }, f, indent=4)
    logger.info(f"Saved monthly RMSE statistics to: {rmse_stats_path}")

    # 6. Monthly average coastline line construction
    # Connect sorted monthly average intersection points to form a LineString
    logger.info("Constructing monthly average coastline lines...")
    monthly_coastline_lines = {} # ym -> LineString
    for ym in ym_groups.keys():
        sorted_points = []
        avg_dists = monthly_average_distances[ym]
        for idx, row in trans_deering.iterrows():
            oid = int(row['TransOrder'])
            if oid in avg_dists:
                t_geom = row.geometry
                pt_coord = t_geom.interpolate(avg_dists[oid])
                sorted_points.append(pt_coord)
        if len(sorted_points) >= 2:
            monthly_coastline_lines[ym] = LineString(sorted_points)
        else:
            monthly_coastline_lines[ym] = None

    # 7. Step 4 & 5: Select Ground Truth Mask for each tile and prepare dataset
    logger.info("Step 4 & 5: Selecting ground truth masks and preparing dataset...")
    training_dataset_dir = os.path.join(new_data_base, "training_dataset")
    os.makedirs(training_dataset_dir, exist_ok=True)
    
    variance_threshold = 100.0 # 10m standard deviation threshold
    logger.info(f"Using coastal variance threshold: {variance_threshold} m^2")
    
    tiles_using_monthly = 0
    tiles_using_individual = 0
    
    # Track items for visualization PDF
    visualizations_list = []
    
    for ym, tiles_in_month in ym_groups.items():
        avg_line = monthly_coastline_lines[ym]
        
        for tile in tiles_in_month:
            filename = tile["filename"]
            tile_box = box(*tile["bounds"])
            
            # Find transects intersecting this tile
            intersecting_trans = trans_deering[trans_deering.intersects(tile_box)]
            
            # Compute average monthly variance for these transects
            vars_list = []
            for idx, row in intersecting_trans.iterrows():
                oid = int(row['TransOrder'])
                if oid in monthly_variance_distances[ym]:
                    vars_list.append(monthly_variance_distances[ym][oid])
                    
            avg_var = np.mean(vars_list) if vars_list else 0.0
            tile["average_monthly_variance"] = avg_var
            
            use_monthly = (avg_var > variance_threshold) and (avg_line is not None)
            tile["use_monthly_mask"] = use_monthly
            
            if use_monthly:
                # Reconstruct mask from monthly average coastline
                tiles_using_monthly += 1
                try:
                    # Extend the monthly average line to ensure a complete split of the bounding box
                    extended_avg_line = extend_line(avg_line, distance=5000.0)
                    split_geom = split(tile_box, extended_avg_line)
                    
                    reconstructed_mask = np.zeros((512, 512), dtype=np.uint8)
                    indiv_mask = tile["individual_water_mask"]
                    transform = tile["transform"]
                    
                    # If split returned multiple polygons, assign water/land to each
                    if len(split_geom.geoms) >= 2:
                        for poly in split_geom.geoms:
                            # Rasterize polygon to a binary mask (col, row)
                            poly_mask = rasterio.features.rasterize(
                                [poly], out_shape=(512, 512), transform=transform, fill=0, default_value=1
                            )
                            # Find average of water pixels inside this polygon
                            total_poly_pixels = np.sum(poly_mask == 1)
                            if total_poly_pixels > 0:
                                avg_water = np.sum(indiv_mask[poly_mask == 1]) / total_poly_pixels
                                if avg_water > 0.5:
                                    reconstructed_mask[poly_mask == 1] = 1
                    else:
                        # Fall back to individual if split failed
                        reconstructed_mask = indiv_mask.astype(np.uint8)
                        
                    tile["final_mask"] = reconstructed_mask
                    
                except Exception as e:
                    logger.error(f"Error splitting tile {filename} by average coastline: {e}")
                    tile["final_mask"] = tile["individual_water_mask"].astype(np.uint8)
            else:
                tiles_using_individual += 1
                tile["final_mask"] = tile["individual_water_mask"].astype(np.uint8)
                
            # Copy file to training dataset folder with matching names for train_unet.py
            image_dest_path = os.path.join(training_dataset_dir, filename)
            # Match the base_name_concatenated_ndwi_mask_clip_01-of-63.tif format
            base_parts = filename.split("_")
            suffix_idx = filename.find("_clip_")
            base_name_extracted = filename[:suffix_idx]
            suffix_part = filename[suffix_idx + 6:] # strip '_clip_'
            mask_dest_name = f"{base_name_extracted}_concatenated_ndwi_mask_clip_{suffix_part}"
            mask_dest_path = os.path.join(training_dataset_dir, mask_dest_name)
            
            # Save final mask TIFF
            profile_updated = tile["profile"].copy()
            profile_updated.update(count=1, dtype=rio.uint8, nodata=0)
            
            with rio.open(mask_dest_path, 'w', **profile_updated) as dst:
                dst.write(tile["final_mask"], 1)
                
            shutil.copy2(tile["img_path"], image_dest_path)
            
    logger.info(f"Dataset preparation complete! Total training pairs generated: {tiles_using_monthly + tiles_using_individual}")
    logger.info(f"Tiles using Monthly average mask (high variance fallback): {tiles_using_monthly}")
    logger.info(f"Tiles using Individual mask (stable low variance): {tiles_using_individual}")
    logger.info(f"All training image-mask pairs saved in: {training_dataset_dir}")

    # Select exactly 10 diverse coastal tiles located on the coastline regions
    coastal_candidates = []
    for tile in accepted_tiles:
        tile_box = box(*tile["bounds"])
        intersecting_trans = trans_deering[trans_deering.intersects(tile_box)]
        avg_line = monthly_coastline_lines[tile["year_month"]]
        
        # Select tiles that intersect Deering transects (on the shore),
        # have some detected individual coastline, and have monthly average line crossing them
        if len(intersecting_trans) >= 8 and len(tile["coastline_lines"]) > 0 and avg_line is not None:
            if avg_line.intersects(tile_box):
                coastal_candidates.append(tile)
                
    # Balanced selection of monthly reconstructed and individual masks
    monthly_caps = [c for c in coastal_candidates if c["use_monthly_mask"]]
    indiv_caps = [c for c in coastal_candidates if not c["use_monthly_mask"]]
    
    logger.info(f"Found {len(coastal_candidates)} coastal candidate tiles (Monthly mask: {len(monthly_caps)}, Indiv mask: {len(indiv_caps)})")
    
    selected_tiles = []
    num_monthly = min(5, len(monthly_caps))
    num_indiv = 10 - num_monthly
    
    selected_tiles.extend(monthly_caps[:num_monthly])
    selected_tiles.extend(indiv_caps[:num_indiv])
    
    if len(selected_tiles) < 10:
        remaining = [c for c in coastal_candidates if c not in selected_tiles]
        selected_tiles.extend(remaining[:10 - len(selected_tiles)])
        
    visualizations_list = selected_tiles
    logger.info(f"Selected {len(visualizations_list)} representative coastal tiles for final PDF report visualization.")

    # 8. Create Sample Visualizations PDF Report
    logger.info("Generating sample visualizations PDF report...")
    pdf_report_path = os.path.join(new_data_base, "coastline_analysis_report.pdf")
    
    with PdfPages(pdf_report_path) as pdf:
        # Title Page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.7, "Deering, Alaska Coastal Data Analysis & Clean Mask Report", 
                transform=ax.transAxes, ha='center', va='center', fontsize=18, fontweight='bold', color='navy')
        ax.text(0.5, 0.55, "Quality Control Pipeline & Monthly Average Contours vs ground truth comparison", 
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='dimgray')
        ax.text(0.5, 0.45, f"Date: {time.strftime('%Y-%m-%d')}", transform=ax.transAxes, ha='center', va='center', fontsize=10)
        ax.text(0.5, 0.35, f"Total Training Pairs Prepared: {tiles_using_monthly + tiles_using_individual}\n"
                           f"Tiles with High Variance (Fell back to Monthly Avg): {tiles_using_monthly}\n"
                           f"Tiles with Low Variance (Used Individual): {tiles_using_individual}",
                transform=ax.transAxes, ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="whitesmoke", ec="lightgray"))
        pdf.savefig(fig)
        plt.close(fig)
        
        # Plot RMSE Chart Page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        sorted_ym = sorted(rmse_planet_report.keys())
        planet_rmses = [rmse_planet_report[ym] for ym in sorted_ym]
        usgs_rmses = [rmse_usgs_report[ym] for ym in sorted_ym]
        
        ax.plot(sorted_ym, planet_rmses, marker='o', label='RMSE vs Planet Labs Reference Coastline', color='teal', linewidth=2)
        ax.plot(sorted_ym, usgs_rmses, marker='s', label='RMSE vs USGS Ground Truth Coastline', color='crimson', linewidth=2)
        ax.set_title("Monthly Rolling Average Coastline RMSE Analysis", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Year-Month", fontsize=11, labelpad=10)
        ax.set_ylabel("RMSE (meters)", fontsize=11, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Plot 10 Sample Tiles Comparison Pages
        for idx, tile in enumerate(visualizations_list):
            fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
            
            # Load original RGB Bands 1, 2, 3
            rgb = tile["rgbn"][:3]
            # Normalize for display
            rgb_stretched = []
            for b in rgb:
                b_min, b_max = np.percentile(b, [2, 98])
                if b_max > b_min:
                    b_norm = np.clip((b - b_min) / (b_max - b_min) * 255, 0, 255).astype(np.uint8)
                else:
                    b_norm = b.astype(np.uint8)
                rgb_stretched.append(b_norm)
            rgb_stretched = np.stack(rgb_stretched, axis=-1)
            
            # Subplot 1: Original Image RGB
            axs[0, 0].imshow(rgb_stretched)
            axs[0, 0].set_title("Original RGB Satellite Tile", fontsize=10, fontweight='bold')
            axs[0, 0].axis('off')
            
            # Subplot 2: Individual Coastline Contour Overlay
            axs[0, 1].imshow(rgb_stretched)
            for line in tile["coastline_lines"]:
                # Convert coords to array of pixels
                coords = np.array(line.coords)
                # Transform coordinates back to pixel coordinates
                rows, cols = rio.transform.rowcol(tile["transform"], coords[:, 0], coords[:, 1])
                axs[0, 1].plot(cols, rows, color='lime', linewidth=1.5, label='Tile Coastline' if 'Tile Coastline' not in axs[0, 1].get_legend_handles_labels()[1] else '')
            axs[0, 1].set_title("Detected Individual Coastline Overlay", fontsize=10, fontweight='bold')
            axs[0, 1].axis('off')
            
            # Subplot 3: Combined Coastlines Overlay (Monthly Mean, Planet Reference, and USGS Ground Truth)
            axs[1, 0].imshow(rgb_stretched)
            tile_box = box(*tile["bounds"])
            
            # 1. USGS Ground Truth
            try:
                u_clipped = hires_union.intersection(tile_box)
                plot_geometry_on_tile(axs[1, 0], u_clipped, tile["transform"], color='red', linewidth=2, label='USGS Ground Truth')
            except Exception as e:
                logger.warning(f"Error plotting USGS coastline: {e}")
                
            # 2. Planet Reference Ground Truth
            try:
                p_clipped = planet_union.intersection(tile_box)
                plot_geometry_on_tile(axs[1, 0], p_clipped, tile["transform"], color='orange', linewidth=2, label='Planet Reference')
            except Exception as e:
                logger.warning(f"Error plotting Planet reference: {e}")
                
            # 3. Monthly Mean Coastline (single line representing monthly average)
            avg_line = monthly_coastline_lines[tile["year_month"]]
            if avg_line:
                try:
                    avg_clipped = avg_line.intersection(tile_box)
                    plot_geometry_on_tile(axs[1, 0], avg_clipped, tile["transform"], color='cyan', linewidth=2, label='Monthly Mean')
                except Exception as e:
                    logger.warning(f"Error plotting Monthly Mean coastline: {e}")
                    
            axs[1, 0].set_title("Coastlines Overlay (Mean vs GTs)", fontsize=10, fontweight='bold')
            axs[1, 0].legend(fontsize=8, loc='upper right')
            axs[1, 0].axis('off')
            
            # Subplot 4: Reconstructed final clean mask
            axs[1, 1].imshow(tile["final_mask"], cmap='Blues')
            axs[1, 1].set_title(f"Clean Selected Mask (Variance: {tile['average_monthly_variance']:.1f} m^2)", fontsize=10, fontweight='bold')
            # Annotation of what type of mask was selected
            mask_type_str = "Monthly Avg (High Var)" if tile["use_monthly_mask"] else "Individual (Low Var)"
            axs[1, 1].text(0.05, 0.05, mask_type_str, transform=axs[1, 1].transAxes, color='white', 
                           fontweight='bold', bbox=dict(boxstyle="square,pad=0.2", fc="darkorange", ec="none"))
            axs[1, 1].axis('off')
            
            fig.suptitle(f"Sample {idx+1}: {tile['filename']} ({tile['year_month']})", fontsize=12, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
    logger.info(f"Sample visualizations report generated at: {pdf_report_path}")
    logger.info("==================================================")
    logger.info("DATA ANALYSIS AND CLEANING PIPELINE COMPLETED!")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
