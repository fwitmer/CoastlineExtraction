"""
Module: validation_framework.py

Description: Comprehensive validation framework for coastline extraction
             accuracy assessment and improvement quantification.

             Features:
             - Transect-based RMSE calculation
             - IoU/Dice metrics for segmentation
             - Per-region accuracy analysis (cliff vs. non-cliff)
             - Comparison against ground truth
             - Statistical significance testing
             - Visualization of results

Author: GSoC 2026 Team
Date: 2026-03-28

Issue: #105 - Validation Framework and Accuracy Metrics
"""

import os
import numpy as np
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points
import cv2
from scipy import stats
from matplotlib import pyplot as plt
import json
from datetime import datetime

# Local imports
from load_config import load_config


# =============================================================================
# Configuration
# =============================================================================

VALIDATION_CONFIG = {
    'transect_spacing': 50,          # Meters between transect points
    'buffer_distance': 100,          # Meters for analysis buffer
    'cliff_slope_threshold': 30,     # Degrees for cliff classification
    'rmse_outlier_threshold': 3.0,   # Standard deviations for outlier removal
}

# Region definitions (Deering study area)
REGION_DEFINITIONS = {
    'northern_cliff': {'transect_range': (17394, 17443)},
    'east_cliff': {'transect_range': (None, 17337)},
    'western_shore': {'transect_range': (17337, 17394)},
    'southern_shore': {'transect_range': (17443, None)},
}


# =============================================================================
# Ground Truth Loading
# =============================================================================

def load_ground_truth(ground_truth_path):
    """
    Load ground truth coastline data.

    Args:
        ground_truth_path (str): Path to ground truth shapefile or raster

    Returns:
        dict: Ground truth data
    """
    print(f"Loading ground truth: {ground_truth_path}")

    if ground_truth_path.endswith('.shp'):
        # Vector ground truth
        gdf = gpd.read_file(ground_truth_path)

        # Ensure proper CRS (UTM Zone 3N for Alaska)
        if gdf.crs is not None and gdf.crs.to_epsg() != 32603:
            gdf = gdf.to_crs(epsg=32603)

        return {
            'type': 'vector',
            'geometry': gdf.geometry,
            'gdf': gdf
        }

    elif ground_truth_path.endswith('.tif'):
        # Raster ground truth
        with rio.open(ground_truth_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs

        return {
            'type': 'raster',
            'data': data,
            'transform': transform,
            'crs': crs
        }

    else:
        raise ValueError(f"Unsupported ground truth format: {ground_truth_path}")


def load_predicted_coastline(prediction_path):
    """
    Load predicted coastline from model output.
    """
    print(f"Loading prediction: {prediction_path}")

    if prediction_path.endswith('.shp'):
        gdf = gpd.read_file(prediction_path)
        if gdf.crs is not None and gdf.crs.to_epsg() != 32603:
            gdf = gdf.to_crs(epsg=32603)
        return {'type': 'vector', 'geometry': gdf.geometry, 'gdf': gdf}

    elif prediction_path.endswith('.tif'):
        with rio.open(prediction_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
        return {'type': 'raster', 'data': data, 'transform': transform, 'crs': crs}

    else:
        raise ValueError(f"Unsupported prediction format: {prediction_path}")


# =============================================================================
# Transect-Based RMSE
# =============================================================================

def calculate_transect_rmse(ground_truth, prediction, transects_path=None):
    """
    Calculate RMSE using transect-based method.

    This is the primary accuracy metric used in the project.

    Args:
        ground_truth (dict): Ground truth data
        prediction (dict): Prediction data
        transects_path (str): Path to transect lines shapefile

    Returns:
        dict: RMSE results
    """
    print("\n" + "=" * 60)
    print("TRANSECT-BASED RMSE CALCULATION")
    print("=" * 60)

    if transects_path is None:
        # Use default transects from config
        config = load_config()
        transects_path = config.get('transects_path', 'USGS_Coastlines/Deering_transects.shp')

    if not os.path.exists(transects_path):
        print(f"Transects not found: {transects_path}")
        return calculate_raster_based_metrics(ground_truth, prediction)

    # Load transects
    transects_gdf = gpd.read_file(transects_path)
    if transects_gdf.crs is not None and transects_gdf.crs.to_epsg() != 32603:
        transects_gdf = transects_gdf.to_crs(epsg=32603)

    print(f"Loaded {len(transects_gdf)} transects")

    # Get geometries
    if ground_truth['type'] == 'vector':
        gt_geometry = ground_truth['geometry'].unary_union
    else:
        gt_geometry = raster_to_vector(ground_truth)

    if prediction['type'] == 'vector':
        pred_geometry = prediction['geometry'].unary_union
    else:
        pred_geometry = raster_to_vector(prediction)

    # Calculate intersection points for each transect
    gt_distances = []
    pred_distances = []
    errors = []
    transect_results = []

    for idx, row in transects_gdf.iterrows():
        transect = row['geometry']

        # Find intersection with ground truth
        gt_intersection = transect.intersection(gt_geometry)
        pred_intersection = transect.intersection(pred_geometry)

        if gt_intersection.is_empty or pred_intersection.is_empty:
            continue

        # Get distance along transect (from land side)
        if isinstance(gt_intersection, Point):
            gt_dist = transect.project(gt_intersection)
        elif isinstance(gt_intersection, LineString):
            gt_dist = transect.project(gt_intersection.centroid)
        else:
            gt_dist = 0

        if isinstance(pred_intersection, Point):
            pred_dist = transect.project(pred_intersection)
        elif isinstance(pred_intersection, LineString):
            pred_dist = transect.project(pred_intersection.centroid)
        else:
            pred_dist = 0

        error = abs(pred_dist - gt_dist)

        gt_distances.append(gt_dist)
        pred_distances.append(pred_dist)
        errors.append(error)

        transect_results.append({
            'transect_id': idx,
            'gt_distance': gt_dist,
            'pred_distance': pred_dist,
            'error': error
        })

    # Calculate statistics
    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)

    # Remove outliers
    z_scores = np.abs((errors - mae) / (std_error + 1e-10))
    non_outlier_mask = z_scores < VALIDATION_CONFIG['rmse_outlier_threshold']
    rmse_filtered = np.sqrt(np.mean(errors[non_outlier_mask] ** 2))

    print(f"\nTransect Results ({len(errors)} valid transects):")
    print(f"  RMSE: {rmse:.2f} m")
    print(f"  RMSE (filtered): {rmse_filtered:.2f} m")
    print(f"  MAE: {mae:.2f} m")
    print(f"  Std Dev: {std_error:.2f} m")
    print(f"  Max Error: {max_error:.2f} m")
    print(f"  Min Error: {min_error:.2f} m")

    return {
        'rmse': float(rmse),
        'rmse_filtered': float(rmse_filtered),
        'mae': float(mae),
        'std_error': float(std_error),
        'max_error': float(max_error),
        'min_error': float(min_error),
        'n_transects': len(errors),
        'transect_details': transect_results
    }


def raster_to_vector(raster_data):
    """
    Convert raster coastline to vector.
    """
    from skimage import measure

    data = raster_data['data']
    transform = raster_data['transform']

    # Find contours
    contours = measure.find_contours(data, 0.5)

    # Convert to world coordinates
    lines = []
    for contour in contours:
        coords = []
        for y, x in contour:
            world_x, world_y = rio.transform.xy(transform, int(y), int(x))
            coords.append((world_x, world_y))
        if len(coords) > 2:
            lines.append(LineString(coords))

    from shapely.geometry import MultiLineString
    return MultiLineString(lines) if lines else None


# =============================================================================
# Raster-Based Metrics
# =============================================================================

def calculate_raster_based_metrics(ground_truth, prediction):
    """
    Calculate pixel-based accuracy metrics.

    Args:
        ground_truth (dict): Ground truth raster
        prediction (dict): Prediction raster

    Returns:
        dict: Accuracy metrics
    """
    print("\n" + "=" * 60)
    print("RASTER-BASED ACCURACY METRICS")
    print("=" * 60)

    # Ensure same extent and resolution
    gt_data = ground_truth['data']
    pred_data = prediction['data']

    # Resize prediction to match ground truth if needed
    if gt_data.shape != pred_data.shape:
        pred_data = cv2.resize(pred_data, (gt_data.shape[1], gt_data.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # Binarize
    gt_binary = (gt_data > 0.5).astype(np.uint8)
    pred_binary = (pred_data > 0.5).astype(np.uint8)

    # Confusion matrix
    tp = np.sum((gt_binary == 1) & (pred_binary == 1))
    tn = np.sum((gt_binary == 0) & (pred_binary == 0))
    fp = np.sum((gt_binary == 0) & (pred_binary == 1))
    fn = np.sum((gt_binary == 1) & (pred_binary == 0))

    # Metrics
    iou = tp / (tp + fp + fn + 1e-10)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)

    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TN: {tn:,}")

    print(f"\nAccuracy Metrics:")
    print(f"  IoU: {iou:.4f}")
    print(f"  Dice: {dice:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Overall Accuracy: {accuracy:.4f}")

    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    }


# =============================================================================
# Region-Specific Analysis
# =============================================================================

def analyze_by_region(ground_truth, prediction, transects_path=None, slope_data=None):
    """
    Analyze accuracy by coastal region type (cliff vs. non-cliff).

    Args:
        ground_truth (dict): Ground truth data
        prediction (dict): Prediction data
        transects_path (str): Path to transect shapefile
        slope_data (np.ndarray): Slope array for cliff identification

    Returns:
        dict: Per-region accuracy results
    """
    print("\n" + "=" * 60)
    print("REGION-SPECIFIC ANALYSIS")
    print("=" * 60)

    results = {}

    # Load transects if provided
    if transects_path and os.path.exists(transects_path):
        transects_gdf = gpd.read_file(transects_path)
        if transects_gdf.crs is not None and transects_gdf.crs.to_epsg() != 32603:
            transects_gdf = transects_gdf.to_crs(epsg=32603)
    else:
        transects_gdf = None

    # Analyze each defined region
    for region_name, region_config in REGION_DEFINITIONS.items():
        print(f"\nAnalyzing region: {region_name}")

        transect_range = region_config.get('transect_range')

        if transects_gdf is not None and transect_range:
            # Filter transects by ID range
            start_id, end_id = transect_range
            filtered = transects_gdf.copy()

            if start_id is not None:
                filtered = filtered[filtered.index >= start_id]
            if end_id is not None:
                filtered = filtered[filtered.index <= end_id]

            if len(filtered) == 0:
                print(f"  No transects in range for {region_name}")
                continue

            # Calculate RMSE for this region
            region_gt = ground_truth
            region_pred = prediction

            region_metrics = calculate_transect_rmse(region_gt, region_pred,
                                                     transects_path=None)
            region_metrics['transects_used'] = len(filtered)
        else:
            # Use raster-based metrics
            region_metrics = calculate_raster_based_metrics(ground_truth, prediction)

        # Classify as cliff or non-cliff
        is_cliff_region = 'cliff' in region_name.lower()
        region_metrics['is_cliff_region'] = is_cliff_region

        results[region_name] = region_metrics
        print(f"  RMSE: {region_metrics.get('rmse', 'N/A')}")
        print(f"  IoU: {region_metrics.get('iou', 'N/A')}")

    # Summary comparison
    cliff_regions = [r for r, m in results.items() if m.get('is_cliff_region', False)]
    non_cliff_regions = [r for r, m in results.items() if not m.get('is_cliff_region', False)]

    print(f"\n" + "=" * 60)
    print("CLIFF VS. NON-CLIFF COMPARISON")
    print("=" * 60)

    if cliff_regions and non_cliff_regions:
        cliff_rmse = np.mean([results[r]['rmse'] for r in cliff_regions if 'rmse' in results[r]])
        non_cliff_rmse = np.mean([results[r]['rmse'] for r in non_cliff_regions if 'rmse' in results[r]])

        print(f"  Cliff regions RMSE: {cliff_rmse:.2f} m")
        print(f"  Non-cliff regions RMSE: {non_cliff_rmse:.2f} m")
        print(f"  Difference: {abs(cliff_rmse - non_cliff_rmse):.2f} m")

    return results


# =============================================================================
# Method Comparison
# =============================================================================

def compare_methods(methods_results, output_dir='validation_results'):
    """
    Compare multiple extraction methods.

    Args:
        methods_results (dict): Results from different methods
        output_dir (str): Output directory

    Returns:
        dict: Comparison summary
    """
    print("\n" + "=" * 60)
    print("METHOD COMPARISON")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    comparison = {
        'methods': {},
        'ranking': {},
        'statistical_tests': {}
    }

    # Collect metrics
    for method_name, results in methods_results.items():
        comparison['methods'][method_name] = {
            'rmse': results.get('rmse', results.get('rmse_filtered')),
            'iou': results.get('iou'),
            'dice': results.get('dice'),
            'f1': results.get('f1_score'),
            'n_transects': results.get('n_transects')
        }

    # Rank methods by RMSE (lower is better)
    methods_by_rmse = sorted(
        comparison['methods'].items(),
        key=lambda x: x[1]['rmse'] if x[1]['rmse'] is not None else float('inf')
    )

    comparison['ranking']['by_rmse'] = [m[0] for m in methods_by_rmse]

    # Statistical significance (paired t-test if we have transect details)
    method_names = list(methods_results.keys())
    if len(method_names) >= 2:
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                details1 = methods_results[method1].get('transect_details', [])
                details2 = methods_results[method2].get('transect_details', [])

                if details1 and details2:
                    errors1 = [d['error'] for d in details1]
                    errors2 = [d['error'] for d in details2]

                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(errors1, errors2)

                    comparison['statistical_tests'][f'{method1}_vs_{method2}'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }

    # Save results
    results_path = os.path.join(output_dir, 'method_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Create visualization
    plot_method_comparison(comparison, output_dir)

    print(f"\nMethod Ranking (by RMSE):")
    for i, method in enumerate(comparison['ranking']['by_rmse']):
        rmse = comparison['methods'][method]['rmse']
        print(f"  {i+1}. {method}: {rmse:.2f} m")

    return comparison


def plot_method_comparison(comparison, output_dir):
    """
    Create visualization of method comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE comparison
    methods = list(comparison['methods'].keys())
    rmse_values = [comparison['methods'][m]['rmse'] for m in methods]

    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = axes[0].bar(methods, rmse_values, color=colors)
    axes[0].set_ylabel('RMSE (meters)')
    axes[0].set_xlabel('Method')
    axes[0].set_title('RMSE Comparison (lower is better)')
    axes[0].tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, val in zip(bars, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # IoU comparison (if available)
    iou_values = [comparison['methods'][m].get('iou') for m in methods]
    if any(v is not None for v in iou_values):
        axes[1].bar(methods, [v if v else 0 for v in iou_values], color=colors)
        axes[1].set_ylabel('IoU')
        axes[1].set_xlabel('Method')
        axes[1].set_title('IoU Comparison (higher is better)')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot: {os.path.join(output_dir, 'method_comparison.png')}")


# =============================================================================
# Improvement Quantification
# =============================================================================

def quantify_improvement(baseline_results, improved_results, improvement_type):
    """
    Quantify improvement from a specific enhancement.

    Args:
        baseline_results (dict): Results from baseline method
        improved_results (dict): Results from improved method
        improvement_type (str): Type of improvement (e.g., "UDM masking", "DEM integration")

    Returns:
        dict: Improvement metrics
    """
    print(f"\n" + "=" * 60)
    print(f"IMPROVEMENT QUANTIFICATION: {improvement_type}")
    print("=" * 60)

    baseline_rmse = baseline_results.get('rmse', baseline_results.get('rmse_filtered'))
    improved_rmse = improved_results.get('rmse', improved_results.get('rmse_filtered'))

    baseline_iou = baseline_results.get('iou')
    improved_iou = improved_results.get('iou')

    improvement = {
        'type': improvement_type,
        'rmse': {
            'baseline': baseline_rmse,
            'improved': improved_rmse,
            'absolute_change': baseline_rmse - improved_rmse if baseline_rmse and improved_rmse else None,
            'percent_change': ((baseline_rmse - improved_rmse) / baseline_rmse * 100) if baseline_rmse and improved_rmse else None
        },
        'iou': {
            'baseline': baseline_iou,
            'improved': improved_iou,
            'absolute_change': improved_iou - baseline_iou if baseline_iou and improved_iou else None,
            'percent_change': ((improved_iou - baseline_iou) / baseline_iou * 100) if baseline_iou and improved_iou else None
        }
    }

    print(f"\nRMSE:")
    print(f"  Baseline: {baseline_rmse:.2f} m" if baseline_rmse else "  Baseline: N/A")
    print(f"  Improved: {improved_rmse:.2f} m" if improved_rmse else "  Improved: N/A")
    if improvement['rmse']['absolute_change']:
        print(f"  Improvement: {improvement['rmse']['absolute_change']:.2f} m "
              f"({improvement['rmse']['percent_change']:.1f}%)")

    print(f"\nIoU:")
    print(f"  Baseline: {baseline_iou:.4f}" if baseline_iou else "  Baseline: N/A")
    print(f"  Improved: {improved_iou:.4f}" if improved_iou else "  Improved: N/A")
    if improvement['iou']['absolute_change']:
        print(f"  Improvement: {improvement['iou']['absolute_change']:.4f} "
              f"({improvement['iou']['percent_change']:.1f}%)")

    return improvement


# =============================================================================
# Main Validation Pipeline
# =============================================================================

def run_full_validation(ground_truth_path, predictions, output_dir='validation_results'):
    """
    Run complete validation pipeline.

    Args:
        ground_truth_path (str): Path to ground truth data
        predictions (dict): Dictionary of predictions {method_name: path}
        output_dir (str): Output directory

    Returns:
        dict: Complete validation results
    """
    print("=" * 70)
    print("FULL VALIDATION PIPELINE")
    print("=" * 70)
    print(f"Ground truth: {ground_truth_path}")
    print(f"Predictions: {list(predictions.keys())}")

    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)

    # Process each prediction
    all_results = {}

    for method_name, pred_path in predictions.items():
        print(f"\n{'=' * 70}")
        print(f"VALIDATING: {method_name}")
        print(f"{'=' * 70}")

        prediction = load_predicted_coastline(pred_path)

        # Calculate metrics
        transect_results = calculate_transect_rmse(ground_truth, prediction)
        raster_results = calculate_raster_based_metrics(ground_truth, prediction)

        # Combine results
        all_results[method_name] = {
            **transect_results,
            **raster_results
        }

    # Compare methods
    comparison = compare_methods(all_results, output_dir)

    # Save full results
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'ground_truth': ground_truth_path,
        'predictions': predictions,
        'per_method': all_results,
        'comparison': comparison
    }

    results_path = os.path.join(output_dir, 'full_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nValidation complete!")
    print(f"Results saved: {results_path}")

    return full_results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    config = load_config()

    print("Validation Framework")
    print("=" * 60)

    # Example validation run
    ground_truth_path = config.get('ground_truth_folder', 'ground_truth') + "/2016_HiRes_Final_Coastline_UTM3N.tif"

    # Collect predictions from different methods
    predictions = {
        'NDWI_Global': 'result_udm_masking/ndwi_global.tif',
        'NDWI_Sliding_Window': 'result_ndwi_udm/ndwi_concatenated.tif',
        'NDWI_UDM_Masked': 'result_ndwi_udm/ndwi_udm_masked.tif',
    }

    # Filter to existing files
    predictions = {k: v for k, v in predictions.items() if os.path.exists(v)}

    if predictions and os.path.exists(ground_truth_path):
        results = run_full_validation(ground_truth_path, predictions)
    else:
        print("Validation data not found. Update paths in the code above.")
