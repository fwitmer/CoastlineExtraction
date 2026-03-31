# Automated Coastline Extraction for Erosion Modeling in Alaska

Work of - Janvi Singh

**Google Summer of Code (GSoC) 2026 Project**

A comprehensive deep learning pipeline for extracting accurate coastlines from high-resolution PlanetLabs satellite imagery, specifically designed for coastal erosion monitoring in Arctic Alaska communities.

## Overview

The rapidly warming Arctic is leading to increased rates of coastal erosion, placing hundreds of Alaska communities at the frontline of climate change. This project provides an automated solution for extracting vectorized coastlines from 3-meter resolution PlanetLabs imagery to support erosion modeling and forecasting.

## What has been implemented in this workspace
1. Unified pipeline entrypoint
   - `coastline_pipeline.py` orchestrates steps: UDM masking, NDWI, DEM integration, quality flags, validation.
2. UDM/QA60 cloud/shadow/snow masking
   - `data_preprocessing/udm_masking.py` supports PlanetLabs UDM2 bit flags and Sentinel-2 QA60.
3. Adaptive NDWI pipeline with windowing + majority voting
   - `data_preprocessing/ndwi_with_udm.py` does local thresholding, mask application, and subscene vector output.
4. DEM/slope/cliff-aware terrain integration
   - `data_preprocessing/dem_integration.py` computes slope/aspect/TRI and cliff masks, influences classification.
5. Shadow/artifact / quality flags
   - `data_preprocessing/shadow_artifact_detection.py` generates 8-bit quality flags, filtering.
6. Data expansion pipeline
   - `data_preprocessing/data_expansion.py` supports 2017-2026 ingestion, seasonal filtering, incremental catalog.
7. DeepWaterMap training and inference
   - `training_pipeline/deepwatermap_train.py`, `train_unet.py`, `predict.py` support U-Net training with Planet imagery.
8. Validation framework
   - `validation_framework.py` with transect RMSE, comparison to ground truth coastlines.

## Status vs. “Potential areas of improvement”
| Area | Status | Notes / Action required |
|---|---|---|
| Improve training data with PlanetLabs UDM | in progress | UDM masking is implemented, verify dataset-level integration for training label creation in `deepwatermap_train.py` and `data_expansion.py`.
| Data expansion beyond 2017-2019 | ✓ done | `data_expansion.py` has 2020-2026 config and pipeline.
| Improved cliff area segmentation | ✓ done | `dem_integration.py` handles cliff detection (slope >30°, elevation-stratified thresholding).
| Handling shadows/buildings/artifacts | ✓ done | `shadow_artifact_detection.py` intended for this; check empirical results and refine thresholds.
| SWIR + elevation integration | partially done | elevation/DEM done; SWIR support likely not yet explicit (only RGBN). Add SWIR path if available.


### Key Features

- **UDM-Based Cloud Masking**: Integrated QA60/UDM2 quality band processing for reliable cloud and shadow removal
- **Sliding Window NDWI**: Localized Otsu thresholding with majority voting for adaptive water detection
- **DEM Integration**: Terrain analysis for improved cliff/steep slope segmentation
- **Quality Flag System**: 8-bit comprehensive pixel-level quality assessment
- **DeepWaterMap Training**: U-Net based deep learning model adapted for 4-band PlanetLabs imagery
- **Data Expansion Pipeline**: Automated ingestion of imagery from 2016-2026 with seasonal filtering
- **Validation Framework**: Transect-based RMSE and raster-based accuracy metrics

### Project Links

- **Source Code**: https://github.com/fwitmer/CoastlineExtraction
- **My Fork**:https://github.com/janvis11/CoastlineExtraction
- **Discussion Forum**: https://github.com/fwitmer/CoastlineExtraction/discussions
- **Mentors**: Frank Witmer (fwitmer@alaska.edu), Ritika Kumari (rkjane333@gmail.com)

---
