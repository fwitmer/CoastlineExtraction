# Automated Coastline Extraction for Erosion Modeling in Alaska

**Google Summer of Code (GSoC) 2026 Project**

A comprehensive deep learning pipeline for extracting accurate coastlines from high-resolution PlanetLabs satellite imagery, specifically designed for coastal erosion monitoring in Arctic Alaska communities.

## Overview

The rapidly warming Arctic is leading to increased rates of coastal erosion, placing hundreds of Alaska communities at the frontline of climate change. This project provides an automated solution for extracting vectorized coastlines from 3-meter resolution PlanetLabs imagery to support erosion modeling and forecasting.

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
