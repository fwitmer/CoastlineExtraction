# Automated Coastline Extraction for Erosion Modeling in Alaska

The primary goal of this project is to enhance the accuracy of coastline extraction, particularly for erosion modeling in Deering, Alaska, using high-resolution Planet imagery with a 3-meter resolution. The project focuses on creating reliable ground truth data and labels that will be used to train the [DeepWaterMap algorithm](https://github.com/isikdogan/deepwatermap), a deep convolutional neural network designed to segment surface water on multispectral imagery. Originally trained on 30-meter resolution Landsat data, DeepWaterMap will be adapted to work with higher-resolution data in this project.

One of the key challenges in coastline extraction is the application of the Normalized Difference Water Index (NDWI), a widely used remote sensing index for identifying water bodies. However, using a single threshold across an entire image often results in suboptimal accuracy. To address this, I implemented a sliding window approach combined with Otsu thresholding, which dynamically adjusts thresholds over localized regions of the image. This method has shown promising improvements in accuracy.

The newly generated labeled data, derived from this approach, will be used to retrain the [DeepWaterMap algorithm](https://github.com/isikdogan/deepwatermap), replacing the original Global Surface Water data. This project aims to produce a more accurate and reliable tool for coastline detection, which is crucial for monitoring and mitigating coastal erosion in vulnerable areas like Alaska.

## Recent Improvements

This fork includes a critical bug fix that restores the core functionality of the satellite data download pipeline:

### Critical Bug Fixed

In the [DeeringAutoDownloadCode.py](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/DeeringAutoDownloadCode.py) file, the [validate_and_compare_dates](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/DeeringAutoDownloadCode.py#L79-L122) function had a critical bug where it was trying to use undefined variables `start_date_input` and `end_date_input` instead of the function parameters `start_date` and `end_date`. This caused a `NameError` that would crash the entire satellite data download pipeline.

### Solution

The bug was fixed by correcting the variable names in lines 92-93 of the function:

```python
# Before (buggy):
start_date_obj = datetime.strptime(start_date_input, '%Y-%m-%d')  # ❌ Undefined variable
end_date_obj = datetime.strptime(end_date_input, '%Y-%m-%d')      # ❌ Undefined variable

# After (fixed):
start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')        # ✅ Correct parameter
end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')            # ✅ Correct parameter
```

## Repository Contents

- **[DeeringAutoDownloadCode.py](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/DeeringAutoDownloadCode.py)** - Code for automatically downloading Planet satellite imagery with the bug fix applied
- **[rastertools.py](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/rastertools.py)** - Core library with functions for raster processing, including NDWI calculation and classification
- **[batchprocess.py](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/batchprocess.py)** - Handles batch processing of satellite imagery files
- **[data_preprocessing.py](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/data_preprocessing.py)** - Data preprocessing utilities
- **[BUG_FIX_REPORT.md](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/BUG_FIX_REPORT.md)** - Comprehensive report documenting the bug and fix
- **[requirements.txt](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/requirements.txt)** - Dependency requirements for the project

## How to Use

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your Planet API key in [DeeringAutoDownloadCode.py](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/DeeringAutoDownloadCode.py)

3. Run the download script:
   ```bash
   python DeeringAutoDownloadCode.py
   ```

## Testing the Fix

To verify that the bug fix works correctly:

1. **Simple Test** (no dependencies required):
   ```bash
   python simple_date_test.py
   ```

2. **Bug Demonstration** (shows the original error):
   ```bash
   python bug_demo.py
   ```

## Impact of the Fix

- Restores core functionality of the satellite data download pipeline
- Enables researchers to download Planet satellite imagery for coastal erosion studies
- Makes the tool actually usable for its intended purpose
- Improves reliability and user experience

## Additional Improvements

The fix also includes:
- Better variable naming to avoid conflicts
- Comprehensive test coverage
- Detailed documentation
- Dependency management setup

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](file:///c%3A/GSOC/Automated%20coastline%20extraction%20for%20erosion%20modeling%20in%20Alaska/CoastlineExtraction/LICENSE) file for details.