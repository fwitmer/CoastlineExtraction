 ### Automated Coastline Extraction for Erosion Modeling in Alaska

The primary goal of this project is to enhance the accuracy of coastline extraction, particularly for erosion modeling in Deering, Alaska, using high-resolution Planet imagery with a 3-meter resolution. The project focuses on creating reliable ground truth data and labels that will be used to train the [DeepWaterMap algorithm](https://github.com/isikdogan/deepwatermap), a deep convolutional neural network designed to segment surface water on multispectral imagery. Originally trained on 30-meter resolution Landsat data, DeepWaterMap will be adapted to work with higher-resolution data in this project.

One of the key challenges in coastline extraction is the application of the Normalized Difference Water Index (NDWI), a widely used remote sensing index for identifying water bodies. However, using a single threshold across an entire image often results in suboptimal accuracy. To address this, I implemented a sliding window approach combined with Otsu thresholding, which dynamically adjusts thresholds over localized regions of the image. This method has shown promising improvements in accuracy.

The newly generated labeled data, derived from this approach, will be used to retrain the [DeepWaterMap algorithm](https://github.com/isikdogan/deepwatermap), replacing the original Global Surface Water data. This project aims to produce a more accurate and reliable tool for coastline detection, which is crucial for monitoring and mitigating coastal erosion in vulnerable areas like Alaska.

## Installation

### Prerequisites

Before installing this project, ensure you have the following requirements:

- **Python**  
  - **Project version**: Tested and developed with **Python 3.13.5**  
  - **Conda environment**: Recommended to use **Python 3.10** for best compatibility with dependencies  

- **Miniconda** (for managing conda environments)

- **Git** (for cloning the repository)

- **GDAL** (install via `conda-forge` for easier setup)

- **Rasterio 1.4.3+** (for geospatial data processing)

---

### Clone the Repository

Clone the project using the default `master` branch:

```bash
git clone https://github.com/fwitmer/CoastlineExtraction.git
cd CoastlineExtraction
```

> ⚠️ Note: The repository currently contains only the `master` branch. Earlier versions of this README referenced a non-existent `dev` branch and an incorrect repository URL (`coastline-extraction.git`), which caused cloning to fail for new users.

---

### Environment Setup

1. **Create a virtual environment**:

```bash
python -m venv coastline_env
source coastline_env/bin/activate  # On Windows: coastline_env\Scripts\activate
```

2. **Install required dependencies**:

```bash
# Core deep learning libraries
pip install torch torchvision

# Geospatial data processing
pip install rasterio gdal

# Data manipulation and visualization
pip install numpy pandas matplotlib

# Image processing
pip install scikit-image opencv-python

# Utilities
pip install tqdm pillow

# Additional dependencies for data preprocessing
pip install shapely fiona geopandas
```

---

## Configuration

This project uses a centralized configuration system to manage file paths and parameters.
Configuration is handled through `config_template.json` and the `load_config.py` module.

### Setting Up Configuration

1. **Copy the template**:

```bash
cp config_template.json config.json
```

2. **Edit the configuration**: Open `config.json` and modify the paths according to your setup:

```json
{
  "data_dir": "data",
  "image_folder": "sample_data/PlanetLabs",
  "raw_data_folder": "raw_data",
  "shapefile_folder": "USGS_Coastlines",
  "ground_truth_folder": "ground_truth",
  "processed_data_folder": "processed_data",
  "training": {
    "model_save_path": "training_pipeline/unet_model.pth",
    "batch_size": 8,
    "epochs": 30,
    "learning_rate": 1e-4,
    "image_size": [256, 256],
    "train_split": 0.8,
    "device": "auto"
  }
}
```

### Using the Configuration System

The `load_config.py` module provides convenient functions to access your data files:

```python
from load_config import load_config, get_image_path, get_shapefile_path

# Load configuration
config = load_config()

# Get specific file paths
image_path = get_image_path(config, 0)        # First image file
shapefile_path = get_shapefile_path(config, 0) # First shapefile
```
---


## Contributing

### Contribution Workflow

This project currently uses the `master` branch as the primary development branch.

When contributing:

1. **Fork the repository on GitHub**

2. **Clone your fork**:

```bash
git clone https://github.com/your-username/CoastlineExtraction.git
cd CoastlineExtraction
```

3. **Create a feature branch from `master`**:

```bash
git checkout -b feature/your-feature-name
```

4. **Make your changes and commit them**:

```bash
git add .
git commit -m "Add your feature description"
```

5. **Push to your fork**:

```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request** targeting the `dev` branch (not `main`)
