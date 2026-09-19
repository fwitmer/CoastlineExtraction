import subprocess
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PYTHON_EXE = sys.executable

def run_script_help(script_path):
    cmd = [PYTHON_EXE, script_path, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_batch_align_cli_help():
    res = run_script_help("data_preprocessing/batch_align.py")
    assert res.returncode == 0
    assert "--target-srs" in res.stdout
    assert "--pixel-size" in res.stdout
    assert "--extent" in res.stdout
    assert "--raw-dir" in res.stdout

def test_create_masks_cli_help():
    res = run_script_help("data_preprocessing/create_masks.py")
    assert res.returncode == 0
    assert "--image-index" in res.stdout
    assert "--custom-image-path" in res.stdout
    assert "--ksize" in res.stdout
    assert "--majority-threshold" in res.stdout
    assert "--out-dir" in res.stdout

def test_ndwi_labels_cli_help():
    res = run_script_help("ndwi_labels.py")
    assert res.returncode == 0
    assert "--image-index" in res.stdout
    assert "--custom-image-path" in res.stdout
    assert "--ksize" in res.stdout
    assert "--majority-threshold" in res.stdout
    assert "--out-dir" in res.stdout

def test_tile_images_cli_help():
    res = run_script_help("data_preprocessing/tile_images.py")
    assert res.returncode == 0
    assert "--tile-size" in res.stdout
    assert "--overlap" in res.stdout
    assert "--skip-nodata" in res.stdout
    assert "--mask-dir" in res.stdout

def test_augment_tiles_cli_help():
    res = run_script_help("data_preprocessing/augment_tiles.py")
    assert res.returncode == 0
    assert "--input-dir" in res.stdout
    assert "--output-dir" in res.stdout
    assert "--max-tiles" in res.stdout

def test_georeference_cli_help():
    res = run_script_help("data_preprocessing/georeference.py")
    assert res.returncode == 0
    assert "--base-image" in res.stdout
    assert "--aligned-dir" in res.stdout
    assert "--output-dir" in res.stdout

def test_add_mask_band_cli_help():
    res = run_script_help("data_preprocessing/add_mask_band.py")
    assert res.returncode == 0
    assert "--mask-dir" in res.stdout
    assert "--output-dir" in res.stdout
    assert "--limit" in res.stdout

def test_tile_images_overlap_validation():
    from data_preprocessing.tile_images import make_tiles_tiff
    with pytest.raises(ValueError, match="overlap must be in the range"):
        make_tiles_tiff("non_existent.tif", "dummy", overlap=1.0)
    with pytest.raises(ValueError, match="overlap must be in the range"):
        make_tiles_tiff("non_existent.tif", "dummy", overlap=-0.1)

