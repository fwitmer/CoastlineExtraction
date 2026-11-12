import importlib.util
import sys
import types
import glob
from pathlib import Path


def test_augment_tiles_finds_files_in_directory_with_square_brackets(
    tmp_path, monkeypatch
):
    # Mock dependencies that are not needed for this test.
    fake_cv2 = types.ModuleType("cv2")
    fake_numpy = types.ModuleType("numpy")

    fake_rasterio = types.ModuleType("rasterio")
    fake_rasterio.__path__ = []

    fake_rasterio_transform = types.ModuleType("rasterio.transform")
    fake_rasterio_transform.from_bounds = lambda *args: None
    fake_rasterio.transform = fake_rasterio_transform

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "rasterio", fake_rasterio)
    monkeypatch.setitem(
        sys.modules, "rasterio.transform", fake_rasterio_transform
    )

    fake_config = types.ModuleType("load_config")
    fake_config.load_config = lambda: None
    fake_config.get_tile_images_output_folder = lambda config: None
    fake_config.get_augment_tiles_output_folder = lambda config: None
    monkeypatch.setitem(sys.modules, "load_config", fake_config)

    # Load the actual project file directly.
    module_path = (
        Path(__file__).parent.parent
        / "data_preprocessing"
        / "augment_tiles.py"
    )

    spec = importlib.util.spec_from_file_location(
        "augment_tiles_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Create a directory containing square brackets.
    tile_dir = tmp_path / "tiles[2026]"
    output_dir = tmp_path / "output"
    tile_dir.mkdir()

    tile_file = tile_dir / "image_01-of-10.tif"
    tile_file.touch()

    discovered_files = []
    original_glob = glob.glob

    def tracking_glob(pattern):
        files = original_glob(pattern)
        discovered_files.extend(files)
        return files

    monkeypatch.setattr(module.glob, "glob", tracking_glob)

    module.augment_tiles(str(tile_dir), str(output_dir))

    assert str(tile_file) in discovered_files
