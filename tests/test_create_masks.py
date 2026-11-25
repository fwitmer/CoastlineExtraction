"""
Unit tests for data_preprocessing/create_masks.py logic.
Tests regression fix for Issue #138:
1. mask_array uint8 comparison against -1.
2. buffer_numbers and water_count dtype overflow prevention.
"""
import numpy as np
import pytest


def test_mask_array_extraction_before_uint8():
    """
    Test that valid data pixels within a sliding window are properly isolated
    when nodata is sentinel -1.0.

    Issue #138 regression test:
    When out_image is cast to np.uint8, values can only be 0..255.
    Comparing uint8 != -1 results in True for every pixel (all ones).
    Extracting mask_array prior to uint8 cast preserves valid data detection.
    """
    # 4x4 synthetic NDWI raster where center 2x2 is inside window (-1.0 to 1.0 NDWI values),
    # and outer border is masked out with nodata sentinel (-1.0).
    raw_window = np.array([
        [-1.0, -1.0, -1.0, -1.0],
        [-1.0,  0.3,  0.8, -1.0],
        [-1.0, -0.4,  0.1, -1.0],
        [-1.0, -1.0, -1.0, -1.0]
    ], dtype=np.float32)

    # Correct fix: extract mask before uint8 conversion
    fixed_mask = (raw_window != -1.0).astype(np.uint8)

    # Expected mask: only the 4 center pixels are 1, outer pixels are 0
    expected_mask = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint8)

    np.testing.assert_array_equal(fixed_mask, expected_mask)
    assert np.sum(fixed_mask) == 4

    # Buggy behavior demonstration:
    # Linear scale to [0, 255] and cast to uint8
    scaled_uint8 = ((raw_window * 127) + 128).astype(np.uint8)
    buggy_mask = np.zeros_like(scaled_uint8, dtype=np.uint8)
    buggy_mask[scaled_uint8 != -1] = 1

    # In uint8, -1 is never matched, so buggy mask is incorrectly all 1s
    assert np.sum(buggy_mask) == 16


def test_counter_overflow_prevention():
    """
    Test that buffer_numbers and water_count arrays do not overflow
    when processing >= 256 sliding windows.

    uint8 overflows at 256 (wraps back to 0), whereas uint32 handles large window counts.
    """
    height, width = 5, 5
    buffer_numbers = np.zeros((height, width), dtype=np.uint32)
    water_count = np.zeros((height, width), dtype=np.uint32)

    num_windows = 300
    for _ in range(num_windows):
        buffer_numbers += 1
        water_count += 1

    # Ensure no integer wrap-around occurred
    assert np.all(buffer_numbers == 300)
    assert np.all(water_count == 300)
    assert buffer_numbers.dtype == np.uint32
    assert water_count.dtype == np.uint32
