#!/usr/bin/env python3
"""
Test script to verify the fix for the date validation bug in DeeringAutoDownloadCode.py
"""

import sys
import os

# Add the current directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function we want to test
from DeeringAutoDownloadCode import validate_and_compare_dates

def test_valid_dates():
    """Test with valid dates"""
    print("Testing valid dates...")
    result = validate_and_compare_dates("2023-08-01", "2023-08-10")
    print(f"Result: {result}")
    assert result == (True, "2023-08-01", "2023-08-10"), f"Expected (True, '2023-08-01', '2023-08-10'), got {result}"
    print("✓ Valid dates test passed")

def test_invalid_format():
    """Test with invalid date format"""
    print("\nTesting invalid date format...")
    result = validate_and_compare_dates("2023/08/01", "2023-08-10")
    print(f"Result: {result}")
    assert result == (False, None, None), f"Expected (False, None, None), got {result}"
    print("✓ Invalid format test passed")

def test_start_after_end():
    """Test with start date after end date"""
    print("\nTesting start date after end date...")
    result = validate_and_compare_dates("2023-08-10", "2023-08-01")
    print(f"Result: {result}")
    assert result == (False, "2023-08-10", "2023-08-01"), f"Expected (False, '2023-08-10', '2023-08-01'), got {result}"
    print("✓ Start after end test passed")

def test_old_dates():
    """Test with dates before 2009"""
    print("\nTesting dates before 2009...")
    result = validate_and_compare_dates("2008-08-01", "2023-08-10")
    print(f"Result: {result}")
    assert result == (False, "2008-08-01", "2023-08-10"), f"Expected (False, '2008-08-01', '2023-08-10'), got {result}"
    print("✓ Old dates test passed")

if __name__ == "__main__":
    print("Running tests for validate_and_compare_dates function...")
    
    try:
        test_valid_dates()
        test_invalid_format()
        test_start_after_end()
        test_old_dates()
        
        print("\n🎉 All tests passed! The bug fix is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)