#!/usr/bin/env python3
"""
Simple test script to verify the fix for the date validation bug
"""

from datetime import datetime

def validate_and_compare_dates_fixed(start_date, end_date):
    """
    Fixed version of the function with the bug corrected
    """
    try:
        # Attempt to parse the input strings as dates in the format "yyyy-mm-dd"
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        print("Valid date format.")
        
        date_valid = True
        # Check if start and end years are 2009 or later
        if start_date_obj.year < 2009 or end_date_obj.year < 2009:
            print("Invalid: Start and End year must be 2009 or later.")
            date_valid = False
        
        # Check if start date is before end date
        if start_date_obj >= end_date_obj:
            print("Invalid: Start date must be before end date.")
            date_valid = False
        
        # Convert start_date and end_date to strings in "YYYY-MM-DD" format
        start_date_str = start_date_obj.strftime('%Y-%m-%d')
        end_date_str = end_date_obj.strftime('%Y-%m-%d')
        
        return date_valid, start_date_str, end_date_str
    
    except ValueError:
        print("Invalid date format. Please write the date correctly (YYYY-MM-DD).")
        return False, None, None

# Test the fixed function
print("Testing the fixed function...")
result = validate_and_compare_dates_fixed("2023-08-01", "2023-08-10")
print(f"Result: {result}")

# This should work without any NameError
expected = (True, "2023-08-01", "2023-08-10")
if result == expected:
    print("✅ Test passed! The bug fix is working correctly.")
else:
    print(f"❌ Test failed! Expected {expected}, got {result}")