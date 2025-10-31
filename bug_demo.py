#!/usr/bin/env python3
"""
Demonstration of the bug that existed before the fix
"""

from datetime import datetime

def validate_and_compare_dates_buggy(start_date, end_date):
    """
    Buggy version of the function that would cause a NameError
    """
    try:
        # BUG: Using undefined variables start_date_input and end_date_input
        # instead of the function parameters start_date and end_date
        start_date_obj = datetime.strptime(start_date_input, '%Y-%m-%d')  # ❌ Undefined variable
        end_date_obj = datetime.strptime(end_date_input, '%Y-%m-%d')      # ❌ Undefined variable
        
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

# Demonstrate the bug
print("Demonstrating the bug that existed before the fix:")
print("Calling validate_and_compare_dates_buggy('2023-08-01', '2023-08-10')...")
try:
    result = validate_and_compare_dates_buggy("2023-08-01", "2023-08-10")
    print(f"Unexpectedly succeeded with result: {result}")
except NameError as e:
    print(f"❌ NameError occurred as expected: {e}")
    print("This is the bug that was fixed!")