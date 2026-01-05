##############################
#
#   Original author: Jack Carroll
#   	27 May 2020
#
#   Deering, AK Imagery Download Code
#
#	Modifications made by Rawan Elframawy, May 2024
#
##############################

"""

# Helpful Links
1. [Planet API - Scenes](https://developers.planet.com/apis/orders/scenes/)
2. [Planet API - Items and Asset Types](https://developers.planet.com/docs/apis/data/items-assets/)
3. [Planet API - Scenes Product Bundles Reference](https://developers.planet.com/apis/orders/product-bundles-reference/)
4. [Planet API - Tools: Clip](https://developers.planet.com/apis/orders/tools/#clip)
5. [Planet Labs Jupyter Notebooks - Data API](https://github.com/planetlabs/notebooks/tree/master/jupyter-notebooks/Data-API)
6. [Planet Labs Jupyter Notebooks - Order API](https://github.com/planetlabs/notebooks/tree/master/jupyter-notebooks/Orders-API)"""

# Import Libraries:
import os
import json
import time
import pathlib
import requests
from datetime import datetime
from requests.auth import HTTPBasicAuth
from planet import Session, DataClient, OrdersClient

# Authenticating :
RAWAN_KEY = "************************************"
FRANK_KEY = "************************************"
JACK_KEY = "************************************"

# if your Planet API Key is not set as an environment variable, you can paste it below
if os.environ.get('PL_API_KEY', ''):
    API_KEY = os.environ.get('PL_API_KEY', '')
else:
    API_KEY = RAWAN_KEY

session = requests.Session() # Setup the session
session.auth = (API_KEY, "") # Authenticate



# Planet URLs:
"""
- This code initializes the environment for interacting with the Planet Data API, 
defining key URLs and setting the content type header.
"""
URL = "https://api.planet.com/data/v1" # Setup Planet Data API base URL
quick_url = "{}/quick-search".format(URL) # Setup the quick search endpoint url
orders_url = 'https://api.planet.com/compute/ops/orders/v2' 

headers = {'content-type': 'application/json'} # set content type to json

# ========================================= Functions =========================================
# Geojson files Functions:
def save_polygon(polygon_coordinates, geojson_folder_path, location_name):
    """
    This function takes a list of polygon coordinates, and saves it to a GeoJSON file.

    * Args:
    - polygon_coordinates: A list of coordinates defining a polygon.
    - output_file (optional): The path to the file where the GeoJSON data will be saved. 
    """

    geojson_geometry = {
        "type" : "Polygon",
        "coordinates" : [polygon_coordinates]
    }

    # Save the GeoJSON data to a file
    file_path = f"{geojson_folder_path}{location_name}.geojson"
    try:
        with open(file_path, "w") as f:
            json.dump(geojson_geometry, f, indent=4)
            print(f"Boundary saved to GeoJSON file: {file_path}")
    except IOError as e:
        print(f"Error saving GeoJSON file: {e}")
        
#=======================================================================
def get_boundry_from_file(geojson_folder_path, location_name):
    """
    This function reads a polygon from a GeoJSON file and returns it as a GeoJSON dictionary.

    * Args:
    - file_path: The path to the GeoJSON file containing the polygon.

    * Returns:
    - A GeoJSON dictionary representing the polygon, or None if an error occurs.
    """
    
    file_path = f"{geojson_folder_path}{location_name}.geojson"
    try:
        with open(file_path, "r") as f:
            geojson_data = f.read()
            print("Read the GeoJson file successfully")
            return json.loads(geojson_data)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading GeoJSON file: {e}")
        return None


# Helper function to printformatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))


# Validate Dates functions:
# - This function validates and compares two dates provided as strings.
#     1. Validate dates are written correctly also as this format "yyyy-mm-dd".
#     2. Make sure that start date is before end date.

# - Dates are Invalid if:
#     1. Dates are not written in this format "yyyy-mm-dd". 
#     2. Year is before 2009.

# Note: Make sure of months that 30 days not 31 days. Also Leap Years. 
def validate_and_compare_dates(start_date, end_date):
    """
    Args:
        start_date (str): The start date as a string in the format "YYYY-MM-DD".
        end_date (str): The end date as a string in the format "YYYY-MM-DD".

    Returns:
        tuple: (date_valid, start_date_str, end_date_str)
            - date_valid (bool): True if both dates are valid and start_date is before end_date, otherwise False.
            - start_date_str (str): The validated start date in the format "YYYY-MM-DD" if valid, otherwise None.
            - end_date_str (str): The validated end date in the format "YYYY-MM-DD" if valid, otherwise None.
    """
    try:
        # Parse the input strings as dates in the format "YYYY-MM-DD"
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

        print("Valid date format.")

        date_valid = True

        # Check if start and end years are 2009 or later
        if start_date_dt.year < 2009 or end_date_dt.year < 2009:
            print("Invalid: Start and End year must be 2009 or later.")
            date_valid = False

        # Check if start date is strictly before end date
        if start_date_dt >= end_date_dt:
            print("Invalid: Start date must be before end date.")
            date_valid = False

        # Convert dates back to strings in "YYYY-MM-DD" format
        start_date_str = start_date_dt.strftime('%Y-%m-%d')
        end_date_str = end_date_dt.strftime('%Y-%m-%d')

        return date_valid, start_date_str, end_date_str

    except ValueError:
        print("Invalid date format. Please write the date correctly (YYYY-MM-DD).")
        return False, None, None
    
    
# Get final filter:
# - This function creates a filter configuration for querying satellite imagery based on:
#     1. Cloud Filter. 
#     2. Geometry.
#     3. Date Range.
def get_filter(geojson_geometry, start_date, end_date, cloud_threshold=0.1):
    """
    * Args:
    - geojson_geometry: A GeoJSON dictionary representing the geometry to filter images within.
    - start_date: The start date as a string in the format "yyyy-mm-dd".
    - end_date: The end date as a string in the format "yyyy-mm-dd".
    - cloud_threshold: A float representing the maximum allowable cloud cover (default is 0.1).

    * Returns:
    - A dictionary representing the combined filter configuration.
    """
    
    # Setup Cloud filter; Filters images with over 10% cloud cover
    cloud_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
      "lte": cloud_threshold
       }
     }
       
    # Setup Geometry filter; Filters images to those that are
    # contained within Deering, AK as an example 
    geom_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geojson_geometry
    }
    
    # Set up DateRangeFilter
    # Find imagery within user-defined dates
    start = start_date + "T00:00:00Z"
    end = end_date + "T23:59:59Z"
    
    date_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gt": start,
        "lte": end
      }
    }
        
    # Setup And logical filter; Combines all filters into one
    and_filter = {
      "type": "AndFilter",
      "config": [cloud_filter, geom_filter, date_filter]
    }
    return and_filter


# Get images ids:
# - This function retrieves planet image IDs based on the search filter and item type using planet quick search.
def get_images_ids(search_filter, item_type):
    """
    * Args:
    - search_filter: A dictionary representing the search filter configuration.
    - item_type: A string represents the class of spacecraft and/or processing level of an item.

    * Returns:
    - A list of strings representing the IDs of the images that match the search criteria.
    """
    
    # API request object
    search_request = {
    "item_types": [item_type],
    "filter": search_filter
    }
    
    # fire off the POST request
    search_result = \
    requests.post(
        quick_url,
        auth=HTTPBasicAuth(API_KEY, ''),
        json=search_request)
    
    geojson = search_result.json()
    image_ids = [feature['id'] for feature in geojson['features']]
    
    print(f"Number of images available is {len(image_ids)}")
    
    return image_ids


# Get images dates from IDs:
# - Parses the image ID using the provided date format and returns the date as a datetime object.
def get_image_date(image_id, date_format, date_length):
    """
    * Args:
    - image_id (str): The image ID containing the date and time information.
    - date_format (str): The date format string that specifies how the date and time are formatted in the image ID.
    - slice_length (int): The number of characters to extract from the start of the image ID for parsing.

    * Returns:
    - datetime: The parsed date and time as a datetime object.

    * Raises:
    - ValueError: If the image ID does not match the provided date format.
    """
    
    try:
        # Extract the specified portion of the image_id based on slice_length
        time_string = image_id[:date_length ]
        time = datetime.strptime(time_string, date_format)
        return time
    except ValueError:
        raise ValueError(f"Image ID '{image_id}' does not match the provided date format '{date_format}'")
    
    
# Remove winter image ids :
# - Removes images taken during winter months specified by day of the year range.
def rem_winter(ids, date_format, date_length, winter_start, winter_end):
    """
    * Args:
    - ids (list of str): List of image IDs.
    - date_format (str): The date format string used to parse the date from the image IDs.
    - date_length (int): The number of characters to extract from image ID for parsing. [The whole date]
    - winter_start (int): The starting day of the year for the winter period (e.g., 290 for Oct 16).
    - winter_end (int): The ending day of the year for the winter period (e.g., 136 for May 15).

    * Returns:
    - list of str: List of image IDs that are not taken during the specified winter period.
    """
    
    clear_ids = []
    
    for image_id in ids:
        try:
            date = get_image_date(image_id, date_format, date_length)
            day_of_year = date.timetuple().tm_yday  # Get the day of the year from the datetime object
            
            # Keep the image if the day is outside the winter period
            if not (winter_start <= day_of_year or day_of_year <= winter_end):
                clear_ids.append(image_id)
        except ValueError as e:
            print(e)  # Print error if the date format does not match
    
    return clear_ids
                
                
# Get Order URL:
# This function defines the order details (item type, product bundle, item IDs, and coordinates) 
# and sends a request to the API. The function then extracts the order ID from the response 
# and returns the complete order URL for further tracking or management.
# [All available item and asset types: https://developers.planet.com/docs/apis/data/items-assets/]
# [All available Product Budles: https://developers.planet.com/apis/orders/product-bundles-reference/]
def place_order(item_type, product_bundle, item_ids, coordinates, auth):
    """
    * Args:
    - item_type (str): A string represents the class of spacecraft and/or processing level of an item.
    - product_bundle (str): Product bundles comprise of a group of assets for an item
    
    - item_ids (list of str): A list of item IDs to include in the order.
    - coordinates (list of lists): A list of coordinates defining the area of interest (AOI). 
    The coordinates should be in the format [[[longitude, latitude], ...]].
    - auth (tuple): Authentication credentials as a tuple.

    * Returns:
    - str: The URL of the created order.
    """
    
    request = {"name": "image_details", "source_type": "scenes",
                "products": [{
                "item_ids": item_ids,
                "item_type": item_type,
                "product_bundle": product_bundle}],
                
                "tools": [{
                "clip": {
                    "aoi": { "type": "Polygon", "coordinates": coordinates
            }}}]}
    response = requests.post(orders_url, data=json.dumps(request), auth=auth, headers=headers)
    print(response)
    order_id = response.json()['id']
    print(order_id)
    order_url = orders_url + '/' + order_id
    return order_url
            

# Poll for Order Success:
# - Polls the order URL until the order reaches a final state.                        
def poll_for_success(order_url, auth):
    """
    * Args:
    - order_url (str): The URL of the order to poll.
    - auth (tuple): Authentication credentials as a tuple

    * Returns:
    - str: The final state of the order, which can be 'success', 'failed', or 'partial'.
    """
    
    end_states = ['success', 'failed', 'partial']
    state = "unknown"
    
    while state not in end_states:
        print("Running...")
        
        r = requests.get(order_url, auth=auth)
        response = r.json()
        state = response['state']
        print(state)
        
        if state not in end_states:
            time.sleep(60)
    return state     
                

# Downloading each asset individually
# Code taken from https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/ordering_and_delivery.ipynb
def download_results(results, folder_path, overwrite=False):
    """
    * Args:
    - results (list of dict): A list of dictionaries containing 'location' (URL) and 'name' (file name) of the results to be downloaded.
    - folder_path (str): The path to the folder where the files will be downloaded.
    - overwrite (bool): If True, existing files will be overwritten. Defaults to False.

    * Returns:
    - None
    """
    
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    print('{} items to download'.format(len(results_urls)))
    
    for url, name in zip(results_urls, results_names):
        path = pathlib.Path(os.path.join(folder_path, name))
        
        if overwrite or not path.exists():
            print('downloading {} to {}'.format(name, path))
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
        else:
            print('{} already exists, skipping {}'.format(path, name))
            
            
# =============================================================================
# MAIN CODE BLOCK
# =============================================================================

# Start Downloading:

# Configuration Vraibles (Change these Vraiables According to your preferences)

item_type = "PSScene"
product_bundle =  "analytic_sr_udm2"

date_format_dict = {
    "PSScene" : "%Y%m%d_%H%M%S" , # 15
    "REScene" : "%Y-%m-%dT%H%M%S" # 17
}

start_date_input = "2023-08-01"
end_date_input = "2023-08-10"

winter_start_day = 290
winter_end_day = 136

date_format = date_format_dict[item_type] # Date format in ID string
date_length = 15                # image_id example: "20240529_213419_83_24b2"

download_folder = "D:/GSoC/download/"

geojson_geometry = {
       "type":"Polygon","coordinates":[[
           [-162.80862808227536,66.05894122802519],
           [-162.67404556274414,66.05636369184131],
           [-162.67919540405273,66.07085023305528],
           [-162.7140426635742,66.07669822834144],
           [-162.73550033569333,66.08216210323748],
           [-162.74871826171872,66.09256457840145],
           [-162.73558616638186,66.09760772349222],
           [-162.73798942565915,66.10125903100771],
           [-162.74631500244138,66.10338002568206],
           [-162.76588439941403,66.09764250032609],
           [-162.76399612426752,66.09576448313807],
           [-162.79583930969235,66.08953821061128],
           [-162.81051635742185,66.09166018442527],
           [-162.80862808227536,66.05894122802519]
         ]]}


coordinates = geojson_geometry["coordinates"]
date_valid, start_date, end_date = validate_and_compare_dates(start_date_input,end_date_input)

if date_valid :
    search_filter = get_filter(geojson_geometry, start_date, end_date)
    images_ids = get_images_ids(search_filter, item_type)

    if len(images_ids) > 0:
        images_ids = rem_winter(images_ids, date_format, date_length, winter_start_day, winter_end_day)
        order_url = place_order(item_type, product_bundle, images_ids, coordinates, session.auth)

        state = poll_for_success(order_url, session.auth)

        if state == "success":
            r = requests.get(order_url, auth=session.auth)
            response = r.json()
            results = response['_links']['results']
            download_results(results, download_folder)

    else:
        print("There are not any available images to download")