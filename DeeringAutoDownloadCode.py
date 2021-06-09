##############################
#
#   Jack Carroll & Kris Carroll
#   Last updated: 27 May 2021
#   Deering, AK Imagery Download Code
#
##############################

from dotenv import load_dotenv, set_key
import os
import json
import requests
import time
from planet import api
from datetime import datetime
from dateutil.relativedelta import relativedelta


# loading the .env file to retrieve environment variables for API keys and appending to a list
load_dotenv()

api_keys = []
api_keys.append(os.getenv("KRIS_API"))
api_keys.append(os.getenv("JACK_API"))
api_keys.append(os.getenv("FRANK_API"))

# get the last downloaded image date from .env
last_date_string = os.getenv("LAST_DATE")
last_datetime = datetime.strptime(last_date_string, "%Y-%m")
next_datetime = last_datetime + relativedelta(months=1)
next_month_string = next_datetime.strftime("%Y-%m")

# Setup boundry region of Deering, AK (Could be imported; Included here for simplicity)
# TODO:Change this to import the GeoJSON file
aoi = {
       "type": "Polygon",
       "coordinates":[[
           [-162.8235626220703, 66.05622435812153],
           [-162.674560546875, 66.05622435812153],
           [-162.674560546875, 66.10883816429516],
           [-162.8235626220703, 66.10883816429516],
           [-162.8235626220703, 66.05622435812153]
        ]]
}

# setting up the search request filters
query = api.filters.and_filter(
    api.filters.geom_filter(aoi),
    api.filters.range_filter('cloud_cover', lte=0.1),
    api.filters.date_range('acquired', gt=last_datetime, lt=next_datetime),
    api.filters.permission_filter('assets:download')
)
item_types = ["REOrthoTile", "PSOrthoTile"]
request = api.filters.build_search_request(query, item_types)

# loop through download process with each API key until quota is met
while api_keys:
    os.environ["PL_API_KEY"] = api_keys.pop()
    client = api.ClientV1()
    results = client.quick_search(request)
    for item in results.items_iter(1):
        print(item)
    if results.size:
        for item in results.items_iter(1):
            print(item)
    # no results, increment the date
    else:
        exit()
    
exit()


# Helper function to printformatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))
   
    
# Function to get and return start date from the user
def get_start():
    
    # Variable init
    start = ''
    start_nums = []
    
    while True:
        start = input("Please enter a start date of the form: \nyear-mo-dy \nex: 2009-01-13 \n")
        
        # Basic checks to ensure formatting
        if len(start) != 10:
            print("\nInvalid input. \n")
        elif start[4] != '-' or start[7] != '-':
            print("\nInvalid input. \n")
        
        # Checks to ensure date is a functional filters
        else:
            
            # Break string into int array
            for word in start.split('-'):
                if word.isdigit():
                    start_nums.append(int(word))
                    
            # Ensures no characters were entered        
            if len(start_nums) != 3:
                print("\nInvalid input. \n")
                
            # Start year must be after 2009
            elif start_nums[0] < 2009:
                print("\nEntered year must be 2009 or later!")
                start_nums = []
                
            # Check month is valid
            elif start_nums[1] < 1 or start_nums[1] > 12:
                print("\nInvalid month entered")
                start_nums = []
                
            # Check day is valid
            elif not day_check(start_nums[0], start_nums[1], start_nums[2]):
                print("\nInvalid day entered")
                start_nums = []
            else:
                return start


# Function to get and return end date from the user
# start is the start date previously given from user
def get_end(start):
    
    # Variable init
    start_nums = []
    end = ''
    end_nums = []
    
    # Break start date string into year, day, month ints for comparison
    for word in start.split('-'):
            start_nums.append(int(word))
    
    while True:
        end = input("Please enter an end date of the same form \nyear-mo-dy \n")
        
        # Basic checks to ensure formatting
        if len(end) != 10:
            print("\nInvalid input. \n")
        elif end[4] != '-' or end[7] != '-':
            print("\nInvalid input. \n")
            
        # Checks to ensure date is a functional filter
        else:
            
            # Break string into int array
            for word in end.split('-'):
                if word.isdigit():
                    end_nums.append(int(word))
                    
            # Ensures no characters were entered
            if len(end_nums) != 3:
                print("\nInvalid input. \n")
                
            # Check month is valid
            elif end_nums[1] < 1 or end_nums[1] > 12:
                print("\nInvalid month entered")
                end_nums = []
                
            # Check day is valid
            elif not day_check(end_nums[0], end_nums[1], end_nums[2]):
                print("\nInvalid day entered")
                end_nums = [] 
                
            #Further checks to ensure end date is later than start date
            # Compare start year and end year
            elif end_nums[0] < start_nums[0]:
                print("\nYour end date must be later than your start date")
                print("Your start date is ", start)
                end_nums = []
                
            # Additional checks when start and end year are same
            elif end_nums[0] == start_nums[0]:
                
                # Check end month is later than start month
                if end_nums[1] < start_nums[1]:
                    print("\nYour end date must be later than your start date")
                    print("Your start date is ", start)
                    end_nums = []
                    
                # Further checks when start and end month are the same
                elif end_nums[1] == start_nums[1]:
                    
                    # Check end day is later than start day
                    if end_nums[2] < start_nums[2]:
                        print("\nYour end date must be later than your start date")
                        print("Your start date is ", start)
                        end_nums = []
                    else:
                        return end
                else:
                    return end
            else:
                return end
  
    
# Helper function to ensure the day entered is valid
# Returns true if valid, false if not
def day_check(year, month, day):
    
    # General Check
    if day > 31 or day < 1:
        return False
    
    # Check for months with 30 days
    elif month == 2 or month == 4 or month == 6 or month == 9 or month == 11:
        if day > 30:
            return False
        
    # Check February individually
    elif month == 2:
        
        # Check if leap year
        leap_year = False
        if (year % 4) == 0:  
            if (year % 100) == 0:  
                if (year % 400) == 0:  
                    leap_year = True  
        if leap_year and day > 29:
            return False
        elif day > 29:
            return False
    return True

                
            
# Initializes the server request
# Inputs are the user-defined start and end date to search through
# Start and end dates must be entered in the form year-mo-dy
    
# TODO: This code also is built to support the download of REScenes, if download capacity is large enough
def request_init(start_date, end_date):
    
    # Specify the sensors/satellites or "item types" to include in our results
    # TODO: To include REScenes in the search results, replace the next line with
    # item_types = ["PSScene4Band", "REScene"]
    item_types = ["PSScene4Band"]
    
    # Get filter
    and_filter = get_filter(start_date, end_date)
    
    # Setup the request
    temp_request = {
        "item_types" : item_types,
        "filter" : and_filter
    }
    
    # Return final request
    return temp_request


# Initializes additional request to return only PSScene3Band images
def request_init_3band(start_date, end_date):
    
    # Specify the sensors/satellites or "item types" to include in our results
    item_types = ["PSScene3Band"]
    
    # Get filter
    and_filter = get_filter(start_date, end_date)

    # Setup the request
    temp_request = {
        "item_types" : item_types,
        "filter" : and_filter
    }
    
    # Return request
    return temp_request

# Helper function to get query filter
def get_filter(start_date, end_date):
    
    # Setup Cloud filter; Filters images with over 10% cloud cover
    cloud_filter = {
       "type": "RangeFilter",
       "field_name": "cloud_cover",
       "config": {
         "lte": 0.1
       }
     }
       
    # Setup Geometry filter; Filters images to those that are
    # contained within Deering, AK
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
            "config": [cloud_filter, geom_filter, 
                        date_filter]
    }
    
    return and_filter


# Prompts user and gets the number of images they would like to download
def get_download_num():
    num = input("\nHow many images would you like to download?\n(Enter 0 to download all images, -1 to cancel)\n")
    return num
    

# Prompts user and gets starting image index for download
def get_starting_num():
    num = input("\nIs there an image index you would like to start the download at?\nEx: If you have already downloaded the first 15 images in this date range, enter 16\nOtherwise enter 0, or -1 to quit\n")
    return num
    
# Function to print a link to an image's thumbnail
# TODO: Remove this function for release; For testing only
def preview(feature):
    link = feature["_links"]
    thumbnail = link["thumbnail"]
    print(thumbnail, "?api_key=", Planet_Key)


# Function to store all search image ids into the image_ids array
# Geojson is post response, res is server connection
def init_image_arr(geojson, res, image_ids):

    while True:
        
        # Add each image id to the list
        features = geojson["features"]
        for i in range(0, len(geojson["features"])):
            feature = features[i]
            image_ids.append(feature["id"])
        
        # Return link of next result page
        next_url = geojson["_links"]["_next"]
        
        # Send POST request to next result page
        res = session.get(next_url)
        
        # Assign response to a variable
        geojson = res.json()
        
        # Exit if page has no further results
        if len(geojson["features"]) == 0:
            break


# Function to remove images of the winter months (Oct 16 to May 15)
# TODO: Alter to desired winter month range
# Returns id array
def rem_winter(ids = []):
    
    # Array to store id indexes for removal
    removal = []
    
    for i in range(0, len(ids)):
        
        # Holds info- 0-3: year, 4-5: month, 6-7: day, 8-11: time
        time = []

        # Reformat current min for comparison
        # Check for RE image
        if ids[i][4] == '-':
            time = RE_format(ids[i])
            
        # Else is PS image
        else:
            time = PS_format(ids[i])
            
        # Convert date to string
        curr_string = list_to_string(time)

        # Get image month and day
        month = int(curr_string[4:6])
        day = int(curr_string[6:8])
        
        # Remove images of Nov-March
        if month in [11, 12, 1, 2, 3, 4]:
            removal.append(1)
        
        # Additional check if month is October or May
        elif month in [10, 5]:
            
            # Remove images after October 15 and before May 15
            if month == 10 and day > 15:
                removal.append(1)
            elif month == 5 and day < 15:
                removal.append(1)
            else:
                removal.append(0)
        else:
            removal.append(0)
            
    # Create new array to return
    clear_ids = []
    
    # Add each non-winter element to new array
    for i in range(0, len(ids)):
        if removal[i] == 0:
            clear_ids.append(ids[i])
            
    # Return array with winter months removed
    return clear_ids


# Function to merge the PSScene3band ids array when image_ids
# Doesn't already contain a 4 banded version of the image
# Returns number of added 3band images
# NOTE: This function is only necessary if both 3 banded and 4 banded images are being searched for
def merge_ids(image_ids = [], scope3band_ids = []):
    
    # Counter to keep track of 3banded images added
    counter_3band = 0
    
    # Loop through each item in scope3band_ids
    for i in range(0, len(scope3band_ids)):
        
        # Bool to keep track of whether item is duplicate
        new = True
        
        # Loop through each item in image_ids, change bool value if image is dupe
        for j in range(0, len(image_ids)):
            if scope3band_ids[i] == image_ids[j]:
                new = False

        # If image is new, append to image_ids list, increment counter_3band
        if new:
            image_ids.append(scope3band_ids[i])
            counter_3band = counter_3band + 1
    
    # Return number of 3band images added
    return counter_3band
            
# Function to sort the image ids based on date
# Uses modified simple selection sort
# Returns array with locations of PSScene3band images
# TODO: To simplify, represent dates as a datetime object
# NOTE: scope3band_count will always be 0 unless 3 banded images are also included in the search
def date_sort(image_ids, scope3band_count):
    
    # Init array with locations of PSScene3band images
    locations = []
    for i in range(0, len(image_ids)):
        if i < len(image_ids) - scope3band_count:
            locations.append(0)
        else:
            locations.append(1)
    
    for i in range(0, len(image_ids)):
        
        # Find min element in unsorted array
        min_id = i
        min_ele = []
        
        # Reformat current min for comparison
        # Check for RE image
        if image_ids[i][4] == '-':
            min_ele = RE_format(image_ids[i])
            
        # Else is PS image
        else:
            min_ele = PS_format(image_ids[i])
            
        ele_string = list_to_string(min_ele)
        # Gets year, month and day of current min
        min_year = int(ele_string[0:4])
        min_mo = int(ele_string[4:6])
        min_day = int(ele_string[6:8])
        min_time = int(ele_string[8:14])
        
        for j in range(i+1, len(image_ids)):
            curr_ele = []
            
            # Reformat current id to comparable format
            if image_ids[j][4] == '-':
                curr_ele = RE_format(image_ids[j])
            else:
                curr_ele = PS_format(image_ids[j])
                
            curr_string = list_to_string(curr_ele)
            # Gets year, month, day of current comparison element
            curr_year = int(curr_string[0:4])
            curr_mo = int(curr_string[4:6])
            curr_day = int(curr_string[6:8])
            curr_time = int(ele_string[8:14])
            
            # Comparison
            # Compare years
            if min_year > curr_year:
                
                # Reassign min variables
                min_id = j
                min_ele = curr_ele
                min_year = curr_year
                min_mo = curr_mo
                min_day = curr_day
                min_time = curr_time
            
            # Additional code if year is the same
            elif min_year == curr_year:
                
                # Compare months
                if min_mo > curr_mo:
                    
                    # Reassign min variables
                    min_id = j
                    min_ele = curr_ele
                    min_year = curr_year
                    min_mo = curr_mo
                    min_day = curr_day
                    min_time = curr_time
                    
                # Additional code if month is the same
                elif min_mo == curr_mo:
                    
                    # Compare days:
                    if min_day > curr_day:
                        
                        # Reassign min variables
                        min_id = j
                        min_ele = curr_ele
                        min_year = curr_year
                        min_mo = curr_mo
                        min_day = curr_day
                        min_time = curr_time
                        
                    # Additional code if day is the same
                    elif min_day == curr_day:
                        
                        # Call comparison
                        if min_time > curr_time:
                            
                            # Reassign min variables
                            min_id = j
                            min_ele = curr_ele
                            min_year = curr_year
                            min_mo = curr_mo
                            min_day = curr_day
                            min_time = curr_time
                    
        # Swap first element and minimum element
        image_ids[i], image_ids[min_id] = image_ids[min_id], image_ids[i]
        
        # Swap bool values representing PSScene 3 band images
        locations[i], locations[min_id] = locations[min_id], locations[i]
        
    # Return locations of PSScene 3 band images
    return locations
                

# TODO: If more image types required; It might be simpler to
# combine format functions into a single function that takes
# any image type, checks for type, and then formats
    
# Helper function to parse RE images into sortable form
# Formates titles of the form 2018-12-03T083453_RE4_1B_band1.tif
def RE_format(image_id):
    ele = []
    ele.append(image_id[0])
    ele.append(image_id[1])
    ele.append(image_id[2])
    ele.append(image_id[3])
    ele.append(image_id[5])
    ele.append(image_id[6])
    ele.append(image_id[8])
    ele.append(image_id[9])
    for i in range(11, 17):
        ele.append(image_id[i])
    return ele 

# Helper function to parse PS images into sortable form
def PS_format(image_id):
    ele = []
    for i in range(0,8):
        ele.append(image_id[i])
    for i in range(9, 15):
        ele.append(image_id[i])
    return ele
                
# Helper function to convert list to string  
def list_to_string(s):  
    
    # initialize an empty string 
    str1 = "" 
    
    # return string   
    return (str1.join(s)) 


# Creates the order to submit to Planet database
# NOTE: Much of this code is built around the potential to also include 
# RE or Planetscope 3 Banded imagery
def create_order(ids, locations, download_num, start, end, starting_num):
    
    # Create arrays for each item type
    PS3Band = []
    PS4Band = []
    RE = []
    
    
    # Add images to respective array
    for i in range(starting_num, download_num + starting_num):
        
        # Check for RE image
        if ids[i][4] == '-':
            RE.append(ids[i])
            
        # Check for PS3Band image
        elif locations[i] == 1:
            PS3Band.append(ids[i])
            
        # Else must be a PS4Band image
        else:
            PS4Band.append(ids[i])
      
    # Create order name    
    name = str(download_num) + " AOI Images " + start + " to " + end
    
    # Create product list
    products = []
    
    # Create PSScene3Band product
    if len(PS3Band) > 0:
        PS3Product = create_product(PS3Band, "PSScene3Band")
        products.append(PS3Product)
    
    # Create PSScene4Band product
    if len(PS4Band) > 0:
        PS4Product = create_product(PS4Band, "PSScene4Band")
        products.append(PS4Product)
        
    # Create REScene product
    if len(RE) > 0:
        REProduct = create_product(RE, "REScene")
        products.append(REProduct)
        
    # Creates order
    if len(products) > 0:
        order = {
            # Sets order delivery to be downloadable as a zip
            "delivery": {'archive_type': 'zip', 'single_archive': True},
            "name": name,
            "products": products
            }
        return order
    
    else:
        print("\nImproper order setup")
        return None
      
        

# Helper function to create each product bundle
# NOTE: If an error is encountered downloading PS images, it is likely due to
# an improper item type declaration; Check here first
# NOTE: For the orthorectification process, RPC data is required, which is only
# present in the basic_analytic bundle, which is why this bundle is required regardless of sattelite type.
def create_product(item, item_type):
            
    product = {
       "item_ids": item,
       "item_type": item_type,
       "product_bundle": "basic_analytic"
    }
    return product


# Function that polls for success, returns when order is ready to download
def poll(geojson):
    
    # Poll server until state is no longer queued
    while(True):
        
        # Refresh order status URL
        status_url = geojson['_links']['_self']
        res = session.get(status_url)
        
        # Assign response to a variable
        geojson = res.json()
        
        # Assign order state to a variable
        state = geojson['state']
        
        # Check if order is finished processing
        if state in ['success', 'failed', 'partial']:
            return state
        
        # Else print notice
        print("\nProccessing, order state", state)
        
        # Wait before repeating loop
        time.sleep(60)


# Function to download results
# Code taken from https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/orders/ordering_and_delivery.ipynb
def download_results(results, overwrite=False):
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    print('{} items to download'.format(len(results_urls)))
    
    for url, name in zip(results_urls, results_names):
        path = pathlib.Path(os.path.join('data', name))
        
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

# NOTE: This code also supports the ability to download 3 Banded PlanetScope imagery when
# 4 Banded images are unavailable. This functionality has been removed from this main block,
# due to the fact that certain permissions are required to download basic_analytic files for
# 3 Banded imagery, which we currently do not have access to. However, the methods in this code 
# still have the capability to do so in case someone else using this code has those permissions.
# Please see this commit for how to include this functionality in the main code block:
# https://github.com/fwitmer/CoastlineExtraction/commit/6b90bd9034e6089434d82e9d8e581ad6e0bdfd79



# Setup the session
session = requests.Session()

# Authenticate
session.auth = (Planet_Key, "")

# Make a GET request to the Planet Data API
res = session.get(URL)

# Check connection
if res.status_code == 200:
    print('Connected to server')
else:
    print('Connection failed, error status code: ', res.status_code)
    exit()
    
# Setup the quick search endpoint url
quick_url = "{}/quick-search".format(URL)

# Setup the orders endpoint url
orders_url = 'https://api.planet.com/compute/ops/orders/v2'

# Array to store IDs of all images
image_ids = []

# Integer to store the number of scope3band images that don't have 
# 4 band copies already in image_ids
# Requiring a scope3band_count is outdated as of 7/24/20 When PSScope3Band functionality is 
# Removed from the main block
scope3band_count = 0

# Bool array to store the positions of scope3band images in the image_ids array
locations = []

# Loops until user inputs meaningful date range
while True:
    # Get user defined parameters
    start = get_start()
    end = get_end(start)
    
    # Creates the request
    request = request_init(start, end)
    
    # Send the POST request to the API quick search endpoint
    res = session.post(quick_url, json=request)
    
    # Assign the response to a variable
    geojson = res.json()
    
    # Puts available images into image_ids array
    try:
        print("\nProcessing...")
        init_image_arr(geojson, res, image_ids)
        print("Images found")
        break
    except:
        #TODO: This except might also trigger if server rejects the request; 
        #Potentially there should be an additional message added here
        print(res)
        print("\nThere is no data available of your AOI in that date range, please try again!")


# Removes all images from Oct 16 - May 15
# TODO: Remove this call if this additional filter isn't desired
image_ids = rem_winter(image_ids)
    

# Sorts ids based on date
locations = date_sort(image_ids, scope3band_count)


print("\nThere are", len(image_ids),"images availabe to download in your AOI from", start, "to", end)


# Gets max number of images to download
while(True):
    download_num = get_download_num()
    
    # Check user input is valid
    if type(download_num) is str:
        try:
            download_num = int(download_num)
            break;
        except:
          print("\nInvalid input")  
    else:
        print("\nInvalid input")
        
# Check for user exit
if download_num < 0:
    print("\nUser has terminated the program")
    exit()
    

# Checks if there is an image download start point
while(True):
    starting_num = get_starting_num()
    
    # Check user input is valid
    if type(starting_num) is str:
        try:
            starting_num = int(starting_num)
            
            # Check starting num is possible
            if starting_num + download_num > len(image_ids):
                print("\n starting index too large for number of image downloads requested")
            else:
                break
            
        except:
          print("\nInvalid input")  
    else:
        print("\nInvalid input")

# Check for user exit
if starting_num < 0:
    print("\nUser has terminated the program")
    exit()
    
# Downloads all images if zero entered
elif download_num == 0:
    download_num = len(image_ids)
    print("\nDownloading", download_num, "images")
    
# Downloads all images and prints response if input number is too large
elif download_num > len(image_ids):
    download_num = len(image_ids)
    print("\nInput number too large; Downloading all available images")
    print("\nDownloading", download_num, "images")


# Creates order for download
order = create_order(image_ids, locations, download_num, start, end, starting_num)

# Send the order request to planet orders_url endpoint
res = session.post(orders_url, json=order)

# Assign the response to a variable
geojson = res.json()

# Check if order went through
try: 
    print("\nOrder submitted successfully: Order status", geojson['state'])
except:
    print("\nOrder submission failed, please ensure order size does not exceed your remaining download quota")
    exit()


# Wait until order is ready to download
while True:
    try:
        print("\nBeginning session poll")
        final_state = poll(geojson)
        # Ensure order is no longer being processed:
        if final_state != "queued" and final_state != "processing":
            break
        else:
            print("\nAn error has occured when processing your request.")
            print("\nYour current order state is: ", final_state)
            print("Waiting for 2 minutes, then attempting to reconnect to server")
            time.sleep(120)
    
    # Sometimes server will reject multiple session requests, this catches the error
    except:
        print("\nAn error has occured when processing your request.")
        print("\nYour current order state is: ", final_state)
        print("Waiting for 2 minutes, then attempting to reconnect to server")
        time.sleep(120)




# Refresh order status URL
status_url = geojson['_links']['_self']
res = session.get(status_url)

# Assign response to a variable
geojson = res.json()

 
               
# Check final order state    
# Check if success
if final_state == "success":
    
    # Downloads complete order
    print("\nOrder activated successfully! Commencing download")
    results = geojson['_links']['results']
    download_results(results)
    
# Check if partial success
elif final_state == "partial":
    
    # Downloads partial order
    print("\nOrder partially successfully! Commencing download")
    results = geojson['_links']['results']
    download_results(results)
    
# Otherwise, order failed
else:
    print("\nOrder was not activated successfully.", "Final order status:", final_state)
    
