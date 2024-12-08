# ======================= THE SCRIPT ===============================
## This script samples SMAP data using point information (geographic coordinates)
# The Appears' API documentation is here: https://appeears.earthdatacloud.nasa.gov/api/?python#introduction
# The Appears' website is here: https://appeears.earthdatacloud.nasa.gov/

# ======================= THE DATA ===============================
# This script downloads SMAP Enhanced L3 Radiometer Global and Polar Grid Daily 9 km EASE-Grid Soil Moisture, Version 5 (SPL3SMP_E)
# https://nsidc.org/data/spl3smp_e/versions/5#anchor-2
# TODO: which product to use, L2 or L3?

# Layers downloaded are as follows:
# Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag
# Soil_Moisture_Retrieval_Data_AM_soil_moisture
# Soil_Moisture_Retrieval_Data_AM_soil_moisture_error
# Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm
# Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm
# Soil_Moisture_Retrieval_Data_PM_soil_moisture_error_pm

# The product includes includes soil moisture retrievals from three algorithms:
# the ones I picked (the ones without notation) are from Dual Channel Algorithm (DCA)

# A list of product code is here
# https://appeears.earthdatacloud.nasa.gov/products

# ======================= START OF THE CODE ===============================

# Import libraries
import json
import requests
import os
from ismn.interface import ISMN_Interface

# Specify current directory and create output directory if it does not exist
os.chdir("G:/Shared drives/Ryoko and Hilary/SMSigxSMAP/analysis/0_code")

## Prepare tasks

# Parameters
params = {'pretty': True}

# Get the target point locations (ISMN sensor coordinates)
network_name = 'OZNET'
# TODO: Generalize the operation
in_path = 'G:/Shared drives/Ryoko and Hilary/SMSigxISMN/analysis/0_data_raw/ISMN'
data_net = ISMN_Interface(in_path, network=[network_name])
station_name_ismn=[]
lat_ismn=[]
lon_ismn=[]
for network, station, sensor in data_net.collection.iter_sensors(variable='soil_moisture'): # Only select the surface soil moisture within 0-5cm sensor depth to compare with SMAP data
    # Read metadata
    # TODO: add a depth information to cut off the stations to look at
    # TODO: add a start & end of record to cut off the stations to look at
    meta_pd = sensor.metadata.to_pd()
    station_name_ismn.append(meta_pd['station'][0])
    lat_ismn.append(meta_pd['latitude'].values[0])
    lon_ismn.append(meta_pd['longitude'].values[0])

# Load the task request template from a file
sample_request_path = "./sample_point_request.json"
with open(sample_request_path) as json_file:
    task = json.load(json_file)

task['params']['dates'][0]['startDate'] = '03-31-2015'
task['params']['dates'][0]['endDate'] = '03-30-2022'

# for i in range(len(lat_ismn)):
sensor_coord = {
    'id': '0',
    'category': station_name_ismn[0],
    'latitude': lat_ismn[0],
    'longitude': lon_ismn[0]
}
task['params']['coordinates'].append(sensor_coord)

# Write a json for target point locations
my_request_path = "./my_test_request.json"
with open(my_request_path, 'w') as json_file:
    json.dump(task, json_file, indent=4)

## Login to appears
my_credential_path = "./auth.json"
with open(my_credential_path, 'r') as infile:
    my_credentials = json.load(infile)

response = requests.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=(my_credentials['username'], my_credentials['password']))
token_response = response.json()
print(token_response)

# Submit the task request
token = token_response['token']
response = requests.post(
    'https://appeears.earthdatacloud.nasa.gov/api/task',
    json=task,
    headers={'Authorization': 'Bearer {0}'.format(token)}
)
task_response = response.json()
print(task_response)

# To check the updates every x seconds
task_id = task_response['task_id']

import time
import datetime
starttime = time.time()
while task_response['status'] =='pending' or task_response['status']=='processing':
    print("Still processing the request at %s" % {datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')})
    response = requests.get(
        'https://appeears.earthdatacloud.nasa.gov/api/task/{0}'.format(task_id),
        headers={'Authorization': 'Bearer {0}'.format(token)}
    )
    task_response = response.json()
    time.sleep(300.0 - ((time.time() - starttime) % 300.0)) # check the request every 300 sec
print("Done processing on Appears' side")

# To bundle download
# TODO: Avoid overwriting the previous data
response = requests.get(
    'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}'.format(task_id),
    headers={'Authorization': 'Bearer {0}'.format(token)}
)
bundle_response = response.json()
print(bundle_response)
len(bundle_response)
for i in range(len(bundle_response)):

    # get a stream to the bundle file
    file_id = bundle_response['files'][i]['file_id']
    filename = bundle_response['files'][i]['file_name']
    print("Download a file %s" % filename)

    response = requests.get(
        'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(task_id,file_id),
        headers={'Authorization': 'Bearer {0}'.format(token)},
        allow_redirects=True,
        stream=True
    )

    # create a destination directory to store the file in
    dest_dir = "../1_data/SMAP/OZNET"
    filepath = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # write the file to the destination directory
    with open(filepath, 'wb') as f:
        for data in response.iter_content(chunk_size=8192):
            f.write(data)


# ======================= END OF THE CODE ===============================

# Point sampling returns the results as csv file
# Area sampling returns the results as nc file

"""
## Request a sample
product_id = 'SPL3SMP_E.005'
response = requests.get(
    'https://appeears.earthdatacloud.nasa.gov/api/product/{0}'.format(product_id),
    params=params)
product_response = response.text
print(product_response)
"""