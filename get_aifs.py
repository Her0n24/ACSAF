"""
This script downloads the latest AIFS run from ECMWF using wget to input folder.

format: wget [ROOT]/20240301/00z/aifs-single/0p25/oper/20240301060000-24h-oper-fc.grib2

Where ROOT is ECMWF:  https://data.ecmwf.int/forecasts

"""
import datetime
import requests
from tqdm import tqdm
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    datefmt= '%Y-%m-%d %H:%M:%S',
                    )

# Define run to 2 digits
run = "00"
run = run.zfill(2)

# Define a function that use reponse to donwload the file from a given url
def download_file(url, local_filename) -> None:
    if os.path.exists(local_filename):
        logging.info(f"File {local_filename} already exists. Skipping download.")
        return
    # NOTE the stream=True parameter below
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
         ) as bar:
             for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    bar.update(len(chunk))
        logging.info(f"Downloaded: {local_filename}")
    else:
        logging.error(f"Failed to download: {url} - Status code: {r.status_code}")

# url = f"https://data.ecmwf.int/forecasts/{today_str}/{run}z/aifs-single/0p25/oper/{today_str}{run}0000-18h-oper-fc.grib2"
# download_file(url, f"input/{today_str}{run}0000-18h-oper-fc.grib2")

# url = f"https://data.ecmwf.int/forecasts/{today_str}/{run}z/aifs-single/0p25/oper/{today_str}{run}0000-42h-oper-fc.grib2"   
# download_file(url, f"input/{today_str}{run}0000-42h-oper-fc.grib2")

