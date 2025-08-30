"""Download AOD550 data from CAMS and save it to a netCDF file

format: 18 is day 1 and 42 is day 2 sunset hours

output: input/cams_AOD550_{today_str}{run}0000.grib'

Data availability (HH:MM)
CAMS Global analyses and forecasts:

00 UTC forecast data availability guaranteed by 10:00 UTC

12 UTC forecast data availability guaranteed by 22:00 UTC

"""
import cdsapi
import datetime
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    datefmt= '%Y-%m-%d %H:%M:%S',
                    )
run = "00"
run = run.zfill(2)
today = datetime.date.today() #- datetime.timedelta(days=1)
today_str = today.strftime("%Y%m%d")

def get_cams_aod(today, run, city, today_str, input_path):
    if os.path.exists(f'{input_path}/cams_AOD550_{today_str}{run}0000_{city}.grib'):
        logging.info("CAMS grib already exists. Skipping download.")
        return
    else:
        if city == "Reading":
            client = cdsapi.Client()
            today_str = today.strftime("%Y%m%d")
            dataset = "cams-global-atmospheric-composition-forecasts"
            request = {
                "date": [f"{today}/{today}"],
                "time": [f"{run}:00"],
                "leadtime_hour": [
                    "18",
                    "42"
                ],
                "type": ["forecast"],
                "data_format": "grib",
                "variable": [
                    "black_carbon_aerosol_optical_depth_550nm",
                    "dust_aerosol_optical_depth_550nm",
                    "organic_matter_aerosol_optical_depth_550nm",
                    "total_aerosol_optical_depth_550nm"
                ],
                "area": [53, -2, 49, 0]
                }
        elif city =='HongKong':
            client = cdsapi.Client()
            today_str = today.strftime("%Y%m%d")
            dataset = "cams-global-atmospheric-composition-forecasts"
            request = {
                "date": [f"{today}/{today}"],
                "time": [f"{run}:00"],
                "leadtime_hour": [
                    "24",
                    "48"
                ],
                "type": ["forecast"],
                "data_format": "grib",
                "variable": [
                    "black_carbon_aerosol_optical_depth_550nm",
                    "dust_aerosol_optical_depth_550nm",
                    "organic_matter_aerosol_optical_depth_550nm",
                    "total_aerosol_optical_depth_550nm"
                ],
                "area": [23, 113, 21, 115] 
                }
        else:
            logging.error("City not supported. Please use 'Reading' or 'HongKong'.")
            raise ValueError("City not supported. Please use 'Reading' or 'HongKong'.")

        client.retrieve(dataset, request,f'{input_path}/cams_AOD550_{today_str}{run}0000_{city}.grib')