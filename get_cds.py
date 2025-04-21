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
run = "00"
run = run.zfill(2)
today = datetime.date.today() #- datetime.timedelta(days=1)
today_str = today.strftime("%Y%m%d")

city = "Reading"

def get_cams_aod(today, run, city):
    client = cdsapi.Client()
    today_str = today.strftime("%Y%m%d")
    dataset = "cams-global-atmospheric-composition-forecasts"
    request = {
        "variable": [
            "black_carbon_aerosol_optical_depth_550nm",
            "dust_aerosol_optical_depth_550nm",
            "organic_matter_aerosol_optical_depth_550nm",
            "total_aerosol_optical_depth_550nm"
        ],
        "date": [f"{today}/{today}"],
        "time": [f"{run}:00"],
        "leadtime_hour": [
            "18",
            "42"
        ],
        "type": ["forecast"],
        "data_format": "grib",
        "area": [53, -2, 49, 0]
    }

    client.retrieve(dataset, request,f'/home/heron_ng/dev/Afterglow/input/cams_AOD550_{today_str}{run}0000_{city}.grib')