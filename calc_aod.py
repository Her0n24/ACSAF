import xarray as xr
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# run = "00"
# run = run.zfill(2)
# today = datetime.date.today() #- datetime.timedelta(days=1)
# today_str = today.strftime("%Y%m%d")

def calc_aod(run, today_str, city):
    ds_aod550 = xr.open_dataset(f'input/cams_AOD550_{today_str}{run}0000_{city}.grib', engine = 'cfgrib') 
    # Average values over all latitude and longitude to get a single value
    # Grid spacing is different to the cloud cover product, so different strategy here
    ds_aod550 = ds_aod550.mean(dim=['latitude', 'longitude'])

    dust_aod550 = ds_aod550['duaod550'].values # Dat 1 to dat 2
    total_aod550 = ds_aod550['aod550'].values

    dust_aod550_ratio = dust_aod550/ total_aod550
    
    return dust_aod550, total_aod550, dust_aod550_ratio

