import xarray as xr
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import logging

# run = "00"
# run = run.zfill(2)
# today = datetime.date.today() #- datetime.timedelta(days=1)
# today_str = today.strftime("%Y%m%d")

def calc_aod(run, today_str, input_path, lats, lons):
    ds_aod550 = xr.open_dataset(f'{input_path}/cams_AOD550_{today_str}{run}0000.grib', engine='cfgrib')
    da_total = ds_aod550['aod550']
    da_dust = ds_aod550['duaod550'] if 'duaod550' in ds_aod550.data_vars else None
    # Sanity checks for input coords
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    if lats.size == 0 or lons.size == 0:
        logging.warning("Empty lats/lons passed to calc_aod; returning NaNs")
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

    # Align longitude convention between ds and requested points (0-360 vs -180..180)
    ds_lon = ds_aod550.longitude
    try:
        ds_lon_min = float(ds_lon.min())
        ds_lon_max = float(ds_lon.max())
    except Exception:
        ds_lon_min, ds_lon_max = None, None

    # If dataset longitudes are 0..360 but requested lons are negative, convert
    if ds_lon_min is not None and ds_lon_min >= 0 and np.any(lons < 0):
        lons = np.where(lons < 0, lons + 360.0, lons)
        logging.debug("Converted negative longitudes to 0-360 to match dataset convention")

    def _safe_mean(arr, context=""):
        arr = np.asarray(arr)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            logging.info(f"No valid AOD samples for {context}; returning NaN")
            return np.nan
        return float(valid.mean())

    # If dataset has a time dimension, compute spatial mean along the azimuth for each time step
    if 'time' in da_total.dims:
        total_list = []
        dust_list = []
        for i in range(da_total.sizes['time']):
            sel_total = da_total.isel(time=i).interp(latitude=xr.DataArray(lats), longitude=xr.DataArray(lons), method='nearest').values
            mean_total = _safe_mean(sel_total, context=f"total time index {i}")
            total_list.append(mean_total)
            if da_dust is not None:
                sel_dust = da_dust.isel(time=i).interp(latitude=xr.DataArray(lats), longitude=xr.DataArray(lons), method='nearest').values
                mean_dust = _safe_mean(sel_dust, context=f"dust time index {i}")
                dust_list.append(mean_dust)
            else:
                dust_list.append(np.nan)
        total_aod550 = np.array(total_list)
        dust_aod550 = np.array(dust_list)
    else:
        total_vals = da_total.interp(latitude=xr.DataArray(lats), longitude=xr.DataArray(lons), method='nearest').values
        total_aod550 = np.array([_safe_mean(total_vals, context="total (no time)")])
        if da_dust is not None:
            dust_vals = da_dust.interp(latitude=xr.DataArray(lats), longitude=xr.DataArray(lons), method='nearest').values
            dust_aod550 = np.array([_safe_mean(dust_vals, context="dust (no time)")])
        else:
            dust_aod550 = np.array([np.nan])

    # ratio per time
    with np.errstate(invalid='ignore', divide='ignore'):
        dust_aod550_ratio = dust_aod550 / total_aod550

    return dust_aod550, total_aod550, dust_aod550_ratio

