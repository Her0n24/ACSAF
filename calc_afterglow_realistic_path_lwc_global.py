"""
Project ACSAF: Aerosols & Cloud geometry based Sunset/Sunrise cloud Afterglow forecaster v2
Cloud and Aerosol Trasmittance Physical Model with realistic ray path
This script uses the ECMWF AIFS cloud cover data and CAMS AOD550 data to visualize the cloud cover maps and calculate various parameters related to afterglow.

Data availability (HH:MM)
CAMS Global analyses and forecast_hour:

00 UTC forecast_hours data availability guaranteed by 10:00 UTC

12 UTC forecast_hours data availability guaranteed by 22:00 UTC
Author: A350XWBoy
"""
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import datetime
import time
import traceback
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import inspect
from astral.sun import sun, elevation, azimuth
from astral import LocationInfo
import pytz
import cfgrib
from get_aifs import download_file
from get_cds_global import get_cams_aod_lwc, get_cams_cloud_cover
from geopy import Nominatim
import logging 
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib.xarray_plugin")
warnings.filterwarnings("ignore", category=UserWarning, module="cfgrib.messages")
logging.basicConfig(
    level=logging.WARNING,
    datefmt= '%Y-%m-%d %H:%M:%S',
                    )
from calc_aod_global import calc_aod
from CONST import (
    MAX_CLOUD_HEIGHT,
    VIEW_ELEVATION_ANGLE,
    TIMESTEP_ARRAY,
    ALPHA_COEFF,
    LCC_HMIN,
    LCC_HMAX,
    MCC_HMIN,
    MCC_HMAX,
    HCC_HMIN,
    HCC_HMAX,
    LCC_HEIGHT,
    MCC_HEIGHT,
    HCC_HEIGHT,
    DESIRED_NUM_POINTS,
    TAU_EFF_MAP,
    H_AERO_KM_DEFAULT,
    DEFAULT_AZIMUTH_LINE_DISTANCE_KM,
    ASSUMED_DISTANCE_BELOW_THRESHOLD_KM,
    DEBUG_TS,
    I_RAY_THRESHOLD_DEFAULT,
    R_EARTH_M,
    RAY_NORM_CONSANT,
    MIN_CLOUD_COVER_THRESHOLD,
    MAX_LCC_THRESHOLD,
    MAX_MCC_THRESHOLD, 
    R_DRY,
    T0,
    P0,
    L,
    R_CLOUDDROP,
    R_ICECRYSTAL,
    RHO_WATER,
    RHO_ICE,
    RAY_NORM_CONSANT_LWC
)
import argparse
from functools import lru_cache

# Derived runtime values
# `TIMESTEP_ARRAY` and `ALPHA_COEFF` are imported from CONST; compute `alpha` here
m = MAX_CLOUD_HEIGHT / (np.arctan(VIEW_ELEVATION_ANGLE * np.pi / 180.0))
t = TIMESTEP_ARRAY
alpha = ALPHA_COEFF * t

def parse_args():
    parser = argparse.ArgumentParser(description="Afterglow forecast")
    parser.add_argument('--date', type=str, default=None,
                        help="Specify the date in YYYYMMDD format (default: today)")
    parser.add_argument('--run', type=str, default=None,
                        help="Optional forecast run hour (e.g. 00 or 12). Defaults to the latest available run")
    parser.add_argument('--workers', type=int, default=2,
                        help="Maximum number of parallel city workers (default: 2 on Raspberry Pi)")
    return parser.parse_args()

output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Afterglow', 'output'))
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Afterglow', 'input'))

logging.info(f"Input path: {input_path}")
logging.info(f"Output path: {output_path}")

def _indexpath(grib_path: str) -> str:
    # Reuse ecCodes index to speed repeated opens
    return f"{grib_path}.idx"

def _format_local_time(dt_value, tz):
    if dt_value is None or tz is None:
        return None
    try:
        return dt_value.astimezone(tz).isoformat()
    except Exception as exc:
        logging.warning(f"Failed to convert {dt_value} to timezone {tz}: {exc}")
        try:
            return dt_value.isoformat()
        except Exception:
            return None

@lru_cache(maxsize=None)
def open_plev(grib_path: str) -> xr.Dataset:
    return xr.open_dataset(
        grib_path,
        engine='cfgrib',
        backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'},
            'indexpath': _indexpath(grib_path),
        },
        decode_timedelta=True,
    )

@lru_cache(maxsize=None)
def open_cloud_da(grib_path: str, short_name: str) -> xr.DataArray:
    ds = xr.open_dataset(
        grib_path,
        engine='cfgrib',
        backend_kwargs={
            'filter_by_keys': {'shortName': short_name},
            'indexpath': _indexpath(grib_path),
        },
        decode_timedelta=True,
    )
    return ds[short_name]

def convert_pressure_coordinate_to_height(da: xr.Dataset) -> xr.Dataset:
    # Your defined mapping
    pressure_to_height = {
        1000: 111, 925: 762, 850: 1500, 700: 3000, 
        600: 4000, 500: 5500, 400: 7500, 300: 10000, 
        250: 12000, 200: 15000, 150: 20000, 100: 30000, 50: 50000
    }
    
    # 1. Identify the vertical dimension name 'isobaricInhPa'
    dim_name = 'isobaricInhPa' 
    
    if dim_name not in da.dims:
        raise ValueError(f"Dimension '{dim_name}' not found in DataArray. "
                         f"Available dims: {list(da.dims)}")

    # 2. Map the old pressure values to the new height values
    # We use .values to get the array, and list comprehension for the lookup
    try:
        new_coords = [pressure_to_height[int(p)] for p in da[dim_name].values]
    except KeyError as e:
        raise KeyError(f"Pressure level {e} found in data but missing from your dictionary!")

    # 3. Assign the new values and rename the dimension to 'height'
    da_new = da.assign_coords({dim_name: new_coords})
    da_new = da_new.rename({dim_name: 'height'})
    
    # Optional: Add metadata
    da_new.height.attrs['units'] = 'm'
    da_new.height.attrs['long_name'] = 'height'

    return da_new

def combine_cloud_layers(grib_path: str, fcst_hr: int | None = None) -> xr.Dataset:
    """
    Load lcc, mcc, hcc and return an xr.Dataset with variables `lcc`, `mcc`, `hcc`.

    This avoids carrying the original DataArray name/GRIB attrs from the first
    element when concatenating and makes per-layer selection unambiguous.
    The function still squeezes non-spatial coords and wraps longitudes to
    -180..180 for consistent ROI slicing.
    """
    def _select_step_if_needed(da: xr.DataArray, fh: int | None) -> xr.DataArray:
        if fh is None:
            return da

        # Prefer selecting on 'step' coordinate if present
        if 'step' in da.coords:
            step_coord = da['step']
            try:
                desired = np.timedelta64(int(fh), 'h')
                if np.issubdtype(step_coord.dtype, np.timedelta64):
                    if desired in step_coord.values:
                        return da.sel(step=desired)
                    step_hours = (step_coord / np.timedelta64(1, 'h')).astype(float)
                    idx = int(np.nanargmin(np.abs(step_hours - float(fh))))
                    return da.isel(step=idx)
                else:
                    try:
                        return da.sel(step=int(fh))
                    except Exception:
                        step_vals = np.asarray(step_coord, dtype=float)
                        idx = int(np.nanargmin(np.abs(step_vals - float(fh))))
                        return da.isel(step=idx)
            except Exception:
                try:
                    if np.issubdtype(step_coord.dtype, np.timedelta64):
                        step_hours = (step_coord / np.timedelta64(1, 'h')).astype(float)
                        idx = int(np.nanargmin(np.abs(step_hours - float(fh))))
                        return da.isel(step=idx)
                except Exception:
                    pass

        for coord in ('valid_time', 'time'):
            if coord in da.coords:
                try: return da
                except Exception: continue

        return da

    lcc = _select_step_if_needed(open_cloud_da(grib_path, 'lcc'), fcst_hr)
    mcc = _select_step_if_needed(open_cloud_da(grib_path, 'mcc'), fcst_hr)
    hcc = _select_step_if_needed(open_cloud_da(grib_path, 'hcc'), fcst_hr)

    # Remove leftover time/step coordinates and WRAP LONGITUDE
    def _squeeze_and_wrap(da: xr.DataArray) -> xr.DataArray:
        # drop common non-spatial coords
        da = da.drop_vars(['time', 'valid_time'], errors='ignore')
        try:
            da = da.squeeze(drop=True)
        except Exception:
            pass
            
        # --- NEW: Convert 0-360 to -180 to 180 and sort ---
        if 'longitude' in da.coords:
            da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
            # Sorting is critical after wrapping so Xarray can slice it properly
            da = da.sortby(['latitude', 'longitude'])
            
        return da

    lcc = _squeeze_and_wrap(lcc)
    mcc = _squeeze_and_wrap(mcc)
    hcc = _squeeze_and_wrap(hcc)

    # Ensure each layer is a clean DataArray and drop time/step coords
    lcc = lcc.drop_vars(['time', 'step', 'valid_time'], errors='ignore').squeeze(drop=True)
    mcc = mcc.drop_vars(['time', 'step', 'valid_time'], errors='ignore').squeeze(drop=True)
    hcc = hcc.drop_vars(['time', 'step', 'valid_time'], errors='ignore').squeeze(drop=True)

    # Build a dataset so variable names reflect the layer names naturally
    cloud_ds = xr.Dataset({'lcc': lcc, 'mcc': mcc, 'hcc': hcc})
    return cloud_ds

def open_cams_data_and_blend_clouds(grib_path: str, cloud_da: xr.DataArray | xr.Dataset, fcst_hr: str) -> xr.Dataset:
    """
    Opens CAMS data for the specific forecast hour.
    Note: Spatial blending with cloud_da has been removed to preserve the native 
    CAMS 3D grid, preventing NaN artifacts and double-interpolation.
    """
    desired_time_step = np.timedelta64(int(fcst_hr), 'h')
    
    def preprocess(da):
        # 1. Select the correct time step
        da = da.sel(step=desired_time_step)
        # 2. Drop time coordinates so they don't cause dimension conflicts later
        # We leave latitude and longitude COMPLETELY ALONE.
        return da.drop_vars(['time', 'step', 'valid_time'], errors='ignore')

    # Load the pure variables
    clwc = preprocess(open_cloud_da(grib_path, 'clwc'))
    ciwc = preprocess(open_cloud_da(grib_path, 'ciwc'))
    aec = preprocess(open_cloud_da(grib_path, 'aerext532'))
    
    # Combine into a single Dataset on the native CAMS grid
    cams_ds_native = xr.Dataset({
        'clwc': clwc,
        'ciwc': ciwc,
        'aerext532': aec
    })
    
    return cams_ds_native


@lru_cache(maxsize=None)
def open_2m_cached(grib_path: str) -> xr.Dataset:
    # load_2m_fields already uses decode_timedelta=True
    return load_2m_fields(grib_path)


def build_profile_payload(distance_axis, profiles):
    payload = {"distance_km": list(distance_axis or [])}
    profiles = profiles or {}
    for layer in ["lcc", "mcc", "hcc"]:
        payload[layer] = list(profiles.get(layer, []))
    return payload

def _slice_roi(da: xr.DataArray, lon: float, lat: float, azimuth_deg: float | None,
               distance_km: float = 500, pad_km: float = 80) -> xr.DataArray:
    # Compute bounding box that contains the city and the end of the azimuth line
    deg_len = 1.0 / 111.0
    L = distance_km * deg_len
    pad = pad_km * deg_len
    if azimuth_deg is None:
        return da
    az = np.deg2rad(azimuth_deg)
    end_lon = lon + L * np.sin(az)
    end_lat = lat + L * np.cos(az)
    lon_min, lon_max = min(lon, end_lon) - pad, max(lon, end_lon) + pad
    lat_min, lat_max = min(lat, end_lat) - pad, max(lat, end_lat) + pad

    # Handle ascending/descending latitude
    try:
        lat0 = float(da.latitude[0])
        latN = float(da.latitude[-1])
        lat_slice = slice(lat_max, lat_min) if lat0 > latN else slice(lat_min, lat_max)
        return da.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    except Exception:
        return da  # fallback if coordinates differ
    
def reduce_clouds_to_roi(data_dict: dict, lon: float, lat: float, azimuth_deg: float | None,
                         distance_km: float = 500, pad_km: float = 80) -> dict:
    return {
        k: _slice_roi(v, lon, lat, azimuth_deg, distance_km, pad_km)
        for k, v in data_dict.items()
        if v is not None
    }

def max_solar_elevation(city, date):
    tz = pytz.timezone(city.timezone)
    observer = city.observer

    # Sample every 5 minutes across the day
    times = [datetime.datetime.combine(date, datetime.datetime.min.time()) + datetime.timedelta(minutes=5*i) for i in range(288)]
    times = [tz.localize(t) for t in times]

    max_angle = max(elevation(observer, t) for t in times)
    return max_angle

#function to get the azimuth angle at sunset
def get__sunset_azimuth(city, date):
    tz = pytz.timezone(city.timezone)
    observer = city.observer

    # Get the sunset time
    s = sun(observer, date=date, tzinfo=tz)
    sunset_time = s['sunset']

    # Calculate the azimuth angle at sunset
    azimuth_angle = azimuth(observer, sunset_time)

    return azimuth_angle

def get__sunrise_azimuth(city, date):
    tz = pytz.timezone(city.timezone)
    observer = city.observer

    # Get the sunrise time
    s = sun(observer, date=date, tzinfo=tz)
    sunrise_time = s['sunrise']

    # Calculate the azimuth angle at sunrise
    azimuth_angle = azimuth(observer, sunrise_time)

    return azimuth_angle

def safe_distance(h):
    # Equation to calculate the safe distance in km
    l1 = 2*np.arccos(6371/((6371+h)))*360*6371*2*np.pi
    return l1

def calc_afterglow_time(h, max_elev, l1):
    # Equation to calculate the time of afterglow in km and minutes
    t1 = np.arccos(6371/((6371+h)/360))*24*60/np.sin(max_elev.radians())
    t2 = l1/(6371*2*np.pi*24*60/np.sin(max_elev.radians()))
    z = t1 + t2
    return z

def parabolic_ray_equation(x, m, alpha, r, H):
    # Parabolic ray equation that depicts the actual cloud ray path height varying with distance assuming local flat 
    # Earth surface frame.
    # alpha = -5.14*10^5 times t
    return (x-m)*np.tan(alpha) + (0.5*(x-m)**2)/r + H

def total_transmitted_light(cloud_cover_profile):
    return np.prod(1 - cloud_cover_profile/100)

def extract_variable(ds, var_name, lat_min, lat_max, lon_min, lon_max, verbose=False):
    var = getattr(ds, var_name)
    var = var.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).squeeze()
    if verbose:
        logging.info(f"{var_name}:")
        logging.info(var.values)
    return var


def azimuth_line_points(lon, lat, azimuth, distance_km, num_points=DESIRED_NUM_POINTS):
    """
    Generate a series of points along a given azimuth line (in degrees) starting from the given coordinates.
    
    Parameters:
    - lon: Longitude of the starting point
    - lat: Latitude of the starting point
    - azimuth: Azimuth angle in degrees (0° is north, 90° is east, 180° is south, 270° is west)
    - distance_km: Distance to generate points (in km)
    - num_points: Number of points to generate along the azimuth line
    
    Returns:
    - lons, lats: Arrays of longitudes and latitudes along the azimuth line
    
    Generated using GPT
    
    """
    # Earth radius in km
    earth_radius = 6371.0

    # Convert azimuth to radians
    azimuth_rad = np.deg2rad(azimuth)

    # Generate points along the azimuth line
    dists = np.linspace(0, distance_km, num_points)
    lons = lon + (dists / earth_radius) * np.sin(azimuth_rad) * (180 / np.pi)
    lats = lat + (dists / earth_radius) * np.cos(azimuth_rad) * (180 / np.pi)

    return lons, lats

def extract_cloud_cover_along_azimuth(data_dict, lon, lat, azimuth, distance_km, num_points=DESIRED_NUM_POINTS):
    """
    Extract cloud cover data along an azimuth line over a certain distance.
    
    Parameters:
    - data_dict: DataArray or dictionary containing cloud cover data ("lcc", "mcc", "hcc")
    - lon, lat: Starting coordinates of the city
    - azimuth: Azimuth angle in degrees
    - distance_km: Distance along the azimuth line (in km)
    - num_points: Number of points to sample along the azimuth line
    
    Returns:
    - cloud_cover_data: 1D array with cloud cover values along the azimuth line
    """
    # Generate azimuth line points
    lons, lats = azimuth_line_points(lon, lat, azimuth, distance_km, num_points)
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    # Try to align ray longitudes to the dataset longitude grid by choosing
    # the periodic candidate (lon, lon+360, lon-360) that is closest to the
    # dataset grid (min distance to any dataset lon). This helps when the
    # dataset uses a different 0/360 convention or the ROI was sliced.
    try:
        data_lons = np.asarray(data_dict.longitude)
        lons_adj = np.empty_like(lons, dtype=float)
        for i, lon_val in enumerate(lons):
            candidates = np.array([lon_val, lon_val + 360.0, lon_val - 360.0])
            try:
                # distance from each candidate to the nearest data lon
                dists = np.nanmin(np.abs(candidates[:, None] - data_lons[None, :]), axis=1)
            except Exception:
                data_mid = 0.5 * (float(np.nanmin(data_lons)) + float(np.nanmax(data_lons)))
                dists = np.abs(candidates - data_mid)
            lons_adj[i] = float(candidates[int(np.nanargmin(dists))])
    except Exception:
        lons_adj = lons

    ray_index = 'ray_index'
    lons_da = xr.DataArray(lons_adj, dims=ray_index)
    lats_da = xr.DataArray(lats, dims=ray_index)

    # 1) Preferred: vectorized nearest selection (robust for out-of-range coords)
    def _reduce_to_1d(vals, n_points):
        vals = np.asarray(vals)
        # Scalar
        if vals.ndim == 0:
            out = np.full(n_points, float(vals)) if n_points > 1 else np.array([float(vals)])
            return out.astype(float)
        # 1D: try to pad/truncate to n_points
        if vals.ndim == 1:
            flat = vals.reshape(-1)
            if flat.size == n_points:
                return flat.astype(float)
            if flat.size > n_points:
                return flat[:n_points].astype(float)
            out = np.full(n_points, np.nan)
            out[:flat.size] = flat
            return out.astype(float)
        # 2D or higher: try to detect which axis corresponds to points and reduce the other
        if vals.ndim >= 2:
            # square diag case
            if vals.shape[0] == vals.shape[1] == n_points:
                return np.diag(vals).astype(float)
            # if axis 0 corresponds to points, average over other axes
            if vals.shape[0] == n_points:
                return np.nanmean(vals, axis=tuple(range(1, vals.ndim))).astype(float)
            # if axis 1 corresponds to points, average over other axes
            if vals.shape[1] == n_points:
                return np.nanmean(vals, axis=tuple([i for i in range(vals.ndim) if i != 1])).astype(float)
            # try mean over axis 0 then axis 1 as fallback
            out0 = np.nanmean(vals, axis=0)
            if out0.size == n_points:
                return out0.astype(float)
            out1 = np.nanmean(vals, axis=1)
            if out1.size == n_points:
                return out1.astype(float)
            # last resort: flatten and slice/pad
            flat = vals.reshape(-1)
            if flat.size >= n_points:
                return flat[:n_points].astype(float)
            out = np.full(n_points, np.nan)
            out[:flat.size] = flat
            return out.astype(float)

    try:
        sel = data_dict.sel(longitude=lons_da, latitude=lats_da, method='nearest')
        vals = np.asarray(sel.values)
        if vals.size != 0 and not np.all(np.isnan(vals)):
            return _reduce_to_1d(vals, len(lons))
    except Exception:
        logging.debug("extract_cloud_cover_along_azimuth: vectorized sel(method='nearest') failed; will try interp/fallback")

    # 2) Try vectorized interp as a second attempt
    try:
        interp = data_dict.interp(longitude=lons_da, latitude=lats_da, method='nearest')
        vals = np.asarray(interp.values)
        if vals.size != 0 and not np.all(np.isnan(vals)):
            return _reduce_to_1d(vals, len(lons))
    except Exception:
        logging.debug("extract_cloud_cover_along_azimuth: vectorized interp failed; falling back to per-point nearest")

    # 3) Fallback: per-point nearest selection/interpolation (slower, robust)
    out = np.full(len(lons), np.nan)
    for i, (lo, la) in enumerate(zip(lons_adj, lats)):
        try:
            try:
                sel = data_dict.sel(longitude=float(lo), latitude=float(la), method='nearest')
                v = sel.values
            except Exception:
                try:
                    v = data_dict.interp(longitude=float(lo), latitude=float(la), method='nearest').values
                except Exception:
                    v = np.nan

            v_arr = np.asarray(v)
            if v_arr.size == 0:
                out[i] = np.nan
            elif v_arr.size == 1:
                out[i] = float(v_arr.item())
            else:
                out[i] = float(np.nanmean(v_arr))
        except Exception:
            out[i] = np.nan
    return out


def extract_cloud_cover_with_ray_tracing(cloud_layers_da, lons, lats, distances_m, cloud_base_lvl, R_earth_m=R_EARTH_M, fcst_hr: int | None = None):
    """
    Extract cloud cover along an azimuth considering parabolic ray paths at multiple timesteps.
    
    For each distance along azimuth and each timestep (alpha value), calculates the ray height
    using parabolic_ray_equation and determines which cloud layer (lcc, mcc, hcc) the ray passes through.
    
    Parameters:
    - cloud_layers_da: Combined cloud layers DataArray with dims (lvl, latitude, longitude)
                       where lvl = ['lcc', 'mcc', 'hcc']
    - lons, lats: Arrays of longitudes/latitudes along the azimuth
    - distances_m: Array of distances in meters along the azimuth
    - cloud_base_lvl: Reference cloud base height (m) for ray equation
    - R_earth_m: Earth radius in meters (default: 6.371e6)
    
    Returns:
    - ray_cloud_profile: 2D array with dims (num_timesteps, num_points)
                        Cloud cover values selected based on ray height at each distance/time
    - ray_heights: 2D array with dims (num_timesteps, num_points)
                  Ray heights from parabolic equation at each distance/time
    """
    global alpha, m
    
    num_points = len(distances_m)
    num_timesteps = len(alpha)
    
    # Compute radius of curvature for ray path
    r = R_earth_m  # approximate ray curvature radius
    
    ray_cloud_profile = np.full((num_timesteps, num_points), np.nan)
    ray_heights = np.full((num_timesteps, num_points), np.nan)
    
    # Note: do not interpolate here — we select forecast-hour slices and
    # perform interpolation later after ensuring the correct fcst snapshot
    # has been chosen. Early interpolation could preserve a time dimension
    # and cause shape mismatches.
    
    # Height boundaries for layer selection
    layer_bounds = {
        'lcc': (LCC_HMIN, LCC_HMAX),
        'mcc': (MCC_HMIN, MCC_HMAX),
        'hcc': (HCC_HMIN, HCC_HMAX),
    }
    
    # (removed interactive breakpoint)
    # Vectorized computation of ray heights and cloud extraction
    try:
        # Compute ray heights for all timesteps and distances at once
        # distances_m: (N,), alpha: (T,) -> broadcast to (T, N)
        d_arr = np.asarray(distances_m)
        ray_heights_calc = parabolic_ray_equation(d_arr[None, :], 0, alpha[:, None], r, cloud_base_lvl)

        # Interpolate cloud layer values at the ray horizontal points in a pointwise manner
        ray_index = 'ray_index'
        lons_da = xr.DataArray(lons, dims=ray_index)
        lats_da = xr.DataArray(lats, dims=ray_index)
        cloud_interp = cloud_layers_da.interp(longitude=lons_da, latitude=lats_da, method='nearest')

        # Extract per-layer arrays and coerce to 2D (timesteps x points).
        def _get_layer(arr, name):
            if isinstance(arr, xr.Dataset):
                return arr[name]
            return arr.sel(lvl=name)

        num_timesteps, num_points = ray_heights_calc.shape

        def _resample_time_array(arr2d, target_len):
            # arr2d: shape (old_t, npoints) -> return shape (target_len, npoints)
            old_t, npts = arr2d.shape
            if old_t == target_len:
                return arr2d
            if npts == 0:
                return np.full((target_len, npts), np.nan)
            out = np.full((target_len, npts), np.nan)
            old_times = np.arange(old_t)
            new_times = np.linspace(0, old_t - 1, target_len)
            for j in range(npts):
                col = arr2d[:, j]
                mask = ~np.isnan(col)
                if mask.sum() == 0:
                    continue
                if mask.sum() == 1:
                    out[:, j] = col[mask][0]
                    continue
                try:
                    out[:, j] = np.interp(new_times, old_times[mask], col[mask])
                except Exception:
                    out[:, j] = np.nan
            return out

        def _to_2d_for_broadcast(layer_obj, layer_name):
            """Return a 2D numpy array shaped (num_timesteps, num_points).

            This function is defensive: it accepts xr.DataArray / Dataset slices or
            plain numpy arrays and handles common mismatches by resampling in
            time or aggregating as necessary. For forecast-hour slices (fcst
            snapshots) we intentionally repeat the chosen fcst slice across the
            ray timesteps (each ray timestep is a 60s interval). This preserves
            the semantics where the GRIB time dimension represents forecast
            hour rather than per-ray timesteps.
            """
            # If this is an xarray object and a forecast-hour was provided,
            # try selecting the matching 'step'/'time'/'forecast_hour' slice
            # before converting to a numpy array. This preserves the GRIB
            # semantics where the time dimension is forecast-hour (fcst_hr)
            # and ray timesteps are per-ray 60s slices.
            try:
                if hasattr(layer_obj, 'coords') and fcst_hr is not None:
                    try:
                        if 'step' in layer_obj.coords:
                            step_coord = layer_obj['step']
                            desired = np.timedelta64(int(fcst_hr), 'h')
                            if np.issubdtype(step_coord.dtype, np.timedelta64):
                                if desired in step_coord.values:
                                    layer_obj = layer_obj.sel(step=desired)
                                else:
                                    step_hours = (step_coord / np.timedelta64(1, 'h')).astype(float)
                                    idx = int(np.nanargmin(np.abs(step_hours - float(fcst_hr))))
                                    layer_obj = layer_obj.isel(step=idx)
                            else:
                                try:
                                    layer_obj = layer_obj.sel(step=int(fcst_hr))
                                except Exception:
                                    step_vals = np.asarray(step_coord, dtype=float)
                                    idx = int(np.nanargmin(np.abs(step_vals - float(fcst_hr))))
                                    layer_obj = layer_obj.isel(step=idx)
                        elif 'time' in layer_obj.coords:
                            # If there's a single time then pick it; otherwise
                            # if fcst_hr looks like an index and falls within
                            # bounds, use it as an index.
                            if layer_obj.sizes.get('time', 0) == 1:
                                layer_obj = layer_obj.isel(time=0)
                            elif isinstance(fcst_hr, (int, np.integer)) and fcst_hr < layer_obj.sizes.get('time', 0):
                                layer_obj = layer_obj.isel(time=int(fcst_hr))
                        elif 'forecast_hour' in layer_obj.coords:
                            try:
                                layer_obj = layer_obj.sel(forecast_hour=int(fcst_hr))
                            except Exception:
                                if layer_obj.sizes.get('forecast_hour', 0) > 0:
                                    idx = min(int(fcst_hr), layer_obj.sizes.get('forecast_hour', 0) - 1)
                                    layer_obj = layer_obj.isel(forecast_hour=idx)
                    except Exception:
                        logging.debug(f"_to_2d_for_broadcast: fcst selection failed for {layer_name}")

                if hasattr(layer_obj, 'values'):
                    arr_np = np.asarray(layer_obj.values, dtype=float)
                else:
                    arr_np = np.asarray(layer_obj, dtype=float)
            except Exception:
                logging.debug(f"_to_2d_for_broadcast: failed to convert {layer_name} to ndarray; filling NaNs")
                return np.full((num_timesteps, num_points), np.nan)

            logging.debug(f"_to_2d_for_broadcast: layer={layer_name} raw_shape={getattr(arr_np,'shape',None)} dtype={getattr(arr_np,'dtype',None)}")

            # 1D -> broadcast across timesteps
            if arr_np.ndim == 1:
                if arr_np.size == num_points:
                    logging.debug(f"_to_2d_for_broadcast: layer={layer_name} using 1D->broadcast")
                    return np.broadcast_to(arr_np[None, :], (num_timesteps, num_points))
                # If 1D but length differs, try to interpolate/resize
                try:
                    arr_1d = np.asarray(arr_np, dtype=float)
                    # simple resize by interpolation
                    old_idx = np.linspace(0, 1, arr_1d.size)
                    new_idx = np.linspace(0, 1, num_points)
                    resized = np.interp(new_idx, old_idx, arr_1d)
                    return np.broadcast_to(resized[None, :], (num_timesteps, num_points))
                except Exception:
                    return np.full((num_timesteps, num_points), np.nan)

            # 2D handling
            if arr_np.ndim == 2:
                r0, r1 = arr_np.shape
                # Common case: (time, points)
                if r1 == num_points:
                    # If time length matches target, use directly
                    if r0 == num_timesteps:
                        logging.debug(f"_to_2d_for_broadcast: layer={layer_name} already matches (time,points)")
                        return arr_np
                    # If this is a single forecast-hour slice, repeat it across
                    # the ray timesteps (ray timesteps are 60s each per event).
                    if r0 == 1:
                        logging.debug(f"_to_2d_for_broadcast: layer={layer_name} repeating single forecast slice across {num_timesteps} timesteps")
                        return np.broadcast_to(arr_np[0][None, :], (num_timesteps, num_points))
                    # If multiple forecast slices are present, prefer explicit
                    # selection upstream; fall back to using the first slice and
                    # repeating it while warning the user.
                    logging.warning(f"_to_2d_for_broadcast: layer={layer_name} has {r0} forecast slices != ray timesteps {num_timesteps}; using first slice and repeating across timesteps")
                    return np.broadcast_to(arr_np[0][None, :], (num_timesteps, num_points))
                # If transposed (points, time)
                if r0 == num_points and r1 == num_timesteps:
                    logging.debug(f"_to_2d_for_broadcast: layer={layer_name} transposing (points,time)->(time,points)")
                    return arr_np.T
                # If one axis matches points, move it to last and reshape
                axes = [i for i, s in enumerate(arr_np.shape) if s == num_points]
                if axes:
                    axis = axes[-1]
                    logging.debug(f"_to_2d_for_broadcast: layer={layer_name} found points axis at {axis}; collapsing and reshaping")
                    moved = np.moveaxis(arr_np, axis, -1)
                    reshaped = moved.reshape(-1, num_points)
                    return _resample_time_array(reshaped, num_timesteps)

                # Fallback: aggregate across first axis to produce per-point means
                try:
                    collapsed = np.nanmean(arr_np, axis=0)
                    if collapsed.size == num_points:
                        logging.debug(f"_to_2d_for_broadcast: layer={layer_name} collapsing mean->broadcast")
                        return np.broadcast_to(collapsed[None, :], (num_timesteps, num_points))
                except Exception:
                    pass

                logging.debug(f"_to_2d_for_broadcast: layer={layer_name} fallback->all NaNs")
                return np.full((num_timesteps, num_points), np.nan)

            # Higher-dim arrays: try to find axis equal to num_points
            if arr_np.ndim > 2:
                axes = [i for i, s in enumerate(arr_np.shape) if s == num_points]
                if axes:
                    axis = axes[-1]
                    logging.debug(f"_to_2d_for_broadcast: layer={layer_name} high-dim with points axis {axis}")
                    moved = np.moveaxis(arr_np, axis, -1)
                    reshaped = moved.reshape(-1, num_points)
                    return _resample_time_array(reshaped, num_timesteps)
                logging.debug(f"_to_2d_for_broadcast: layer={layer_name} high-dim fallback->all NaNs")
                return np.full((num_timesteps, num_points), np.nan)

            return np.full((num_timesteps, num_points), np.nan)

        # Prepare 2D arrays for each layer
        cloud_lcc_2d = _to_2d_for_broadcast(_get_layer(cloud_interp, 'lcc'), 'lcc')
        cloud_mcc_2d = _to_2d_for_broadcast(_get_layer(cloud_interp, 'mcc'), 'mcc')
        cloud_hcc_2d = _to_2d_for_broadcast(_get_layer(cloud_interp, 'hcc'), 'hcc')

        # Layer masks (T, N)
        lcc_mask = (LCC_HMIN <= ray_heights_calc) & (ray_heights_calc <= LCC_HMAX)
        mcc_mask = (MCC_HMIN <= ray_heights_calc) & (ray_heights_calc <= MCC_HMAX)
        hcc_mask = (HCC_HMIN <= ray_heights_calc) & (ray_heights_calc <= HCC_HMAX)

        
        # Assign based on masks; lcc has priority then mcc then hcc
        ray_cloud_profile_calc = np.where(lcc_mask, cloud_lcc_2d,
                                         np.where(mcc_mask, cloud_mcc_2d,
                                                  np.where(hcc_mask, cloud_hcc_2d, np.nan)))

        # Any timestep that has ray height <= 0 is invalid (discard entire timestep)
        invalid_timesteps = np.any(ray_heights_calc <= 0, axis=1)
        if np.any(invalid_timesteps):
            ray_cloud_profile_calc[invalid_timesteps, :] = np.nan
            ray_heights_calc[invalid_timesteps, :] = np.nan

        # Store results into the outputs
        ray_heights[:, :] = ray_heights_calc
        ray_cloud_profile[:, :] = ray_cloud_profile_calc

    except Exception as e:
        # Fall back to original iterative implementation if vectorized path fails
        logging.warning(f"Vectorized ray tracing failed, falling back to iterative method: {e}")
        for t_idx, alpha_val in enumerate(alpha):
            # First pass: calculate all ray heights for this timestep
            timestep_ray_heights = np.full(len(distances_m), np.nan)
            try:
                for d_idx, dist_m in enumerate(distances_m):
                    try:
                        ray_h = parabolic_ray_equation(dist_m, 0, alpha_val, r, cloud_base_lvl)
                        timestep_ray_heights[d_idx] = ray_h
                    except Exception as e2:
                        logging.warning(f"Ray calculation failed at distance {dist_m}m, timestep {t_idx}: {e2}")
                        continue

                # Check if any point in this ray path goes below or touches ground
                if np.any(timestep_ray_heights <= 0):
                    ray_heights[t_idx, :] = np.nan
                    ray_cloud_profile[t_idx, :] = np.nan
                    continue

                ray_heights[t_idx, :] = timestep_ray_heights

                # Second pass: extract cloud cover for valid ray path
                for d_idx, dist_m in enumerate(distances_m):
                    ray_h = timestep_ray_heights[d_idx]
                    selected_layer = None
                    for layer_name, (h_min, h_max) in layer_bounds.items():
                        if h_min <= ray_h <= h_max:
                            selected_layer = layer_name
                            break
                    if selected_layer is None:
                        continue
                    try:
                        # Try to extract the per-point value in a way that handles
                        # different layouts produced by xarray interpolation:
                        # - preferred: interpolation produced a 'ray_index' dim
                        # - common: interpolation produced latitude/longitude dims
                        # - fallback: query the original cloud_layers_da with nearest

                        # Support both Dataset (vars 'lcc','mcc','hcc') and older
                        # DataArray with 'lvl' coordinate
                        if isinstance(cloud_interp, xr.Dataset):
                            arr = cloud_interp[selected_layer]
                        else:
                            arr = cloud_interp.sel(lvl=selected_layer)

                        if 'ray_index' in getattr(arr, 'dims', ()):  # directly indexed along ray
                            v = arr.isel(ray_index=d_idx).values

                        elif 'latitude' in getattr(arr, 'dims', ()) and 'longitude' in getattr(arr, 'dims', ()):  # 2D grid
                            try:
                                v = arr.sel(latitude=float(lats[d_idx]), longitude=float(lons[d_idx]), method='nearest').values
                            except Exception:
                                lat_coords = np.asarray(arr.coords['latitude'])
                                lon_coords = np.asarray(arr.coords['longitude'])
                                lat_idx = int(np.nanargmin(np.abs(lat_coords - float(lats[d_idx]))))
                                lon_idx = int(np.nanargmin(np.abs(lon_coords - float(lons[d_idx]))))
                                v = arr.isel(latitude=lat_idx, longitude=lon_idx).values

                        elif getattr(arr, 'ndim', 0) == 1:  # 1D array along points
                            try:
                                v = arr[d_idx].values
                            except Exception:
                                v = np.asarray(arr)[d_idx]

                        else:
                            # Last-resort: query the original combined source using the same
                            # branching logic so both Dataset and DataArray are supported.
                            if isinstance(cloud_layers_da, xr.Dataset):
                                v = cloud_layers_da[selected_layer].sel(latitude=float(lats[d_idx]), longitude=float(lons[d_idx]), method='nearest').values
                            else:
                                v = cloud_layers_da.sel(lvl=selected_layer, latitude=float(lats[d_idx]), longitude=float(lons[d_idx]), method='nearest').values

                        v_arr = np.asarray(v)
                        if v_arr.size == 0:
                            ray_cloud_profile[t_idx, d_idx] = np.nan
                        else:
                            ray_cloud_profile[t_idx, d_idx] = float(v_arr.reshape(-1)[0])

                    except Exception as e2:
                        logging.warning(f"Cloud extraction failed at distance {dist_m}m, layer {selected_layer}: {e2}")
            except Exception as e3:
                logging.warning(f"Timestep {t_idx} processing failed: {e3}")
                ray_heights[t_idx, :] = np.nan
                ray_cloud_profile[t_idx, :] = np.nan
    
    return ray_cloud_profile, ray_heights


def extract_cams_variable_along_ray(cams_ds: xr.Dataset, lons, lats, ray_heights, vertical_coord='height') -> dict:
    """
    Interpolate CAMS variables along a ray path. 
    Handles the 360->0 longitude seam by padding the dataset.
    """
    # --- STEP 1: FIX THE DATASET SEAM (The Circular Wrap) ---
    # We take the slice at 0.0 and append it at 360.0 to allow interpolation 
    # between 359.6 and 0.0.
    if 'longitude' in cams_ds.coords:
        try:
            # Check if we need to close the circle
            lon_max = float(cams_ds.longitude.max())
            if lon_max < 360 and lon_max > 350:
                # Take the first slice (at 0.0)
                first_slice = cams_ds.isel(longitude=0).copy()
                # Rename its coordinate to 360.0 (or whatever is exactly 360 degrees away)
                first_slice.coords['longitude'] = first_slice.coords['longitude'] + 360.0
                # Combine them
                cams_ds = xr.concat([cams_ds, first_slice], dim='longitude')
                logging.debug("Padded CAMS dataset to handle circular longitude.")
        except Exception as e:
            logging.warning(f"Failed to pad longitude: {e}")

    num_timesteps, num_points = ray_heights.shape
    cams_along_ray = {}
    sample_dim = 'sample'

    # Normalize lats/lons to 2D
    if np.ndim(lats) == 1:
        lats_2d = np.repeat(lats[np.newaxis, :], num_timesteps, axis=0)
    else:
        lats_2d = lats
    if np.ndim(lons) == 1:
        lons_2d = np.repeat(lons[np.newaxis, :], num_timesteps, axis=0)
    else:
        lons_2d = lons

    # Flatten coordinates
    heights_flat = np.asarray(ray_heights).reshape(-1)
    lats_flat = np.asarray(lats_2d).reshape(-1)
    lons_flat = np.asarray(lons_2d).reshape(-1)

    # Wrap longitudes to 0-360 range
    lons_safe = lons_flat % 360.0

    # Get dataset bounds for clipping (prevents floating point "out of bounds" errors)
    lat_min, lat_max = float(cams_ds.latitude.min()), float(cams_ds.latitude.max())
    lon_min, lon_max = float(cams_ds.longitude.min()), float(cams_ds.longitude.max())
    h_min, h_max = float(cams_ds[vertical_coord].min()), float(cams_ds[vertical_coord].max())

    valid_mask = (~np.isnan(heights_flat)) & (~np.isnan(lons_flat)) & (~np.isnan(lats_flat))

    for var in cams_ds.data_vars:
        # Initial output with NaNs
        cams_along_ray[var] = np.full((num_timesteps, num_points), np.nan)
        
        if not np.any(valid_mask):
            continue

        try:
            # Prepare coords for vectorized interpolation
            sample_valid = 'sample_valid'
            coords_valid = {
                vertical_coord: xr.DataArray(np.clip(heights_flat[valid_mask], h_min, h_max), dims=sample_valid),
                'latitude': xr.DataArray(np.clip(lats_flat[valid_mask], lat_min, lat_max), dims=sample_valid),
                'longitude': xr.DataArray(np.clip(lons_safe[valid_mask], lon_min, lon_max), dims=sample_valid),
            }

            # Perform interpolation
            # Use 'nearest' as a fallback if 'linear' still hits NaNs in the source data
            res_valid = cams_ds[var].interp(coords_valid, method='nearest', kwargs={"fill_value": np.nan})
            
            # Reconstruct the 2D array
            flat_out = np.full(heights_flat.shape, np.nan)
            flat_out[valid_mask] = res_valid.values.reshape(-1)
            cams_along_ray[var] = flat_out.reshape(num_timesteps, num_points)
            # Check if we still have all NaNs (this means the source data at these points is actually NaN)
            if np.isnan(cams_along_ray[var]).all():
                logging.debug(f"Vectorized {var} result is all NaN. Data likely masked in this region.")
            else:
                continue # Success, move to next variable

        except Exception as e:
            logging.debug(f"Vectorized interp failed for {var}: {e}")

        # --- FALLBACK: POINT-BY-POINT (If Vectorized Fails) ---
        for t in range(num_timesteps):
            t_h = ray_heights[t, :]
            t_lat = lats_2d[t, :]
            t_lon = lons_2d[t, :] % 360.0
            
            t_mask = ~np.isnan(t_h)
            if not np.any(t_mask): continue

            coords_t = {
                vertical_coord: xr.DataArray(np.clip(t_h[t_mask], h_min, h_max), dims='p'),
                'latitude': xr.DataArray(np.clip(t_lat[t_mask], lat_min, lat_max), dims='p'),
                'longitude': xr.DataArray(np.clip(t_lon[t_mask], lon_min, lon_max), dims='p'),
            }
            try:
                # Try nearest if linear fails to avoid source data NaNs
                res = cams_ds[var].interp(coords_t, method='nearest', kwargs={"fill_value": np.nan})
                cams_along_ray[var][t, t_mask] = res.values.reshape(-1)
            except Exception as e:
                raise RuntimeError(f"Failed to interpolate {var} at timestep {t}: {e}")
    return cams_along_ray

def aggregate_ray_traced_profile(ray_cloud_profile, aggregation='mean'):
    """
    Aggregate ray-traced cloud profiles across all timesteps into a single profile.
    
    Parameters:
    - ray_cloud_profile: 2D array with dims (num_timesteps, num_points)
    - aggregation: 'mean', 'max', or 'median' - how to combine across timesteps
    
    Returns:
    - aggregated_profile: 1D array with shape (num_points,)
    """
    if aggregation == 'mean':
        return np.nanmean(ray_cloud_profile, axis=0)
    elif aggregation == 'max':
        return np.nanmax(ray_cloud_profile, axis=0)
    elif aggregation == 'median':
        return np.nanmedian(ray_cloud_profile, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def select_layers_by_threshold(ray_cloud_profile, lcc_val, mcc_val, hcc_val) -> dict:
    """
    Select which cloud layers to analyze based on local cloud cover (first 3 indices).
    Works from lowest layer upward with thresholds: LCC 50%, MCC 60%, HCC 20%.
    If a layer exceeds threshold, include it and all layers above with lower thresholds.
    
    Parameters:
    - ray_cloud_profile: 2D array with dims (num_timesteps, num_points)
    - lcc_val, mcc_val, hcc_val: Local cloud cover percentages (first 3 indices average)
    
    Returns:
    - selected_layers: dict with layer names as keys (e.g., {'lcc': True, 'mcc': False, 'hcc': False})
    """
    # Enforce minimum threshold: if all layers are below threshold, return all False
    if all(val < MIN_CLOUD_COVER_THRESHOLD for val in [lcc_val, mcc_val, hcc_val]):
        logging.info("All cloud layers below minimum threshold; returning no cloud.")
        return {'lcc': False, 'mcc': False, 'hcc': False}
    
    selected_layers = {'lcc': False, 'mcc': False, 'hcc': False}

    # Step 1: If all layers are below threshold, return no cloud
    if all(val < MIN_CLOUD_COVER_THRESHOLD for val in [lcc_val, mcc_val, hcc_val]):
        logging.info("All cloud layers below minimum threshold; returning no cloud.")
        return selected_layers
    
    # Step 2: Search from lowest upward
    if lcc_val >= MIN_CLOUD_COVER_THRESHOLD:
        if lcc_val >= MAX_LCC_THRESHOLD:
            selected_layers['lcc'] = True
        else:
            selected_layers['lcc'] = True
            if mcc_val >= MIN_CLOUD_COVER_THRESHOLD:
                if mcc_val >= MAX_MCC_THRESHOLD:
                    selected_layers['mcc'] = True
                else:
                    selected_layers['mcc'] = True
                    if hcc_val >= MIN_CLOUD_COVER_THRESHOLD:
                        selected_layers['hcc'] = True
            # If MCC < threshold, only LCC is selected (already set)
    elif mcc_val >= MIN_CLOUD_COVER_THRESHOLD:
        if mcc_val >= MAX_MCC_THRESHOLD:
            selected_layers['mcc'] = True
        else:
            selected_layers['mcc'] = True
            if hcc_val >= MIN_CLOUD_COVER_THRESHOLD:
                selected_layers['hcc'] = True
        # If HCC < threshold, only MCC is selected (already set)
    elif hcc_val >= MIN_CLOUD_COVER_THRESHOLD:
        selected_layers['hcc'] = True

    logging.info(f"Selected layers: LCC={selected_layers['lcc']}, MCC={selected_layers['mcc']}, HCC={selected_layers['hcc']}")
    return selected_layers


def calculate_cumulative_transmittance(ray_heights, ray_cloud_profile, cams_dict_all, distances_km: float = DEFAULT_AZIMUTH_LINE_DISTANCE_KM/DESIRED_NUM_POINTS) -> np.ndarray:
    """
    Calculate cumulative transmittance (total transmitted light) for each timestep.
    
    Parameters:
    - ray_heights: 2D array (num_timesteps, num_points) - ray heights at each point
    - ray_cloud_profile: 2D array (num_timesteps, num_points) - cloud cover (0-100)
    - cams_dict_all: dict of with CAMS variables ('clwc', 'ciwc', 'aec') as key, content are 
    2D array with shape (num_timesteps, num_points)
    
    Returns:
    - total_transmitted_light: 1D array (num_timesteps,) - cumulative transmittance
    - 
    """
    alpha_lwc = 3/(2*RHO_WATER*R_CLOUDDROP)
    alpha_iwc = 3/(2*RHO_ICE*R_ICECRYSTAL)
    
    num_timesteps = ray_heights.shape[0]
    total_transmittance = np.ones(num_timesteps)

    for ts in range(num_timesteps):
        # 1. Geometry
        dx = distances_km * 1000  # Should be equal spacing in distance along azimuth
        dy = np.diff(ray_heights[ts, :])
        ds = np.sqrt(dx**2 + dy**2)
    
        # Midpoint heights for the segments (num_points - 1)
        h_mid = 0.5 * (ray_heights[ts, :-1] + ray_heights[ts, 1:])
        
        # Calculate T, P, and Rho for the 1D path ONLY
        temp_k = T0 - (L * h_mid)
        pres_pa = P0 * (1 - (L * h_mid) / T0)**5.2558
        rho_air = pres_pa / (R_DRY * temp_k)

        # Helper to get midpoints of CAMS variables
        def to_mid(arr): return 0.5 * (arr[:-1] + arr[1:])
        
        # Convert kg/kg to kg/m^3 for this specific ray path
        # Assuming clwc/ciwc were extracted using your extract_cams_variable_along_ray function
        lwc_vol = to_mid(cams_dict_all['clwc'][ts, :]) * rho_air
        iwc_vol = to_mid(cams_dict_all['ciwc'][ts, :]) * rho_air
        
        # Aerosol Extinction (ensure it's in m^-1)
        aec_path = to_mid(cams_dict_all['aerext532'][ts, :])
        
        # 3. Sum up the Optical Depth (tau)
        beta_ext = (alpha_lwc * lwc_vol) + (alpha_iwc * iwc_vol) + aec_path
        
        # --- MASKING LOGIC ---
        # We skip index 0 (the segment connecting point 0 to point 2)
        # That's a distance of 28*2 km
        # This prevents "local" grid-smearing from killing the light immediately
        beta_ext_masked = beta_ext[2:]
        ds_masked = ds[2:]
        
        # 4. Sum up the Optical Depth (tau)
        tau_total = np.sum(beta_ext_masked * ds_masked)
        
        # Resulting transmittance for this timestep
        total_transmittance[ts] = np.exp(-tau_total)
    
    return total_transmittance


def calculate_afterglow_time_from_ray_tracing(total_transmitted_light, I_ray_threshold=I_RAY_THRESHOLD_DEFAULT) -> tuple[int, int]:
    """
    Calculate afterglow time based on number of timesteps where I_ray > threshold.
    Each timestep represents 60 seconds.
    
    Parameters:
    - total_transmitted_light: 1D array (num_timesteps,) - transmittance values
    - I_ray_threshold: Minimum I_ray for afterglow occurrence (default 0.05)
    
    Returns:
    - afterglow_time_seconds: Time in seconds (int)
    - num_above_threshold: Number of timesteps above threshold
    """
    if total_transmitted_light is None or len(total_transmitted_light) == 0:
        import pdb; pdb.set_trace()
    
    # Filter out NaN values before comparison
    valid_transmittance = total_transmitted_light[~np.isnan(total_transmitted_light)]
    if len(valid_transmittance) == 0:
        import pdb; pdb.set_trace()
    
    timesteps_above_threshold = np.sum(valid_transmittance > I_ray_threshold)
    afterglow_time_seconds = int(timesteps_above_threshold * 60)  # Each timestep is 60 seconds
    
    logging.info(f"Timesteps above I_ray threshold: {timesteps_above_threshold}, afterglow time: {afterglow_time_seconds}s")
    return afterglow_time_seconds, int(timesteps_above_threshold)


def get_possible_colors_by_layer_threshold(selected_layers):
    """
    Determine possible afterglow colors based on which cloud layers meet thresholds.
    LCC: orange-yellow
    MCC: orange-red, dark-red, crimson
    HCC: magenta, coral
    Only colors for layers with selected=True are included.
    
    Parameters:
    - selected_layers: dict with keys 'lcc', 'mcc', 'hcc' and bool values
    - totalAOD550: AOD value (for reference, not used in color selection)
    
    Returns:
    - colors: Tuple of color names for this event
    """
    colors = []
    
    if selected_layers.get('lcc', False):
        colors.append('orange-yellow')
    
    if selected_layers.get('mcc', False):
        colors.extend(['orange-red', 'dark-red', 'crimson'])
    
    if selected_layers.get('hcc', False):
        colors.extend(['magenta', 'coral'])
    
    if not colors:
        colors = ['none']
    
    return tuple(colors)


def process_event_with_ray_tracing(cloud_vars, lat, lon, azimuth, cams_ds, distance_km=DEFAULT_AZIMUTH_LINE_DISTANCE_KM, num_points=DESIRED_NUM_POINTS, event_type='event', city_name='forecast', fcst_hr=0, run='00', run_date_str='20260321'):
    """
    Process a sunset/sunrise event using ray tracing to calculate transmittance and event score.
    
    Parameters:
    - cloud_vars: dict with combined cloud layers
    - lat, lon: Observer location
    - azimuth: Sun azimuth angle
    - cams_ds: Dataset containing CAMS variables including cloud liquid water content, aerosol exteinction coefficient
    - distance_km: Distance along azimuth line
    - num_points: Number of points for ray tracing
    
    Returns:
    - event_score: Score 0-100 (0 if no afterglow)
    - afterglow_time_seconds: Afterglow duration
    - possible_colors: Tuple of possible colors
    - selected_layers: Dict of which layers were selected
    - transmittance_array: 1D array of transmittance values (one per timestep)
    """
    try:
        # Get combined cloud layers
        cloud_layers_da = cloud_vars.get("cloud_layers")
        if cloud_layers_da is None:
            logging.warning("No combined cloud layers available for ray tracing")
            return 0, 0, ('none',), {'lcc': False, 'mcc': False, 'hcc': False}, np.array([]), np.nan
        # If the cloud variables include a forecast-hour/time dimension, select
        # the requested forecast-hour slice for this event. GRIB time is a
        # forecast-hour (fcst_hr) while ray timesteps are per-ray (60s apart).
        def _select_fcst_slice(obj, fcst_hr_val):
            try:
                if obj is None:
                    return obj
                if not hasattr(obj, 'dims'):
                    return obj
                # Common coordinate names: 'step', 'time', 'forecast_hour'
                if 'step' in obj.dims:
                    try:
                        return obj.sel(step=fcst_hr_val)
                    except Exception:
                        return obj.isel(step=fcst_hr_val)
                if 'time' in obj.dims:
                    try:
                        return obj.isel(time=fcst_hr_val)
                    except Exception:
                        # If there's only one time, squeeze and return it
                        if obj.sizes.get('time', 0) == 1:
                            return obj.isel(time=0)
                        return obj
                if 'forecast_hour' in obj.dims:
                    try:
                        return obj.sel(forecast_hour=fcst_hr_val)
                    except Exception:
                        return obj.isel(forecast_hour=fcst_hr_val)
                return obj
            except Exception as e:
                logging.debug(f"_select_fcst_slice: failed to select fcst_hr={fcst_hr_val}: {e}")
                return obj
        
        # Get individual layers for local cloud cover extraction
        lcc = cloud_vars.get("lcc")
        mcc = cloud_vars.get("mcc")
        hcc = cloud_vars.get("hcc")
        # If a forecast-hour index was provided for this event, try to select
        # that fcst slice from any xarray objects so the same fcst snapshot is
        # used for all ray timesteps (we repeat fcst snapshots across ray
        # timesteps later in the broadcasting helper).
        try:
            if fcst_hr is not None:
                cloud_layers_da = _select_fcst_slice(cloud_layers_da, fcst_hr)
                lcc = _select_fcst_slice(lcc, fcst_hr)
                mcc = _select_fcst_slice(mcc, fcst_hr)
                hcc = _select_fcst_slice(hcc, fcst_hr)
                logging.debug(f"process_event: selected fcst_hr={fcst_hr} for cloud slices")
        except Exception as e:
            logging.debug(f"process_event: fcst selection failed: {e}")
        
        # Extract distances and coordinates for ray tracing
        lons, lats = azimuth_line_points(lon, lat, azimuth, distance_km=distance_km, num_points=num_points)
        distances_m = np.linspace(0, distance_km * 1000, num_points)
        
        # Get local cloud cover (first 3 points) for layer selection BEFORE ray tracing
        if lcc is not None and lcc.size > 0:
            lcc_profile = extract_cloud_cover_along_azimuth(lcc, lon, lat, azimuth, distance_km, num_points)
            lcc_local = float(np.nanmean(lcc_profile[:3])) if len(lcc_profile) >= 3 else 0
            lcc_local *= 100.0
        else:
            lcc_profile = np.zeros(num_points)
            lcc_local = 0
            
        if mcc is not None and mcc.size > 0:
            mcc_profile = extract_cloud_cover_along_azimuth(mcc, lon, lat, azimuth, distance_km, num_points)
            mcc_local = float(np.nanmean(mcc_profile[:3])) if len(mcc_profile) >= 3 else 0
            mcc_local *= 100.0
        else:
            mcc_profile = np.zeros(num_points)
            mcc_local = 0
            
        if hcc is not None and hcc.size > 0:
            hcc_profile = extract_cloud_cover_along_azimuth(hcc, lon, lat, azimuth, distance_km, num_points)
            hcc_local = float(np.nanmean(hcc_profile[:3])) if len(hcc_profile) >= 3 else 0
            hcc_local *= 100.0
        else:
            hcc_profile = np.zeros(num_points)
            hcc_local = 0
        
        # Select layers based on thresholds (this determines which cloud layer(s) to analyze)
        # Use a dummy ray profile for this call - we just need the layer selection
        dummy_profile = np.zeros((60, num_points))  # 60 timesteps, num_points
        
        selected_layers = select_layers_by_threshold(dummy_profile, lcc_local, mcc_local, hcc_local)
        
        if not any(selected_layers.values()):
            logging.warning("No cloud layers cover; continuing to run ray tracing and plotting for diagnostics")
        
        # Determine cloud_base_lvl from selected layers (use HIGHEST selected layer)
        # Use presumed cloud heights for ray tracing base
        if selected_layers.get('hcc', False):
            cloud_base_lvl = HCC_HEIGHT  # 7.5km presumed center
        elif selected_layers.get('mcc', False):
            cloud_base_lvl = MCC_HEIGHT  # 4km presumed center
        elif selected_layers.get('lcc', False):
            cloud_base_lvl = LCC_HEIGHT  # 1km presumed center
        else:
            cloud_base_lvl = MCC_HEIGHT  # Default fallback (4km)
        
        logging.info(f"Selected cloud_base_lvl: {cloud_base_lvl}m based on selected layers: {selected_layers}")
        
        # Initialize in case of early failure so variable exists for error handling
        total_transmitted_light_array = np.array([])

        # NOW perform ray tracing with the determined cloud base level
        ray_cloud_profile, ray_heights = extract_cloud_cover_with_ray_tracing(
            cloud_layers_da, lons, lats, distances_m, cloud_base_lvl=cloud_base_lvl, fcst_hr=fcst_hr
        )
        
        cams_dict = extract_cams_variable_along_ray(cams_ds, lons, lats, ray_heights)
        
        # Use the local cloud cover of the highest selected cloud layer (lcc < mcc < hcc)
        # Determine which cloud level was used for analysis (highest selected)
        if selected_layers.get('hcc', False):
            highest_local = hcc_local
            cloud_lvl_used = 'hcc'
        elif selected_layers.get('mcc', False):
            highest_local = mcc_local
            cloud_lvl_used = 'mcc'
        elif selected_layers.get('lcc', False):
            highest_local = lcc_local
            cloud_lvl_used = 'lcc'
        else:
            highest_local = 0
            cloud_lvl_used = 'mcc'
            

        # Debug: log shapes and types to help diagnose ambiguous-array boolean errors
        logging.debug(f"ray_heights: type={type(ray_heights)}, shape={getattr(ray_heights,'shape',None)}")
        logging.debug(f"ray_cloud_profile: type={type(ray_cloud_profile)}, shape={getattr(ray_cloud_profile,'shape',None)}")
        logging.debug(f"distances_m: type={type(distances_m)}, shape={getattr(distances_m,'shape',None)}")
        
        total_transmitted_light_array = calculate_cumulative_transmittance(ray_heights, ray_cloud_profile, cams_dict)
        

        # --- NEW: Calculate the Reddened Transmittance (Combined Score) Globally ---
        t0_map = {'lcc': 3.0, 'mcc': 3.0, 'hcc': 3.0}
        t0 = t0_map.get(cloud_lvl_used, 4.0)
        k = 1.0 
        
        timesteps = np.arange(len(total_transmitted_light_array))
        # Sigmoid reddening curve
        reddening_weights = 1.0 / (1.0 + np.exp(-k * (timesteps - t0)))
        
        # This is the TRUE "Afterglow Intensity" that drives the score
        combined_scores = total_transmitted_light_array * reddening_weights
        
        # Calculate event score
        I_ray_max = 0.0
        try:
            valid_mask = ~np.isnan(combined_scores)
            valid_timesteps = timesteps[valid_mask]
            valid_combined = combined_scores[valid_mask]
            
            if valid_timesteps.size > 0:
                # Find the timestep that maximizes the combination of redness and intensity
                max_idx = np.argmax(valid_combined)
                
                I_ray_max = float(valid_combined[max_idx])
                selected_ts = int(valid_timesteps[max_idx])
                
                logging.info(f"Selected candidate ts: {selected_ts}, Combined Max={I_ray_max:.6f}")
            else:
                I_ray_max = 0.0
        except Exception as e:
            logging.warning(f"Error in continuous scoring: {e}")
            I_ray_max = float(np.nanmax(combined_scores)) if len(combined_scores) > 0 else 0.0
        
        # Normalise I_ray_max to 0-0.15 range for scoring
        # (Assuming your constant is tuned for the post-sigmoid values)
        I_ray_max_norm = np.clip(I_ray_max, 0.0, RAY_NORM_CONSANT_LWC) / RAY_NORM_CONSANT_LWC
        
        # Compute score using the bell-curve weight for local cloud cover
        weight = bell_curve_cloud_cover_weight(highest_local / 100)
        event_score = I_ray_max_norm * weight * 100
        event_score = int(round(np.clip(event_score, 0, 100)))
        

        # Calculate afterglow time (Now using the corrected curve!)
        afterglow_time_seconds, _ = calculate_afterglow_time_from_ray_tracing(combined_scores)
        
        # Get colors based on selected layers
        possible_colors = get_possible_colors_by_layer_threshold(selected_layers)
        
        logging.info(f"Ray tracing event score: {event_score}, afterglow time: {afterglow_time_seconds}s")
        # Optional: Plot the ray path with cloud profiles, passing avg_aod, aerosol_trans and cloud level
        # try:
        #     # Plot the ray path with cloud profiles
        #     plot_ray_path_with_cloud_profiles(
        #         ray_heights, ray_cloud_profile, lcc_profile, mcc_profile, hcc_profile,
        #         distances_m, azimuth, distance_km, event_score,
        #         city_name, run_date_str, run, str(fcst_hr),
        #         # --- NEW: PASS BOTH ARRAYS ---
        #         raw_transmittance_array=total_transmitted_light_array,
        #         combined_scores_array=combined_scores, 
        #         event_type=event_type,
        #         cloud_level=cloud_lvl_used
        #     )
        # except Exception as e:
        #     logging.warning(f"Failed to plot ray path: {e}")
        
        return event_score, afterglow_time_seconds, possible_colors, selected_layers, total_transmitted_light_array, cloud_base_lvl
        
    except Exception as e:
        logging.exception("Ray tracing processing failed")
        import pdb; pdb.set_trace()
        return 0, 0, ('none',), {'lcc': False, 'mcc': False, 'hcc': False}, np.array([]), np.nan


def _serialize_series(values):
    """Serialize a (possibly nested) sequence of values into a list of float scalars or None.

    Diagnostics: if any element is array-like with more than one element, emit an error log
    with type/shape/dtype/context and raise a ValueError so the caller can inspect the
    offending object (useful when debugging unexpected extra dimensions).
    """
    if values is None:
        return []

    try:
        arr = np.asarray(values, dtype=object)
    except Exception:
        arr = np.array(values, dtype=object)

    flat = np.ravel(arr)
    result = []
    for idx, item in enumerate(flat):
        # xarray objects often carry dims/coords we want to see
        try:
            if 'xarray' in globals() and (isinstance(item, xr.DataArray) or isinstance(item, xr.Dataset)):
                try:
                    shape = getattr(item, 'shape', None)
                    dims = getattr(item, 'dims', None)
                    name = getattr(item, 'name', None)
                    # collect small samples of coords (avoid huge dumps)
                    coord_sample = {}
                    try:
                        for k in list(getattr(item, 'coords', {}))[:5]:
                            cv = item.coords[k].values
                            coord_sample[k] = tuple(cv.flatten()[:5]) if hasattr(cv, 'flatten') else repr(cv)[:200]
                    except Exception:
                        coord_sample = '<failed to sample coords>'

                    msg = (
                        f"_serialize_series: non-scalar xarray item idx={idx} name={name} type={type(item).__name__} "
                        f"shape={shape} dims={dims} coord_sample={coord_sample}"
                    )
                    logging.error(msg)
                except Exception:
                    logging.exception("_serialize_series: failed to introspect xarray item")
                # raise to stop execution so user can inspect
                raise ValueError(msg)

            # Generic numpy / list-like handling
            item_arr = np.asarray(item)
            shape = getattr(item_arr, 'shape', None)
            dtype = getattr(item_arr, 'dtype', type(item).__name__)
            if item_arr.size == 0:
                result.append(None)
                continue
            if item_arr.size > 1:
                # Log context and raise so user can inspect the data that has extra dimension
                try:
                    sample = item_arr.reshape(-1)[:10]
                    sample_repr = np.array2string(sample, threshold=10, max_line_width=200)
                except Exception:
                    sample_repr = repr(item)[:400]
                msg = (f"_serialize_series: non-scalar item idx={idx} type={type(item).__name__} "
                       f"shape={shape} dtype={dtype} sample={sample_repr}")
                logging.error(msg)
                raise ValueError(msg)

            # Single-element: coerce to float safely
            try:
                v = float(np.asarray(item, dtype=float).reshape(-1)[0])
                result.append(None if np.isnan(v) else v)
            except Exception as e:
                logging.exception(f"_serialize_series: failed to coerce item idx={idx} to float: {e}")
                raise

        except ValueError:
            # propagate ValueError for debugging
            raise
        except Exception:
            logging.exception(f"_serialize_series: unexpected error processing item idx={idx}")
            raise

    return result


def _serialize_distance(values):
    if values is None:
        return []
    return [float(v) for v in np.asarray(values, dtype=float)]

def plot_cloud_cover_along_azimuth(cloud_cover_data, azimuth, distance_km, fcst_hr, threshold, cloud_lvl_used, city, run_date_str, run, save_fig: bool, save_path= output_path):
    """
    Extract cloud cover data along the azimuth path. Plot the cloud cover along the azimuth line.
    
    Parameters:
    - cloud_cover_data: Array containing cloud cover data along the azimuth line
    - azimuth: Azimuth angle in degrees
    - distance_km: Distance along the azimuth line (in km)
    - save_path: Path to save the plot
    """
    # Ensure input is a numeric ndarray and normalize to percent if it's 0..1
    try:
        arr_in = np.asarray(cloud_cover_data, dtype=float)
        if arr_in.size == 0:
            cloud_cover_data = arr_in
        else:
            maxv = float(np.nanmax(arr_in))
            if maxv <= 1.0 + 1e-8:
                cloud_cover_data = (arr_in * 100.0)
            else:
                cloud_cover_data = arr_in
    except Exception:
        try:
            cloud_cover_data = np.asarray(cloud_cover_data, dtype=float)
        except Exception:
            # leave as-is; subsequent code will likely error and raise
            pass

    # Check for the first distance where cloud cover falls below the threshold
    below_threshold_index = np.argmax(cloud_cover_data <= threshold)
    logging.info(f" below_threshold_index: {below_threshold_index}")

    if below_threshold_index > 0:  # Ensure that the threshold is met somewhere in the data
        distance_below_threshold = np.linspace(0, distance_km, len(cloud_cover_data))[below_threshold_index]
        avg_first_three = float(np.mean(cloud_cover_data[:3]))
        logging.info(f"Local cloud cover is {avg_first_three}%.")
        logging.info(f"The cloud cover falls below {threshold}% at {distance_below_threshold} km.")
    else:
        logging.info(f"Cloud cover does not fall below {threshold}%.")
        distance_below_threshold = np.nan
        avg_first_three = float(np.mean(cloud_cover_data[:3]))
        if avg_first_three > 10 and avg_first_three < threshold:
            distance_below_threshold = ASSUMED_DISTANCE_BELOW_THRESHOLD_KM
            logging.info(f"Local cloud cover is {avg_first_three}%. Meet Criteria even threshold requirement not met.")
            logging.info(f"There is cloud cover above, we assume distance below threshold is {distance_below_threshold} km.")
            
    if save_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the cloud cover data
        ax.plot(np.linspace(0, distance_km, len(cloud_cover_data)), cloud_cover_data, label=f"Cloud Cover ({cloud_lvl_used})")
        
        # Plot the threshold line
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold}% threshold')
        
        if avg_first_three > 10 and avg_first_three < threshold:
            ax.axhline(y=avg_first_three, color='orange', linestyle='--', label=f'Local cloud cover {avg_first_three}%')
            
        if ~np.isnan(distance_below_threshold):
            ax.axvline(x=distance_below_threshold, color='r', linestyle='--', label=f'{threshold}% threshold')
        
        ax.set_ylim(0,100)
        ax.set_xlabel('Distance along Azimuth (km)')
        ax.set_ylabel('AIFS Cloud Cover (%)')
        ax.set_title(f'{run_date_str} {run}z +{fcst_hr}h EC AIFS cloud cover Along Azimuth path ')
        #set subtitle
        ax.text(0.5, 1.02, f"Azimuth {azimuth}°", fontsize=10)
        ax.legend()
        
        plt.savefig(save_path + f"/{run_date_str}{run}0000_{fcst_hr}h_AIFS_cloud_cover_azimuth_{city.name}.png")
        plt.close()
    return distance_below_threshold, avg_first_three


def plot_ray_path_with_cloud_profiles(ray_heights, ray_cloud_profile, lcc_profile, mcc_profile, hcc_profile, 
                                       distances_m, azimuth, distance_km, event_score, city_name, run_date_str, 
                                       run, fcst_hr, raw_transmittance_array=None, combined_scores_array=None, 
                                       event_type='event', save_path=output_path, aod_val=None, aerosol_trans=None, 
                                       aerosol_t_array=None, cloud_level=None):
    """
    Plot height vs distance along the ray path with cloud profiles in grayscale.
    
    Parameters:
    - ray_heights: 2D array of ray heights (num_timesteps, num_points)
    - ray_cloud_profile: 2D array of cloud cover along ray (num_timesteps, num_points)
    - lcc_profile, mcc_profile, hcc_profile: 1D arrays of cloud cover for each layer
    - distances_m: 1D array of distances in meters
    - azimuth: Azimuth angle in degrees
    - distance_km: Total distance in km
    - event_score: Score for this event
    - city: City object with name
    - run_date_str, run, fcst_hr: For plot title
    - raw_transmittance_array: 1D array of raw intensity values
    - combined_scores_array: 1D array of sigmoid-corrected scores to identify best path
    - save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    import numpy as np

    fig, ax = plt.subplots(figsize=(15, 8))
    
    distances_km = distances_m / 1000.0  # Convert to km
    num_points = len(distances_km)
    # --- Normalize cloud layer profiles: accept both 0..1 and 0..100 ranges ---
    def _to_1d(p):
        try:
            a = np.asarray(p, dtype=float)
            if a.ndim > 1:
                a = a.reshape(-1)
            return a
        except Exception:
            return None

    lcc_arr = _to_1d(lcc_profile)
    mcc_arr = _to_1d(mcc_profile)
    hcc_arr = _to_1d(hcc_profile)

    # find maximum across provided profiles to decide scaling
    max_candidates = [np.nanmax(x) for x in (lcc_arr, mcc_arr, hcc_arr) if x is not None and x.size > 0]
    max_val = float(np.nanmax(max_candidates)) if max_candidates else np.nan
    if not np.isnan(max_val) and max_val <= 1.0 + 1e-8:
        if lcc_arr is not None:
            lcc_arr = lcc_arr * 100.0
        if mcc_arr is not None:
            mcc_arr = mcc_arr * 100.0
        if hcc_arr is not None:
            hcc_arr = hcc_arr * 100.0

    # fallback to originals if conversion failed
    lcc_profile = lcc_arr if lcc_arr is not None else lcc_profile
    mcc_profile = mcc_arr if mcc_arr is not None else mcc_profile
    hcc_profile = hcc_arr if hcc_arr is not None else hcc_profile
    
    # Interpolate cloud profiles for smooth visualization
    distances_km_interp = np.linspace(0, distance_km, num_points * 4)  # 4x finer grid
    
    # Use numpy's interp for smooth interpolation
    lcc_smooth = np.interp(distances_km_interp, distances_km, lcc_profile)
    mcc_smooth = np.interp(distances_km_interp, distances_km, mcc_profile)
    hcc_smooth = np.interp(distances_km_interp, distances_km, hcc_profile)
    
    # Use inverted grayscale colormap so 100% cloud cover -> white, 0% -> black
    cmap_clouds = plt.cm.get_cmap('Greys_r')
    # Set plot background to black so non-cloud areas appear black
    ax.set_facecolor('black')
    
    # Plot cloud layer background regions
    ax.axhspan(LCC_HMIN, LCC_HMAX, alpha=0.05, color='gray', label='LCC (0-1km)')
    ax.axhspan(MCC_HMIN, MCC_HMAX, alpha=0.05, color='gray', label='MCC (1-4km)')
    ax.axhspan(HCC_HMIN, HCC_HMAX, alpha=0.05, color='gray', label='HCC (4-9km)')
    
    # Plot smooth cloud profiles as continuous fills
    bar_width = (distances_km_interp[-1] - distances_km_interp[0]) / (len(distances_km_interp) * 0.8)
    for idx, (dist, lcc_val, mcc_val, hcc_val) in enumerate(zip(distances_km_interp, lcc_smooth, mcc_smooth, hcc_smooth)):
        # LCC - grayscale based on cloud cover %
        if lcc_val > 0:
            gray_shade = cmap_clouds(lcc_val / 100.0)
            ax.barh(np.mean([LCC_HMIN, LCC_HMAX]), bar_width, left=dist, height=LCC_HMAX - LCC_HMIN,
                   color=gray_shade, edgecolor='none')
        
        # MCC - grayscale based on cloud cover %
        if mcc_val > 0:
            gray_shade = cmap_clouds(mcc_val / 100.0)
            ax.barh(np.mean([MCC_HMIN, MCC_HMAX]), bar_width, left=dist, height=MCC_HMAX - MCC_HMIN,
                   color=gray_shade, edgecolor='none')
        
        # HCC - grayscale based on cloud cover %
        if hcc_val > 0:
            gray_shade = cmap_clouds(hcc_val / 100.0)
            ax.barh(np.mean([HCC_HMIN, HCC_HMAX]), bar_width, left=dist, height=HCC_HMAX - HCC_HMIN,
                   color=gray_shade, edgecolor='none')
    
    # Plot layer boundaries as dotted lines
    ax.axhline(y=LCC_HMIN, color='black', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=LCC_HMAX, color='black', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=MCC_HMIN, color='black', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=MCC_HMAX, color='black', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=HCC_HMIN, color='black', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=HCC_HMAX, color='black', linestyle=':', linewidth=1, alpha=0.3)
    
    # Identify the best ray path (timestep with maximum combined score)
    best_timestep = 0
    if combined_scores_array is not None and len(combined_scores_array) > 0:
        valid_trans = ~np.isnan(combined_scores_array)
        if np.any(valid_trans):
            # Choose start index based on cloud_level
            start_idx_map = {'lcc': 3, 'mcc': 3, 'hcc': 3}
            start_idx = start_idx_map.get(cloud_level, 0)
            # Build candidate indices from start_idx onward, filter valid
            if start_idx < len(combined_scores_array):
                candidates = np.arange(start_idx, len(combined_scores_array))
                valid_candidates = candidates[valid_trans[candidates]] if candidates.size > 0 else np.array([], dtype=int)
            else:
                valid_candidates = np.array([], dtype=int)

            if valid_candidates.size > 0:
                # Define two indices:
                # - reddening_boundary_idx: earliest valid candidate (visual boundary),
                # - best_timestep: candidate with maximum combined score among valid candidates
                reddening_boundary_idx = int(valid_candidates[0])
                rel = np.nanargmax(combined_scores_array[valid_candidates])
                best_timestep = int(valid_candidates[rel])
            else:
                # fallback to global maximum and set reddening boundary equal to it
                best_timestep = int(np.nanargmax(combined_scores_array))
                reddening_boundary_idx = best_timestep
        else:
            reddening_boundary_idx = 0
    
    # Plot all ray paths
    num_timesteps = ray_heights.shape[0]
    timesteps_to_plot = np.linspace(0, num_timesteps - 1, min(12, num_timesteps), dtype=int)
    
    # Plot and label every second ray path with its Raw and Score values
    # Spread out labels horizontally along the ray path
    label_positions = np.linspace(0.2, 0.8, num_timesteps)  # Fractional positions along the path
    for t_idx in range(num_timesteps):
        ray_path = ray_heights[t_idx, :]
        valid_mask = ~np.isnan(ray_path)
        if np.sum(valid_mask) < 2:
            continue

        # Extract both values for labels
        raw_val = None
        score_val = None
        if raw_transmittance_array is not None and t_idx < len(raw_transmittance_array):
            raw_val = raw_transmittance_array[t_idx]
        if combined_scores_array is not None and t_idx < len(combined_scores_array):
            score_val = combined_scores_array[t_idx]

        # Format label text
        if raw_val is not None and score_val is not None:
            label_text = f"I_ray={raw_val:.2g}\nScore={score_val:.2g}"
        elif raw_val is not None:
            label_text = f"I_ray={raw_val:.2g}"
        else:
            label_text = ""

        # Plot reddening boundary as dashed red with legend
        if t_idx == reddening_boundary_idx:
            ax.plot(distances_km[valid_mask], ray_path[valid_mask],
                    color='red', linewidth=1, alpha=0.95, zorder=5,
                    linestyle='-', label='Reddening Boundary')

        # Plot selected ray as dotted orange with legend
        if t_idx == best_timestep:
            label_color = 'orange'
            label_fontweight = 'normal'
            label_fontsize = 10
            label_zorder = 20
            ax.plot(distances_km[valid_mask], ray_path[valid_mask],
                    color='orange', linewidth=2.5, alpha=0.95, zorder=10, linestyle=':', label='Selected Ray Path')
            # Always label the selected (best) path
            if label_text:
                pos_frac = label_positions[t_idx]
                idx = int(pos_frac * (np.sum(valid_mask) - 1))
                x_label = distances_km[valid_mask][idx]
                y_label = ray_path[valid_mask][idx]
                ax.text(
                    x_label, y_label, label_text,
                    color='white', fontsize=label_fontsize, fontweight=label_fontweight, zorder=label_zorder,
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')],
                    clip_on=True, ha='center', va='center'
                )
        # Plot all other rays faded for context
        elif t_idx != reddening_boundary_idx:
            ax.plot(distances_km[valid_mask], ray_path[valid_mask],
                    color='C0', linewidth=0.8, alpha=0.4, zorder=1)
            label_color = 'gray'
            label_fontweight = 'normal'
            label_fontsize = 9
            label_zorder = 10
            # Only label every second (even-indexed) path to reduce congestion
            if t_idx % 2 == 0 and label_text:
                pos_frac = label_positions[t_idx]
                idx = int(pos_frac * (np.sum(valid_mask) - 1))
                x_label = distances_km[valid_mask][idx]
                y_label = ray_path[valid_mask][idx]
                ax.text(
                    x_label, y_label, label_text,
                    color=label_color, fontsize=label_fontsize, fontweight=label_fontweight, zorder=label_zorder,
                    path_effects=[patheffects.withStroke(linewidth=1.0, foreground='black')],
                    clip_on=True, ha='center', va='center'
                )

    # Set larger font size for axis labels and ticks
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # Add legend to distinguish selected ray and reddening boundary
    ax.legend(loc='upper right', fontsize=12)
    
    # Create colorbar for cloud cover percentage
    from matplotlib.colors import Normalize
    sm = plt.cm.ScalarMappable(cmap=cmap_clouds, norm=Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('AIFS Cloud Cover (%)', fontsize=18, fontweight='bold')
    cbar.ax.set_yticks([0, 25, 50, 75, 100])
    
    # Formatting
    ax.set_xlabel('Distance along Azimuth (km)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Height (m)', fontsize=18, fontweight='bold')
    ax.set_ylim(0, 10000)
    ax.set_xlim(0, distance_km)
    
    # Cap azimuth to integer for display
    azimuth_int = int(round(azimuth))

    # Compute local cloud cover for all layers (first 3 points average)
    local_lcc = float(np.nanmean(lcc_profile[:3])) if lcc_profile is not None and len(lcc_profile) >= 3 else None
    local_mcc = float(np.nanmean(mcc_profile[:3])) if mcc_profile is not None and len(mcc_profile) >= 3 else None
    local_hcc = float(np.nanmean(hcc_profile[:3])) if hcc_profile is not None and len(hcc_profile) >= 3 else None

    # Compose title with run datetime and fcst_hr
    run_dt_str = run_date_str if 'run_date_str' in locals() and run_date_str is not None else ''
    run_hr_str = run if 'run' in locals() and run is not None else ''
    fcst_hr_str = fcst_hr if 'fcst_hr' in locals() and fcst_hr is not None else ''

    # Decide displayed score: if local cloud is below threshold, show 0 in title
    highest_local = 0.0
    for v in (local_lcc, local_mcc, local_hcc):
        if v is not None:
            highest_local = max(highest_local, float(v))
    displayed_score = event_score if highest_local >= MIN_CLOUD_COVER_THRESHOLD else 0

    # Compose title
    title = f'Ray Path IFS Intensities & AIFS Cloud Profiles for {city_name}'
    title += f"\n(Azimuth {azimuth_int}°, Score: {displayed_score}) Run: {run_dt_str} {run_hr_str}z, fcst_hr: +{fcst_hr_str}h"
    extra = []
    # Always show AOD and aerosol transmittance; prefer per-timestep aerosol trans if provided
    if aod_val is not None:
        extra.append(f"AOD={aod_val:.3f}")
    else:
        extra.append("AOD=0.000")

    # Show all local cloud covers if available
    if local_lcc is not None:
        extra.append(f"LCC={local_lcc:.1f}%")
    if local_mcc is not None:
        extra.append(f"MCC={local_mcc:.1f}%")
    if local_hcc is not None:
        extra.append(f"HCC={local_hcc:.1f}%")
    if extra:
        title += "\n" + ", ".join(extra)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend in lower-right including reddening boundary
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='red', lw=0.8, label='reddening boundary'),
        Line2D([0], [0], color='orange', lw=1.5, linestyle=':', label='selected ray path')
    ]
    ax.legend(handles=legend_lines, loc='lower right', fontsize=11, framealpha=0.95, prop={'weight':'normal'})
    
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plot_filename = f"{save_path}/{run_date_str}{run}0000-{fcst_hr}h-{event_type}-AIFS_cloud_cover_ray_path_{city_name}.png"
    plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved ray path plot: {plot_filename}")

def _first_valid_scalar(p, default=np.nan):
    """Return first valid scalar from array-like `p`.
    Handles nested arrays/lists and object dtypes. Returns `default` when nothing valid.
    """
    if p is None:
        return float(default)
    try:
        arr = np.asarray(p, dtype=object)
    except Exception:
        try:
            return float(p)
        except Exception:
            return float(default)
    # Flatten and find first numeric element
    try:
        flat = arr.ravel()
    except Exception:
        flat = np.array([arr])
    for item in flat:
        if item is None:
            continue
        try:
            sub = np.asarray(item)
            if sub.size == 0:
                continue
            val = float(np.asarray(sub).reshape(-1)[0])
            if not np.isnan(val):
                return val
        except Exception:
            continue
    return float(default)

def get_cloud_extent(data_dict, lon, lat, azimuth, distance_km = DEFAULT_AZIMUTH_LINE_DISTANCE_KM, num_points=DESIRED_NUM_POINTS):
    profile_layers = ["lcc", "mcc", "hcc"]
    cloud_present = True
    local_cloud_cover_dict = {}

    avg_first_three = np.nan

    raw_profiles = {}
    distance_axis_values = None

    for layer in profile_layers:
        da = data_dict.get(layer)
        if da is None:
            continue
        cloud_cover_data = extract_cloud_cover_along_azimuth(da, lon, lat, azimuth, distance_km, num_points)
        # Make a robust decision whether returned values are 0..1 or 0..100.
        p_raw = np.asarray(cloud_cover_data, dtype=float)
        if p_raw.size == 0:
            raw_profiles[layer] = p_raw
        else:
            max_p = np.nanmax(p_raw)
            if np.isnan(max_p):
                raw_profiles[layer] = p_raw
            elif max_p <= 1.0 + 1e-8:
                # values in 0..1 -> convert to percent
                raw_profiles[layer] = p_raw * 100.0
            else:
                # assume already percent
                raw_profiles[layer] = p_raw

        if distance_axis_values is None and p_raw.size:
            distance_axis_values = np.linspace(0, distance_km, p_raw.size)

    # Ensure all available profiles are in percent (0..100)
    scaled_profiles = {}
    avail_layers = [l for l in profile_layers if raw_profiles.get(l) is not None]
    for l in avail_layers:
        p = np.asarray(raw_profiles[l], dtype=float)
        scaled_profiles[l] = p

    # Compute combined total cloud cover from available (scaled) layer profiles
    combined_profile = None
    avail_profiles = [scaled_profiles.get(l) for l in ["lcc", "mcc", "hcc"] if scaled_profiles.get(l) is not None]
    if avail_profiles:
        try:
            arr = np.vstack(avail_profiles).astype(float)  # shape: (n_layers, n_points)
            frac = arr / 100.0
            combined_frac = 1.0 - np.prod(1.0 - frac, axis=0)
            combined_profile = (combined_frac * 100.0)
        except Exception:
            combined_profile = None

    if combined_profile is not None and getattr(combined_profile, 'size', 0) > 0:
        try:
            avg_first_three = float(np.nanmean(combined_profile[:3]))
        except Exception:
            avg_first_three = float('nan')
    else:
        avg_first_three = np.nan

    # Mark cloud_present False when the average is NaN or below threshold.
    if np.isnan(avg_first_three) or avg_first_three < MIN_CLOUD_COVER_THRESHOLD:
        cloud_present = False
    else:
        cloud_present = True

    # Build local cloud cover dict (from scaled profiles if available)
    for l in profile_layers:
        p = scaled_profiles.get(l) if l in scaled_profiles else raw_profiles.get(l)
        if p is not None and getattr(p, 'size', 0) > 0:
            local_cloud_cover_dict[l] = _first_valid_scalar(p, default=np.nan)

    # Diagnostic introspection: log types/shapes to help find unexpected extra dims
    try:
        logging.debug(f"get_cloud_extent: scaled_profiles.keys={list(scaled_profiles.keys())}, raw_profiles.keys={list(raw_profiles.keys())}, distance_axis_values_type={type(distance_axis_values)}")
    except Exception:
        logging.exception("get_cloud_extent: failed to log profile keys")

    for l in profile_layers:
        p = scaled_profiles.get(l) if l in scaled_profiles else raw_profiles.get(l)
        if p is None:
            logging.debug(f"get_cloud_extent: layer {l} is missing")
            continue
        try:
            if isinstance(p, xr.DataArray):
                logging.error(f"get_cloud_extent: layer={l} is xarray.DataArray name={p.name} dims={p.dims} shape={p.shape}")
            elif isinstance(p, xr.Dataset):
                logging.error(f"get_cloud_extent: layer={l} is xarray.Dataset vars={list(p.data_vars)} dims={{k: v.dims for k, v in p.data_vars.items()}}")
            else:
                arr = np.asarray(p)
                logging.debug(f"get_cloud_extent: layer={l} ndarray shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
                try:
                    flat = arr.reshape(-1)
                except Exception:
                    flat = np.ravel(arr)
                for idx_item, item in enumerate(flat[:10]):
                    try:
                        item_arr = np.asarray(item)
                        if getattr(item_arr, 'size', 1) > 1:
                            try:
                                sample = item_arr.reshape(-1)[:10]
                                sample_repr = np.array2string(sample, threshold=10, max_line_width=200)
                            except Exception:
                                sample_repr = repr(item)[:400]
                            logging.error(f"get_cloud_extent: layer={l} element_idx={idx_item} has extra-dim shape={getattr(item_arr,'shape',None)} dtype={getattr(item_arr,'dtype',None)} sample={sample_repr}")
                    except Exception:
                        logging.exception(f"get_cloud_extent: failed inspecting element idx={idx_item} for layer={l}")
        except Exception:
            logging.exception(f"get_cloud_extent: failed introspection for layer {l}")

    # Serialize the (scaled) profiles for output with guarded diagnostics
    try:
        serialized_profiles = {layer: _serialize_series((scaled_profiles.get(layer) if layer in scaled_profiles else raw_profiles.get(layer))) for layer in profile_layers}
    except Exception as e:
        logging.exception("get_cloud_extent: _serialize_series failed; dumping full context for debugging")
        for l in profile_layers:
            p = scaled_profiles.get(l) if l in scaled_profiles else raw_profiles.get(l)
            try:
                p_type = type(p)
                p_repr = repr(p)
            except Exception:
                p_type = type(p)
                p_repr = '<repr-failed>'
            logging.error(f"get_cloud_extent: SERIALIZE FAILURE layer={l} type={p_type} repr_preview={p_repr[:800]}")
        raise

    distance_axis_serialized = _serialize_distance(distance_axis_values)

    return (
        avg_first_three,
        cloud_present,
        serialized_profiles,
        distance_axis_serialized,
    )
    
# Color fill based on index using the 'plasma' colormap
def color_fill(index):
    # Normalize the index to a value between 0 and 1 for the colormap
    norm = plt.Normalize(vmin=0, vmax=100) #type: ignore
    cmap = plt.get_cmap('magma')  # 
    return cmap(norm(index))

# Weighted likelihod index
def bell_curve_cloud_cover_weight(cloud_cover, peak=0.6, sigma_left=0.3, sigma_right=0.8, peak_height=1.0):
    # cloud_cover: 0-1
    try: 
        c = np.clip(np.asarray(cloud_cover, dtype=float), 0.0, 1.0)
        left = peak_height * np.exp(-((c - peak) ** 2) / (2 * sigma_left ** 2))
        right = peak_height * np.exp(-((c - peak) ** 2) / (2 * sigma_right ** 2))
        return np.where(c <= peak, left, right)
    except Exception as e:
        logging.error(f"Error calculating asym bell curve weight: {e}")
        return 1.0

def load_2m_fields(grib_path: str) -> xr.Dataset:
    """
    Load 2 m temperature and dewpoint from a GRIB file, handling both
    shortName variants (t2m or 2t, d2m or 2d). Returns dataset with
    variables named t2m and d2m.

    Function By ChartGPT
    """
    variants = [
        {'shortName': 't2m', 'typeOfLevel': 'heightAboveGround', 'level': 2},
        {'shortName': '2t',  'typeOfLevel': 'heightAboveGround', 'level': 2},
    ]
    t_ds = None
    for f in variants:
        try:
            t_ds = xr.open_dataset(
                grib_path,
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': f},
                decode_timedelta=True
            )
            if t_ds.data_vars:
                break
        except Exception:
            pass
    if t_ds is None or not t_ds.data_vars:
        raise KeyError(f"No 2 m temperature (t2m/2t) found in {grib_path}")
    # rename temp var to t2m
    for v in list(t_ds.data_vars):
        if v != 't2m':
            t_ds = t_ds.rename({v: 't2m'})

    variants = [
        {'shortName': 'd2m', 'typeOfLevel': 'heightAboveGround', 'level': 2},
        {'shortName': '2d',  'typeOfLevel': 'heightAboveGround', 'level': 2},
    ]
    d_ds = None
    for f in variants:
        try:
            d_ds = xr.open_dataset(
                grib_path,
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': f},
                decode_timedelta=True
            )
            if d_ds.data_vars:
                break
        except Exception:
            pass
    if d_ds is None or not d_ds.data_vars:
        raise KeyError(f"No 2 m dewpoint (d2m/2d) found in {grib_path}")
    for v in list(d_ds.data_vars):
        if v != 'd2m':
            d_ds = d_ds.rename({v: 'd2m'})

    with xr.set_options(use_new_combine_kwarg_defaults=True):
        merged = xr.merge([t_ds, d_ds], compat='override')
    return merged

def process_city(city_name: str, country: str, lat: float, lon: float, timezone_str: str, today, run, today_str, run_date_str, input_path, output_path, create_dashboard_flag: bool):
    try:
        tz = pytz.timezone(timezone_str)
        current_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        #current_utc = datetime.datetime(2026, 3, 29, 23, 0, 0, tzinfo=datetime.timezone.utc)  # Fixed date for testing
        local_today = current_utc.astimezone(tz).date()

        tomorrow = today + datetime.timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y%m%d')
        
        # Create city object
        try:
            city = LocationInfo(city_name, country, timezone_str, lat, lon)
        except Exception as e:
            logging.error(f"Error creating LocationInfo for {city_name}: {e}")
            timezone_str = 'UTC'
            city = LocationInfo(city_name, country, timezone_str, lat, lon)
            return {"city": city_name, "error": "Invalid timezone, skipped city and defaulted to UTC"}

        sunrise_day_after = None
        sunset_day_after = None
        try:
            s_tdy = sun(city.observer, date=local_today, tzinfo=tz)
            sunset_tdy = s_tdy['sunset']
            sunrise_tdy = s_tdy['sunrise']

            s_tmr = sun(city.observer, date=local_today + datetime.timedelta(days=1), tzinfo=tz)
            sunrise_tmr = s_tmr['sunrise']
            sunset_tmr = s_tmr['sunset']

            s_day_after = sun(city.observer, date=local_today + datetime.timedelta(days=2), tzinfo=tz)
            sunrise_day_after = s_day_after['sunrise']
            sunset_day_after = s_day_after['sunset']
        except Exception as e:
            logging.warning(f"Sun calculation failed for {city.name}: {e}")
            # handle polar/no-sun or skip city
            sunrise_tdy = sunset_tdy = sunrise_tmr = sunset_tmr = None
            sunrise_day_after = sunset_day_after = None

        sunrise_azimuth = sunrise_azimuth_tmr = sunrise_azimuth_day_after = None
        sunset_azimuth = sunset_azimuth_tmr = sunset_azimuth_day_after = None
        if sunrise_tdy is not None:
            sunrise_azimuth = get__sunrise_azimuth(city, today)
        if sunrise_tmr is not None:
            sunrise_azimuth_tmr = get__sunrise_azimuth(city, today + datetime.timedelta(days=1))
        if sunrise_day_after is not None:
            sunrise_azimuth_day_after = get__sunrise_azimuth(city, today + datetime.timedelta(days=2))
        if sunset_tdy is not None:
            sunset_azimuth = get__sunset_azimuth(city, today)
        if sunset_tmr is not None:
            sunset_azimuth_tmr = get__sunset_azimuth(city, today + datetime.timedelta(days=1))
        if sunset_day_after is not None:
            sunset_azimuth_day_after = get__sunset_azimuth(city, today + datetime.timedelta(days=2))

        run_dt = latest_forecast_hours_run_to_download()
        #run_dt = datetime.datetime(2026, 3, 29, 23, 0, 0, tzinfo=datetime.timezone.utc)  # Fixed date for testing
            # Use the provided run argument to set run_dt correctly (00 or 12 UTC)
        if run is not None:
            run_hour = int(run)
            run_dt = datetime.datetime.combine(today, datetime.time(run_hour, 0), tzinfo=datetime.timezone.utc)
        else:
            run_dt = latest_forecast_hours_run_to_download()
        logging.info(f"process_city: Using run_dt={run_dt.isoformat()} (run={run})")
        picks = determine_city_forecast_hours_time_to_use(city, sunrise_tdy, sunset_tdy, run_dt)
        print("run_dt:", run_dt)
        print("picks:", picks)

        # Extract date strings from actual sunrise/sunset times in UTC to match picks dictionary keys
        sunrise_tdy_date_str = sunrise_tdy.astimezone(datetime.timezone.utc).date().strftime('%Y%m%d') if sunrise_tdy else None
        sunrise_tmr_date_str = sunrise_tmr.astimezone(datetime.timezone.utc).date().strftime('%Y%m%d') if sunrise_tmr else None
        sunset_tdy_date_str = sunset_tdy.astimezone(datetime.timezone.utc).date().strftime('%Y%m%d') if sunset_tdy else None
        sunset_tmr_date_str = sunset_tmr.astimezone(datetime.timezone.utc).date().strftime('%Y%m%d') if sunset_tmr else None

        # Extract forecast hours for both sunrise and sunset using the correct date strings
        sunrise_fh_tdy = picks.get(f'sunrise_{sunrise_tdy_date_str}') if sunrise_tdy_date_str else None
        sunrise_fh_tmr = picks.get(f'sunrise_{sunrise_tmr_date_str}') if sunrise_tmr_date_str else None
        sunset_fh_tdy = picks.get(f'sunset_{sunset_tdy_date_str}') if sunset_tdy_date_str else None
        sunset_fh_tmr = picks.get(f'sunset_{sunset_tmr_date_str}') if sunset_tmr_date_str else None

        # check for day-after-tomorrow pick and allow shifting when today is missing
        day_after_sunrise = sunrise_day_after.astimezone(datetime.timezone.utc).date().strftime('%Y%m%d') if sunrise_day_after else None
        day_after_sunset = sunset_day_after.astimezone(datetime.timezone.utc).date().strftime('%Y%m%d') if sunset_day_after else None
        sunrise_fh_day_after = picks.get(f"sunrise_{day_after_sunrise}") if day_after_sunrise else None
        sunset_fh_day_after = picks.get(f"sunset_{day_after_sunset}") if day_after_sunset else None

        # Handle missing sunrise picks
        if sunrise_fh_tdy is None and sunrise_fh_tmr is not None:
            logging.info(f"No pick for sunrise_{sunrise_tdy_date_str} for {city.name}; using tomorrow's pick ({sunrise_fh_tmr}) as today's.")
            sunrise_fh_tdy = sunrise_fh_tmr
            sunrise_fh_tmr = sunrise_fh_day_after if sunrise_fh_day_after is not None else None
            sunrise_tdy = sunrise_tmr
            sunrise_tmr = sunrise_day_after
            sunrise_azimuth = sunrise_azimuth_tmr
            sunrise_azimuth_tmr = sunrise_azimuth_day_after

        # Handle missing sunset picks
        if sunset_fh_tdy is None and sunset_fh_tmr is not None:
            logging.info(f"No pick for sunset_{sunset_tdy_date_str} for {city.name}; using tomorrow's pick ({sunset_fh_tmr}) as today's.")
            sunset_fh_tdy = sunset_fh_tmr
            sunset_fh_tmr = sunset_fh_day_after if sunset_fh_day_after is not None else None
            sunset_tdy = sunset_tmr
            sunset_tmr = sunset_day_after
            sunset_azimuth = sunset_azimuth_tmr
            sunset_azimuth_tmr = sunset_azimuth_day_after
        
        logging.info(f"Final picks for {city.name}: sunrise today +{sunrise_fh_tdy}h, sunrise tomorrow +{sunrise_fh_tmr}h, sunset today +{sunset_fh_tdy}h, sunset tomorrow +{sunset_fh_tmr}h")

        sunrise_time_tdy_local = _format_local_time(sunrise_tdy, tz)
        sunrise_time_tmr_local = _format_local_time(sunrise_tmr, tz)
        sunset_time_tdy_local = _format_local_time(sunset_tdy, tz)
        sunset_time_tmr_local = _format_local_time(sunset_tmr, tz)

        max_elev = max_solar_elevation(city, datetime.date.today())
        logging.info(f"Maximum solar elevation in {city.name}: {max_elev:.2f}°")
        
        logging.info(f"Sunrise azimuth angle in {city.name}: {sunrise_azimuth:.2f}°")
        logging.info(f"Sunset azimuth angle in {city.name}: {sunset_azimuth:.2f}°")
        logging.info(f"sunrise time in {city.name}: {sunrise_tdy}")
        logging.info(f"sunset time in {city.name}: {sunset_tdy}")

        # Define a square box enclosing the region of interest
        lat_min, lat_max = lat - 5, lat + 5
        lon_min, lon_max = lon - 5, lon + 5

        # Guard for missing forecast hours
        if sunrise_fh_tdy is None and sunset_fh_tdy is None:
            logging.warning(f"No forecast-hour files available for {city.name} (today); skipping city.")
            return {"city": city.name, "error": "missing forecast-hour files"}

        # Process sunset (existing logic)
        sunset_results = {}
        sunset_profile_payload_tdy = build_profile_payload([], {})
        sunset_profile_payload_tmr = build_profile_payload([], {})
        if sunset_fh_tdy is not None and sunset_fh_tmr is not None:
            logging.info(f"Processing sunset for {city.name}")
            
            cams_cloud_tdy = f"{input_path}/cams_cloud_cover_data_{today_str}{run}0000.grib"
            cams_cloud_tmr = f"{input_path}/cams_cloud_cover_data_{today_str}{run}0000.grib"

            # Load combined cloud layers (lcc, mcc, hcc) with lvl dimension
            ds_tdy_cloud_layers = combine_cloud_layers(cams_cloud_tdy, sunset_fh_tdy)  # dims: (lvl, latitude, longitude)
            ds_tmr_cloud_layers = combine_cloud_layers(cams_cloud_tmr, sunset_fh_tmr)
            
            cams_grib = f'{input_path}/cams_data_{run_date_str}{run}0000.grib'
            
            ds_cams_all_tdy = open_cams_data_and_blend_clouds(cams_grib, ds_tdy_cloud_layers, sunset_fh_tdy)
            ds_cams_all_tmr = open_cams_data_and_blend_clouds(cams_grib, ds_tmr_cloud_layers, sunset_fh_tmr)
            
            ds_cams_all_tdy = convert_pressure_coordinate_to_height(ds_cams_all_tdy)
            ds_cams_all_tmr = convert_pressure_coordinate_to_height(ds_cams_all_tmr)
            
            cloud_vars_tdy = {
                "lcc": ds_tdy_cloud_layers['lcc'],
                "mcc": ds_tdy_cloud_layers['mcc'],
                "hcc": ds_tdy_cloud_layers['hcc'],
                "cloud_layers": ds_tdy_cloud_layers  # combined with lvl dimension (Dataset)
            }

            cloud_vars_tmr = {
                "lcc": ds_tmr_cloud_layers['lcc'],
                "mcc": ds_tmr_cloud_layers['mcc'],
                "hcc": ds_tmr_cloud_layers['hcc'],
                "cloud_layers": ds_tmr_cloud_layers  # combined with lvl dimension (Dataset)
            }
            cloud_vars_tdy = reduce_clouds_to_roi(cloud_vars_tdy, lon, lat, sunset_azimuth, distance_km=700, pad_km=80)
            cloud_vars_tmr = reduce_clouds_to_roi(cloud_vars_tmr, lon, lat, sunset_azimuth_tmr, distance_km=700, pad_km=80)

            if create_dashboard_flag:
                pass
            
            (
                avg_first_three_tdy,
                cloud_present_tdy,
                cloud_profiles_tdy,
                distance_axis_tdy) = get_cloud_extent(
                cloud_vars_tdy, lon, lat, sunset_azimuth)
            
            (
                avg_first_three_tmr,
                cloud_present_tmr,
                cloud_profiles_tmr,
                distance_axis_tmr,) = get_cloud_extent(
                cloud_vars_tmr, lon, lat, sunset_azimuth_tmr)

            sunset_profile_payload_tdy = build_profile_payload(distance_axis_tdy, cloud_profiles_tdy)
            sunset_profile_payload_tmr = build_profile_payload(distance_axis_tmr, cloud_profiles_tmr)
            
            #import pdb; pdb.set_trace()
            
            # Use ray tracing-based scoring and colors
            likelihood_index_tdy, actual_afterglow_time_tdy, possible_colors_tdy, selected_layers_tdy, trans_array_tdy, cb_from_ray_tdy = process_event_with_ray_tracing(
                cloud_vars_tdy, lat, lon, sunset_azimuth, ds_cams_all_tdy, distance_km=700, num_points=DESIRED_NUM_POINTS, event_type='sunset', city_name=city_name, fcst_hr=sunset_fh_tdy, run=run, run_date_str=run_date_str
            )
            likelihood_index_tmr, actual_afterglow_time_tmr, possible_colors_tmr, selected_layers_tmr, trans_array_tmr, cb_from_ray_tmr = process_event_with_ray_tracing(
                cloud_vars_tmr, lat, lon, sunset_azimuth_tmr, ds_cams_all_tmr, distance_km=700, num_points=DESIRED_NUM_POINTS, event_type='sunset', city_name=city_name, fcst_hr=sunset_fh_tmr, run=run, run_date_str=run_date_str
            )
            
            # Use cloud base from ray tracing if available. Cloud base straight from model output to be implemented.
            cloud_base_lvl_tdy = cb_from_ray_tdy
            cloud_base_lvl_tmr = cb_from_ray_tmr
            
            logging.info(f"{city.name} sunset likelihood_index_tdy: {likelihood_index_tdy}")
            logging.info(f"{city.name} sunset likelihood_index_tmr: {likelihood_index_tmr}")

            if create_dashboard_flag:
                pass
            # Normalize selected_layers to a single key for frontend display
            def _selected_layer_key(sel):
                if not isinstance(sel, dict):
                    return None
                if sel.get('hcc', False):
                    return 'hcc'
                if sel.get('mcc', False):
                    return 'mcc'
                if sel.get('lcc', False):
                    return 'lcc'
                return None

            key_sunset_tdy = _selected_layer_key(selected_layers_tdy)
            key_sunset_tmr = _selected_layer_key(selected_layers_tmr)
            
            sunset_results = {
                "sunset_likelihood_index_tdy": likelihood_index_tdy,
                "sunset_likelihood_index_tmr": likelihood_index_tmr,
                "sunset_possible_colors_tdy": possible_colors_tdy,
                "sunset_possible_colors_tmr": possible_colors_tmr,
                "sunset_afterglow_time_tdy": actual_afterglow_time_tdy,
                "sunset_afterglow_time_tmr": actual_afterglow_time_tmr,
                "sunset_cloud_base_lvl_tdy": cloud_base_lvl_tdy,
                "sunset_cloud_base_lvl_tmr": cloud_base_lvl_tmr,
                "sunset_cloud_local_cover_tdy": avg_first_three_tdy,
                "sunset_cloud_local_cover_tmr": avg_first_three_tmr,
                "sunset_cloud_present_tdy": cloud_present_tdy,
                "sunset_cloud_present_tmr": cloud_present_tmr,
                "sunset_time_tdy": sunset_time_tdy_local,
                "sunset_time_tmr": sunset_time_tmr_local,
                "sunset_cloud_layer_key_tdy": key_sunset_tdy,
                "sunset_cloud_layer_key_tmr": key_sunset_tmr,
                "sunset_azimuth_tdy": sunset_azimuth,
                "sunset_azimuth_tmr": sunset_azimuth_tmr,
                "sunset_cloud_profiles_tdy": sunset_profile_payload_tdy,
                "sunset_cloud_profiles_tmr": sunset_profile_payload_tmr,
            }
        else:
            logging.warning(f"Skipping sunset processing for {city.name}: sunset_fh_tdy={sunset_fh_tdy}, sunset_fh_tmr={sunset_fh_tmr}")
            sunset_results = {
                "sunset_likelihood_index_tdy": np.nan,
                "sunset_likelihood_index_tmr": np.nan,
                "sunset_possible_colors_tdy": ('none',),
                "sunset_possible_colors_tmr": ('none',),
                "sunset_afterglow_time_tdy": np.nan,
                "sunset_afterglow_time_tmr": np.nan,
                "sunset_cloud_base_lvl_tdy": np.nan,
                "sunset_cloud_base_lvl_tmr": np.nan,
                "sunset_cloud_local_cover_tdy": np.nan,
                "sunset_cloud_local_cover_tmr": np.nan,
                "sunset_cloud_present_tdy": False,
                "sunset_cloud_present_tmr": False,
                "sunset_time_tdy": sunset_time_tdy_local,
                "sunset_time_tmr": sunset_time_tmr_local,
                
                "sunset_cloud_layer_key_tdy": None,
                "sunset_cloud_layer_key_tmr": None,
                
                "sunset_azimuth_tdy": sunset_azimuth,
                "sunset_azimuth_tmr": sunset_azimuth_tmr,
                "sunset_cloud_profiles_tdy": build_profile_payload([], {}),
                "sunset_cloud_profiles_tmr": build_profile_payload([], {}),
            }

        # Process sunrise 
        sunrise_results = {}
        sunrise_profile_payload_tdy = build_profile_payload([], {})
        sunrise_profile_payload_tmr = build_profile_payload([], {})
        if sunrise_fh_tdy is not None and sunrise_fh_tmr is not None:
            logging.info(f"Processing sunrise for {city.name}")
            
            cams_cloud_grib= f"{input_path}/cams_cloud_cover_data_{today_str}{run}0000.grib"

            # Load combined cloud layers (lcc, mcc, hcc) with lvl dimension
            # Prefer selecting the forecast-hour slice at load time so the
            # returned dataset doesn't carry a forecast/time axis that can
            # lead to mismatched broadcasting later.
            sunrise_tdy_cloud_layers = combine_cloud_layers(cams_cloud_grib, fcst_hr=sunrise_fh_tdy)
            sunrise_tmr_cloud_layers = combine_cloud_layers(cams_cloud_grib, fcst_hr=sunrise_fh_tmr)
            
            cams_grib = f'{input_path}/cams_data_{run_date_str}{run}0000.grib'
            
            ds_cams_all_tdy = open_cams_data_and_blend_clouds(cams_grib, sunrise_tdy_cloud_layers, sunrise_fh_tdy)
            ds_cams_all_tmr = open_cams_data_and_blend_clouds(cams_grib, sunrise_tmr_cloud_layers, sunrise_fh_tmr)
            
            ds_cams_all_tdy = convert_pressure_coordinate_to_height(ds_cams_all_tdy)
            ds_cams_all_tmr = convert_pressure_coordinate_to_height(ds_cams_all_tmr)
            
            cloud_vars_sunrise_tdy = {
                "lcc": sunrise_tdy_cloud_layers['lcc'],
                "mcc": sunrise_tdy_cloud_layers['mcc'],
                "hcc": sunrise_tdy_cloud_layers['hcc'],
                "cloud_layers": sunrise_tdy_cloud_layers  # combined with lvl dimension (Dataset)
            }

            cloud_vars_sunrise_tmr = {
                "lcc": sunrise_tmr_cloud_layers['lcc'],
                "mcc": sunrise_tmr_cloud_layers['mcc'],
                "hcc": sunrise_tmr_cloud_layers['hcc'],
                "cloud_layers": sunrise_tmr_cloud_layers  # combined with lvl dimension (Dataset)
            }

            # Slice to ROI around sunrise azimuth
            cloud_vars_sunrise_tdy = reduce_clouds_to_roi(cloud_vars_sunrise_tdy, lon, lat, sunrise_azimuth, distance_km=700, pad_km=80)
            cloud_vars_sunrise_tmr = reduce_clouds_to_roi(cloud_vars_sunrise_tmr, lon, lat, sunrise_azimuth_tmr, distance_km=700, pad_km=80)

            if create_dashboard_flag:
                pass
            

            avg_first_three_sunrise_tdy, cloud_present_sunrise_tdy, cloud_profiles_sunrise_tdy, distance_axis_sunrise_tdy = get_cloud_extent(
                cloud_vars_sunrise_tdy, lon, lat, sunrise_azimuth
            )

            avg_first_three_sunrise_tmr, cloud_present_sunrise_tmr, cloud_profiles_sunrise_tmr, distance_axis_sunrise_tmr = get_cloud_extent(
                cloud_vars_sunrise_tmr, lon, lat, sunrise_azimuth_tmr
            )

            sunrise_profile_payload_tdy = build_profile_payload(distance_axis_sunrise_tdy, cloud_profiles_sunrise_tdy)
            sunrise_profile_payload_tmr = build_profile_payload(distance_axis_sunrise_tmr, cloud_profiles_sunrise_tmr)

            # Use ray tracing-based scoring and colors
            likelihood_index_sunrise_tdy, actual_afterglow_time_sunrise_tdy, possible_colors_sunrise_tdy, selected_layers_sunrise_tdy, trans_array_sunrise_tdy, cb_from_ray_sunrise_tdy = process_event_with_ray_tracing(
                cloud_vars_sunrise_tdy, lat, lon, sunrise_azimuth, ds_cams_all_tdy, distance_km=700, num_points=25, event_type='sunrise', city_name=city_name, fcst_hr=sunrise_fh_tdy, run=run, run_date_str=run_date_str
            )
            likelihood_index_sunrise_tmr, actual_afterglow_time_sunrise_tmr, possible_colors_sunrise_tmr, selected_layers_sunrise_tmr, trans_array_sunrise_tmr, cb_from_ray_sunrise_tmr = process_event_with_ray_tracing(
                cloud_vars_sunrise_tmr, lat, lon, sunrise_azimuth_tmr, ds_cams_all_tmr , distance_km=700, num_points=25, event_type='sunrise', city_name=city_name, fcst_hr=sunrise_fh_tmr, run=run, run_date_str=run_date_str
            )

            cloud_base_lvl_sunrise_tdy = cb_from_ray_sunrise_tdy
            cloud_base_lvl_sunrise_tmr = cb_from_ray_sunrise_tmr
        
            # Derive the selected layer key from selected_layers result
            def _selected_layer_key(sel):
                if not isinstance(sel, dict):
                    return None
                if sel.get('hcc', False):
                    return 'hcc'
                if sel.get('mcc', False):
                    return 'mcc'
                if sel.get('lcc', False):
                    return 'lcc'
                return None

            key_sunrise_tdy = _selected_layer_key(selected_layers_sunrise_tdy)
            key_sunrise_tmr = _selected_layer_key(selected_layers_sunrise_tmr)
            
            logging.info(f"{city.name} sunrise likelihood_index_tdy: {likelihood_index_sunrise_tdy}")
            logging.info(f"{city.name} sunrise likelihood_index_tmr: {likelihood_index_sunrise_tmr}")

            sunrise_results = {
                "sunrise_likelihood_index_tdy": likelihood_index_sunrise_tdy,
                "sunrise_likelihood_index_tmr": likelihood_index_sunrise_tmr,
                "sunrise_possible_colors_tdy": possible_colors_sunrise_tdy,
                "sunrise_possible_colors_tmr": possible_colors_sunrise_tmr,
                "sunrise_afterglow_time_tdy": actual_afterglow_time_sunrise_tdy,
                "sunrise_afterglow_time_tmr": actual_afterglow_time_sunrise_tmr,
                "sunrise_cloud_base_lvl_tdy": cloud_base_lvl_sunrise_tdy,
                "sunrise_cloud_base_lvl_tmr": cloud_base_lvl_sunrise_tmr,
                "sunrise_cloud_local_cover_tdy": avg_first_three_sunrise_tdy,
                "sunrise_cloud_local_cover_tmr": avg_first_three_sunrise_tmr,
                "sunrise_cloud_present_tdy": cloud_present_sunrise_tdy,
                "sunrise_cloud_present_tmr": cloud_present_sunrise_tmr,
                "sunrise_time_tdy": sunrise_time_tdy_local,
                "sunrise_time_tmr": sunrise_time_tmr_local,
                
                "sunrise_cloud_layer_key_tdy": key_sunrise_tdy,
                "sunrise_cloud_layer_key_tmr": key_sunrise_tmr,
                "sunrise_azimuth_tdy": sunrise_azimuth,
                "sunrise_azimuth_tmr": sunrise_azimuth_tmr,
                "sunrise_cloud_profiles_tdy": sunrise_profile_payload_tdy,
                "sunrise_cloud_profiles_tmr": sunrise_profile_payload_tmr,
            }
        else:
            logging.warning(f"Skipping sunrise processing for {city.name}: sunrise_fh_tdy={sunrise_fh_tdy}, sunrise_fh_tmr={sunrise_fh_tmr}")
            sunrise_results = {
                "sunrise_likelihood_index_tdy": np.nan,
                "sunrise_likelihood_index_tmr": np.nan,
                "sunrise_possible_colors_tdy": ('none',),
                "sunrise_possible_colors_tmr": ('none',),
                "sunrise_afterglow_time_tdy": np.nan,
                "sunrise_afterglow_time_tmr": np.nan,
                "sunrise_cloud_base_lvl_tdy": np.nan,
                "sunrise_cloud_base_lvl_tmr": np.nan,
                "sunrise_cloud_local_cover_tdy": np.nan,
                "sunrise_cloud_local_cover_tmr": np.nan,
                "sunrise_cloud_present_tdy": False,
                "sunrise_cloud_present_tmr": False,
                "sunrise_time_tdy": sunrise_time_tdy_local,
                "sunrise_time_tmr": sunrise_time_tmr_local,
                
                "sunrise_cloud_layer_key_tdy": None,
                "sunrise_cloud_layer_key_tmr": None,
                "sunrise_azimuth_tdy": sunrise_azimuth,
                "sunrise_azimuth_tmr": sunrise_azimuth_tmr,
                "sunrise_cloud_profiles_tdy": build_profile_payload([], {}),
                "sunrise_cloud_profiles_tmr": build_profile_payload([], {}),
            }
        
        # Combine results
        return {
            "city": city.name,
            "country": country,
            **sunset_results,
            **sunrise_results
        }
    except Exception as e:
        logging.error(f"Error processing city {city_name}: {e}")
        traceback.print_exc()
        return {"city": city_name, "error": str(e)}

def latest_forecast_hours_run_to_download() -> datetime.datetime:
    """
    Determine the latest forecast_hours run to download based on current UTC time.
    ECMWF AIFS runs are at 00z and 12z, with a 10-hour delay for data availability.
    
    Logic:
    - If current UTC time >= 10:00: Download today's 00z run
    - If current UTC time < 10:00: Download yesterday's 12z run
    - If current UTC time >= 22:00 : Download today's 12z run
    Returns:
        forecast_hours initialization time (datetime in UTC)
    """

    now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    threshold_00z = now_utc.replace(hour=10, minute=0, second=0, microsecond=0)
    threshold_12z = now_utc.replace(hour=22, minute=0, second=0, microsecond=0)
    today_end_of_day = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    if now_utc >= threshold_12z and now_utc <= today_end_of_day:
        # Return datetime for downloading today's 12z forecast_hours
        return now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    elif now_utc >= threshold_00z and now_utc < threshold_12z:
        # Return datetime for downloading today's 00z forecast_hours
        return now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Return datetime for downloading yesterday's 12z forecast_hours
        return now_utc.replace(hour=12, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)

def determine_city_forecast_hours_time_to_use(city, city_sunrise_time, city_sunset_time, run_datetime, time_threshold_h = 15) -> dict:
    """
    AIFS forecast_hours are in increments of 6 hours. To determine which forecast_hours time to use for a given city with their sunset time, 
    we need to find the nearest forecast_hours time that is less than or equal to the city sunset time.
    
    Args:
        city: City object with name and timezone
        city_sunset_time: Sunset time in UTC (datetime in UTC)
       
        run_datetime: forecast_hours run initialization time (datetime in UTC)

    Returns:
        Sunrise_forecast_hours_times: Latest run forecast_hours time to use (datetime in UTC) for the city sunrise.
        Sunset_forecast_hours_times: Latest run forecast_hours time to use (datetime in UTC) for the city sunset.
    """
    # Convert city sunset time to UTC
    sunset_time_tdy = city_sunset_time.astimezone(datetime.timezone.utc)
    sunrise_time_tdy = city_sunrise_time.astimezone(datetime.timezone.utc)
    sunset_time_tmr = (city_sunset_time + datetime.timedelta(days=1)).astimezone(datetime.timezone.utc)
    sunrise_time_tmr = (city_sunrise_time + datetime.timedelta(days=1)).astimezone(datetime.timezone.utc)
    threshold_time = run_datetime + datetime.timedelta(hours= time_threshold_h) # default is +15h, which is 15Z for 00Z run and +1 03Z for 12Z run. Do not change, will mess up the logic.
    logging.info(f"determine_city_forecast_hours_time_to_use: run_dt={run_datetime.isoformat()}, threshold_time={threshold_time.isoformat()}, time_threshold_h={time_threshold_h}")

    # Write out the forecast_hours times from run initialization time in 6-hour increments
    offsets = list(range(0,61,6)) # 0 to 60 hours in 6-hour increments
    max_offset_hours = offsets[-1]
    forecast_hours_times = [run_datetime + datetime.timedelta(hours=i) for i in offsets]  # 0 to 60 hours
    last_ft = forecast_hours_times[-1]

    def closest_forecast_hours_time(event_time):
        return min(forecast_hours_times, key=lambda ft: abs(ft - event_time))
    
    def pick_for_event(event_time, name):
        if event_time is None:
            logging.warning(f"{name} is None for {city.name}; returning None")
            return None

        # event before run initialization -> no forecast_hours
        if event_time < run_datetime:
            logging.warning(f"Run not initialised before {name} for {city.name}: event={event_time.isoformat()} run_init={run_datetime.isoformat()}")
            return None
        
        # event before availability threshold -> run not available in time
        if event_time < threshold_time:
            logging.warning(f"Run not available before {name} for {city.name}: event={event_time.isoformat()} threshold={threshold_time.isoformat()}")
            return None

        # ensure event is within forecast_hours window
        if event_time > last_ft + datetime.timedelta(hours=3): # allow 3 hours buffer
            logging.warning(f"{name} for {city.name} outside forecast_hours window [{run_datetime.isoformat()} - {last_ft.isoformat()}]: event={event_time.isoformat()}; returning None")
            return None

        ft = closest_forecast_hours_time(event_time)
        if ft is None:
            logging.warning(f"No forecast_hours step <= {name} for {city.name}; returning None")
            return None
        
        hours_after = int((ft - run_datetime).total_seconds() // 3600)
        # clamp and format
        hours_after = max(0, min(hours_after, offsets[-1]))
        return str(hours_after)

    def warn_promoted_unavailable(event_time, promotion_label):
        if event_time is None:
            return
        hours_ahead = (event_time - run_datetime).total_seconds() / 3600.0
        if hours_ahead > max_offset_hours:
            logging.warning(
                f"Computed time for the now promoted {promotion_label} in {city.name} exceeds +{max_offset_hours}h window; treating the key as None."
            )
    
    forecast_hours = {"city": city.name}

    if sunrise_time_tdy is not None:
        date_tdy = sunrise_time_tdy.date().strftime("%Y%m%d")
        date_tmr = (sunrise_time_tdy + datetime.timedelta(days=1)).date().strftime("%Y%m%d")
        date_day_next = (sunrise_time_tdy + datetime.timedelta(days=2)).date().strftime("%Y%m%d")

        # Only create keys for dates on/after run init
        if sunrise_time_tdy >= run_datetime:
            forecast_hours[f"sunrise_{date_tdy}"] = pick_for_event(sunrise_time_tdy, f"sunrise_{date_tdy}")
        if sunrise_time_tmr >= run_datetime:
            forecast_hours[f"sunrise_{date_tmr}"] = pick_for_event(sunrise_time_tmr, f"sunrise_{date_tmr}")

        need_day_after = False
        if run_datetime < sunrise_time_tdy <= threshold_time:
            need_day_after = True
        elif sunrise_time_tdy < run_datetime:
            logging.info(
                f"sunrise_{date_tdy} for {city.name} occurs before run init; preparing to promote tomorrow's pick."
            )
            need_day_after = True

        if need_day_after:
            next_sunrise_time = sunrise_time_tmr + datetime.timedelta(days=1) if sunrise_time_tmr is not None else None
            key = f"sunrise_{date_day_next}"
            if next_sunrise_time is None:
                logging.warning(f"Unable to compute promoted {key} for {city.name}; missing downstream sunrise time.")
            elif next_sunrise_time >= run_datetime:
                fh_val = pick_for_event(next_sunrise_time, key)
                if fh_val is None:
                    warn_promoted_unavailable(next_sunrise_time, "sunrise_tmr")
                forecast_hours[key] = fh_val
    else:
        logging.error(f"Sunrise times missing for {city.name}")
    
    if sunset_time_tdy is not None:
        date_tdy = sunset_time_tdy.date().strftime("%Y%m%d")
        date_tmr = (sunset_time_tdy + datetime.timedelta(days=1)).date().strftime("%Y%m%d")
        date_day_next = (sunset_time_tdy + datetime.timedelta(days=2)).date().strftime("%Y%m%d")

        # Only create keys for dates on/after run init
        if sunset_time_tdy >= run_datetime:
            forecast_hours[f"sunset_{date_tdy}"] = pick_for_event(sunset_time_tdy, f"sunset_{date_tdy}")
        if sunset_time_tmr >= run_datetime:
            forecast_hours[f"sunset_{date_tmr}"] = pick_for_event(sunset_time_tmr, f"sunset_{date_tmr}")

        need_day_after = False
        if run_datetime < sunset_time_tdy <= threshold_time:
            need_day_after = True
        elif sunset_time_tdy < run_datetime:
            logging.info(
                f"sunset_{date_tdy} for {city.name} occurs before run init; preparing to promote tomorrow's pick."
            )
            need_day_after = True

        if need_day_after:
            next_sunset_time = sunset_time_tmr + datetime.timedelta(days=1) if sunset_time_tmr is not None else None
            key = f"sunset_{date_day_next}"
            if next_sunset_time is None:
                logging.warning(f"Unable to compute promoted {key} for {city.name}; missing downstream sunset time.")
            elif next_sunset_time >= run_datetime:
                fh_val = pick_for_event(next_sunset_time, key)
                if fh_val is None:
                    warn_promoted_unavailable(next_sunset_time, "sunset_tmr")
                forecast_hours[key] = fh_val
    else:
        logging.error(f"Sunset times missing for {city.name}")

    return forecast_hours

def main():
    args = parse_args()

    if args.date:
        today = datetime.datetime.strptime(args.date, "%Y%m%d").date()
    else:
        today = datetime.date.today()

    if args.run is not None:
        run = args.run.zfill(2)
        run_dt = datetime.datetime.combine(today, datetime.time(int(run), 0), tzinfo=datetime.timezone.utc)
    else:
        run_dt = latest_forecast_hours_run_to_download()
        run = str(run_dt.hour).zfill(2)
    print(run)
    today_str = today.strftime("%Y%m%d")
    run_date_str = run_dt.strftime("%Y%m%d")

    # Need global dataset for this
    get_cams_aod_lwc(today, run, run_date_str, input_path) # type: ignore
    
    get_cams_cloud_cover(today, run, today_str, input_path) # type: ignore

    # Load city data
    df = pd.read_csv('worldcities_info_wtimezone.csv', header=0, delimiter=',')

    city_jobs = []
    for _, row in df.iterrows():
        try:
            city_jobs.append(
                (
                    row['city'],
                    row['country'],
                    float(row['lat']),
                    float(row['lng']),
                    row['timezone'],
                    today,
                    run,
                    today_str,
                    run_date_str,
                    input_path,
                    output_path,
                    bool(row.get('create_dashboard', False)),
                )
            )
        except (KeyError, ValueError) as exc:
            logging.error(f"Invalid city row encountered: {exc}", exc_info=True)
    if not city_jobs:
        logging.error("No valid city rows found; aborting processing.")
        return pd.DataFrame()

    cpu_cap = os.cpu_count() or 1
    requested_workers = args.workers if args.workers and args.workers > 0 else 1
    worker_target = min(requested_workers, cpu_cap)
    worker_count = max(1, min(worker_target, len(city_jobs)))
    logging.info(f"Processing {len(city_jobs)} cities using {worker_count} worker(s)...")

    results = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_city = {
            executor.submit(process_city, *args): args[0]
            for args in city_jobs
        }
        for future in as_completed(future_to_city):
            city_name = future_to_city[future]
            try:
                result = future.result()
                results.append(result)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                logging.error(f"Error processing {city_name}: {exc}", exc_info=True)
            except Exception as exc:  # safeguard unexpected issues per city
                logging.error(f"Unexpected error processing {city_name}: {exc}", exc_info=True)

    results_df = pd.DataFrame(results)
    
    # Round all numeric columns to 3 decimal places
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(3)
    
    # Convert likelihood index columns to integers (they should be 0-100, not floats)
    index_cols = [col for col in results_df.columns if 'likelihood_index' in col]
    for col in index_cols:
        results_df[col] = results_df[col].fillna(-1).astype(int).replace(-1, np.nan)
    
    # Export to JSON with controlled precision
    output_json_path = f'{output_path}/all_cities_summary_{run_date_str}_{run}Z.json'
    results_df.to_json(output_json_path, orient='records', indent=2, date_format='iso', double_precision=3)
    logging.info(f"Results saved to {output_json_path}")
    
    return results_df


if __name__ == "__main__":
    results = main()
