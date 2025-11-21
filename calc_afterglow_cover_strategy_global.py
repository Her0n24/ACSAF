"""
Project ACSAF: Aerosols & Cloud geometry based Sunset/Sunrise cloud Afterglow forecaster

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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astral.sun import sun, elevation, azimuth
from astral import LocationInfo
import pytz
import cfgrib
from calc_cloudbase import specific_to_relative_humidity, calc_cloud_base
from get_aifs import download_file
from get_cds_global import get_cams_aod
from geopy import Nominatim
import logging 
logging.basicConfig(
    level=logging.INFO,
    datefmt= '%Y-%m-%d %H:%M:%S',
                    )
from calc_aod_global import calc_aod
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Afterglow forecast")
    parser.add_argument('--date', type=str, default=None,
                        help="Specify the date in YYYYMMDD format (default: today)")
    return parser.parse_args()

output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Afterglow', 'output'))
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Afterglow', 'input'))

logging.info(f"Input path: {input_path}")
logging.info(f"Output path: {output_path}")

def max_solar_elevation(city, date):
    tz = pytz.timezone(city.timezone)
    observer = city.observer

    # Sample every 5 minutes across the day
    times = [datetime.datetime.combine(date, datetime.datetime.min.time()) + datetime.timedelta(minutes=5*i) for i in range(288)]
    times = [tz.localize(t) for t in times]

    max_angle = max(elevation(observer, t) for t in times)
    return max_angle

#function to get the azuimuth angle at sunset
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

def plot_azimuth_line(ax, lon, lat, azimuth_deg, length=5.0):
    """
    Draw a line from (lon, lat) in the direction of azimuth_deg.
    
    Parameters:
    - ax: Matplotlib axis
    - lon, lat: starting point (e.g., city location)
    - azimuth_deg: direction in degrees (0=N, 90=E, etc.)
    - length: length of the line in degrees
    """
    # Convert azimuth to radians
    azimuth_rad = np.deg2rad(azimuth_deg)

    # Estimate the end point of the line
    dx = length * np.sin(azimuth_rad)
    dy = length * np.cos(azimuth_rad)

    end_lon = lon + dx
    end_lat = lat + dy

    # Plot the line
    ax.plot([lon, end_lon], [lat, end_lat], c='orange', lw=2)

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

def calc_cloud_hole_size(z):
    # Equation to calculate the cloud hole size in km
    q = (z/1440)*40076

def extract_variable(ds, var_name, lat_min, lat_max, lon_min, lon_max, verbose=False):
    var = getattr(ds, var_name)
    var = var.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).squeeze()
    if verbose:
        logging.info(f"{var_name}:")
        logging.info(var.values)
    return var


def plot_cloud_cover_map(data_dict, city, lon, lat, today_str, run, title_prefix, fcst_hr, sunset_azimuth, save_path= output_path, cmap='gray'):
    """
    Plot 2x2 cloud cover maps: TCC, LCC, MCC, HCC.

    Parameters:
    - data_dict: dict with keys "tcc", "lcc", "mcc", "hcc" and values as xarray DataArrays
    - city: LocationInfo object (from astral)
    - lon, lat: coordinates of the city
    - title_prefix: string to prefix plot titles
    - save_path: output file path
    - cmap: colormap to use (default: 'gray')
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    var_order = [("tcc", "Total Cloud Cover"),
                 ("lcc", "Low Cloud Cover"),
                 ("mcc", "Mid Cloud Cover"),
                 ("hcc", "High Cloud Cover")]

    mappables = []
    for ax, (key, label) in zip(axs.flat, var_order):
        if key in data_dict:
            mappable = data_dict[key].plot(ax=ax, cmap=cmap, add_colorbar=False, vmin=0, vmax=100)
            mappables.append(mappable)
            ax.set_title(f"{label}")
            ax.scatter(lon, lat, color='red', marker='x', label=city.name)
            plot_azimuth_line(ax, lon, lat, sunset_azimuth, length=5.0)
        else:
            ax.set_visible(False)
            
    for ax in axs[0]:  # second row
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.03, pos.width, pos.height])
            
    for ax in axs[1]:  # second row
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.06, pos.width, pos.height])
                
    if mappables:
        cbar = fig.colorbar(mappables[0], ax=axs, orientation='vertical', fraction=0.03, pad=0.04)
        cbar.set_label("Cloud Cover (%)")

    fig.suptitle(title_prefix, fontsize=14)
    
    # Shared explanation under the title
    fig.text(0.5, 0.92, f"Red × marks {city.name}", ha='center', fontsize=10, color='red')
    fig.text(0.5, 0.90, f"Orange marks sunset azimuth {round(sunset_azimuth,1)}°", ha='center', fontsize=10, color='darkorange')

    plt.savefig(f'{save_path}' + f"/{today_str}{run}0000-{fcst_hr}h-AIFS_cloud_cover_{city.name}.png")
    plt.close()

def azimuth_line_points(lon, lat, azimuth, distance_km, num_points=100):
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

def extract_cloud_cover_along_azimuth(data_dict, lon, lat, azimuth, distance_km, num_points=20):
    """
    Extract cloud cover data along an azimuth line over a certain distance.
    
    Parameters:
    - data_dict: Dictionary containing cloud cover data ("tcc", "lcc", "mcc", "hcc")
    - lon, lat: Starting coordinates of the city
    - azimuth: Azimuth angle in degrees
    - distance_km: Distance along the azimuth line (in km)
    - num_points: Number of points to sample along the azimuth line
    
    Returns:
    - cloud_cover_data: Dictionary with cloud cover data along the azimuth line
    """
    # Generate azimuth line points
    lons, lats = azimuth_line_points(lon, lat, azimuth, distance_km, num_points)
    
    # Interpolate the cloud cover data along the azimuth line
    cloud_cover_data = np.diag(data_dict.interp(longitude=lons, latitude=lats, method='nearest').values) #diagonal entries are the target coordinates (points along the azimuth line)
    
    return cloud_cover_data

def plot_cloud_cover_along_azimuth(cloud_cover_data, azimuth, distance_km, fcst_hr, threshold, cloud_lvl_used, city, today_str, run, save_fig: bool, save_path= output_path):
    """
    Extract cloud cover data along the azimuth path. Plot the cloud cover along the azimuth line.
    
    Parameters:
    - cloud_cover_data: Array containing cloud cover data along the azimuth line
    - azimuth: Azimuth angle in degrees
    - distance_km: Distance along the azimuth line (in km)
    - save_path: Path to save the plot
    """
    # Check for the first distance where cloud cover falls below the threshold
    below_threshold_index = np.argmax(cloud_cover_data <= threshold)
    logging.info(f" below_threshold_index: {below_threshold_index}")

    if below_threshold_index > 0:  # Ensure that the threshold is met somewhere in the data
        distance_below_threshold = np.linspace(0, distance_km, len(cloud_cover_data))[below_threshold_index]
        avg_first_three = np.mean(cloud_cover_data[:3])
        avg_path = np.mean(cloud_cover_data[4:])
        logging.info(f"Local cloud cover is {avg_first_three}%. Average path cloud cover is {avg_path}%.")
        logging.info(f"The cloud cover falls below {threshold}% at {distance_below_threshold} km.")
    else:
        logging.info(f"Cloud cover does not fall below {threshold}%.")
        distance_below_threshold = np.nan
        avg_first_three = np.mean(cloud_cover_data[:3])
        avg_path = np.mean(cloud_cover_data[4:])
        if avg_first_three > 10 and avg_first_three < threshold and avg_path < threshold: # We still think there are cloud if local above 10% total cloud cover
            distance_below_threshold = 250 # Assume a distance below threshold of 250 km
            logging.info(f"Local cloud cover is {avg_first_three}%. Average path cloud cover is {avg_path}%. Meet Criteria even threshold requirement not met.")
            logging.info(f"There is cloud cover above, we assume disance below thres is {distance_below_threshold} km.")
            
    if save_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the cloud cover data
        ax.plot(np.linspace(0, distance_km, len(cloud_cover_data)), cloud_cover_data, label=f"Cloud Cover ({cloud_lvl_used})")
        
        # Plot the threshold line
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold}% threshold')
        
        if avg_first_three > 10 and avg_first_three < threshold and avg_path < threshold:
            ax.axhline(y=avg_first_three, color='orange', linestyle='--', label=f'Local cloud cover {avg_first_three}%')
            
        if ~np.isnan(distance_below_threshold):
            ax.axvline(x=distance_below_threshold, color='r', linestyle='--', label=f'{threshold}% threshold')
        
        ax.set_ylim(0,100)
        ax.set_xlabel('Distance along Azimuth (km)')
        ax.set_ylabel('Cloud Cover (%)')
        ax.set_title(f'{today_str} {run}z +{fcst_hr}h EC AIFS cloud cover Along Azimuth path ')
        #set subtitle
        ax.text(0.5, 1.02, f"Azimuth {azimuth}°", fontsize=10)
        ax.legend()
        
        plt.savefig(save_path + f"/{today_str}{run}0000_{fcst_hr}h_AIFS_cloud_cover_azimuth_{city.name}.png")
        plt.close()
    return distance_below_threshold, avg_first_three, avg_path


def get_cloud_extent(data_dict, city, lon, lat, azimuth, cloud_base_lvl: float, fcst_hr, plot_graph_flag, today_str, run, distance_km = 500, num_points=25, threshold=60.0):
    priority_order = [("lcc",2000), ("mcc",4000), ("hcc",6000)]
    cloud_lvl_used = None
    cloud_present = True
    local_cloud_cover_dict = {}

    # Initialize these variables to avoid unbound
    avg_first_three_met = False
    hcc_condition = False
    data = None
    distance_below_threshold = np.nan
    key = None
    avg_first_three = np.nan
    avg_path = np.nan
    
    for key, base_height in priority_order:
        if key == 'lcc' or key == 'mcc':
            #Extract cloud_cover data along the azimuth
            cloud_cover_data = extract_cloud_cover_along_azimuth(
                data_dict[key], lon, lat, azimuth, distance_km, num_points
            )
            # Calculate the average of the first 3 indices
            avg_first_three = np.nanmean(cloud_cover_data[:3])
            local_cloud_cover_dict[key] = cloud_cover_data[0]

            if avg_first_three > threshold:
                cloud_lvl_used = key 
                data = data_dict[key]  # Return the first key that meets the criteria
                avg_first_three_met = True
                hcc_condition = False
            else:
                avg_first_three_met = False
                hcc_condition = False
        if key == 'hcc':
            #Extract cloud_cover data along the azimuth
            cloud_cover_data = extract_cloud_cover_along_azimuth(
                data_dict[key], lon, lat, azimuth, distance_km, num_points
            )
            # Calculate the average of the first 3 indices
            avg_first_three = np.nanmean(cloud_cover_data[:3])
            local_cloud_cover_dict[key] = cloud_cover_data[0]

            avg_path = np.nanmean(cloud_cover_data[4:])
            if avg_first_three > threshold:
                cloud_lvl_used = key 
                data = data_dict[key]
                avg_first_three_met = True
                hcc_condition = False
            else:
                hcc_condition = False
                avg_first_three_met = False
            if avg_first_three < avg_path:
                avg_first_three_met = True
                logging.info(f"Average first three cloud cover {avg_first_three}% is less than average path cloud cover {avg_path}%.")
                if avg_path < 85.00:
                    cloud_lvl_used = key 
                    data = data_dict[key]
                    hcc_condition = True # hcc condition is added to compensate for high clouds, it might be translucent enough for sunrays to pass through
                    # an extensive layer of high clouds. Hence, we lowered the reqruiement for avg_path to be below 85% for afterglows to be possible. 
                    avg_first_three_met = False
                    logging.info("HCC condition met")
                else:
                    hcc_condition = False
                    avg_first_three_met = False
                    logging.info("HCC condition not met")
    
    # Check if cloud base level is NAN. We assign the cloud base level to the first layer with cloud cover >= 15% if it is NaN.
    # We pick the first (lowest) layer so we put a break 
    if np.isnan(cloud_base_lvl):
        for key, base_height in priority_order:
            if local_cloud_cover_dict.get(key, 0) >= 15:
                cloud_base_lvl = base_height
                break

    # Check if data is empty
    if avg_first_three_met is False:
        cloud_lvl_used = 'tcc' 
        data = data_dict['tcc']
        logging.info(f"RH requirement not met. Cloud base level {cloud_base_lvl} is assumed based on the first layer with cloud cover >= 15%")
        logging.info("Trying to use tcc for cloud cover data")
        data = data_dict['tcc']
        cloud_lvl_used = 'tcc'
        try:
            cloud_cover_data = extract_cloud_cover_along_azimuth(data, lon, lat, azimuth, distance_km, num_points)
            distance_below_threshold, avg_first_three, avg_path  = plot_cloud_cover_along_azimuth(cloud_cover_data, azimuth, distance_km, fcst_hr, threshold, cloud_lvl_used, city, today_str, run, plot_graph_flag, save_path= output_path)
        except:
            logging.error(f"Cloud cover data is not available for forecast_hours hour {fcst_hr}.")
            cloud_present = False
    # # Check if data is associated with a value or a NaN
    # if data is None:
    #     logging.info(f"Cloud cover NaN for forecast_hours hour {fcst_hr}. There is no stratiform cloud cover. Afterglow not probable.")
    #     cloud_present = False
    #     return cloud_present
    if hcc_condition is True:
        logging.info('hcc condition is True. We will assume a distance below threshold of 150 km.')
        cloud_cover_data = extract_cloud_cover_along_azimuth(data, lon, lat, azimuth, distance_km, num_points)
        distance_below_threshold = 150
        avg_first_three = np.nanmean(cloud_cover_data[:3])
        avg_path = np.nanmean(cloud_cover_data[4:])
    elif hcc_condition is False:
        cloud_cover_data = extract_cloud_cover_along_azimuth(data, lon, lat, azimuth, distance_km, num_points)
        distance_below_threshold, avg_first_three, avg_path = plot_cloud_cover_along_azimuth(cloud_cover_data, azimuth, distance_km, fcst_hr, threshold, cloud_lvl_used, city, today_str, run, plot_graph_flag, save_path= output_path)
    distance_below_threshold = distance_below_threshold * 1000 # convert to meters

    if avg_first_three < 10.0:
        cloud_present = False

    return distance_below_threshold, key, avg_first_three, avg_path, cloud_present, cloud_base_lvl

def geom_condition(cloud_base_height, cloud_extent, LCL):
    """
    Calculate the geometric condition for afterglow based on cloud base height and cloud extent.
    
    Parameters:
    - cloud_base_height: Cloud base height (hPa)
    - cloud_extent: Cloud extent (m)
    
    Returns:
    - geom_condition: Geometric condition for afterglow
    """
    # Cosntant
    R = 6371*10**3  # Radius of the Earth in meters
    lf_ma = 2*np.sqrt(2*R*cloud_base_height) 
    try:
        logging.info(f"lf_ma: {round(lf_ma)} m")
        geom_condition_LCL_used = False
    except ValueError as e:
        logging.error(f"Error: {e}")
        logging.error(f"lf_ma is {lf_ma}")
        logging.error('We will assume lf_ma using LCL')
        lf_ma = 2*np.sqrt(2*R*LCL)
        logging.info(f"lf_ma: {round(lf_ma)} m")
        geom_condition_LCL_used = True
        geom_condition = False
    # Compare with cloud_extent
    if lf_ma > cloud_extent:
        geom_condition = True
    else:
        geom_condition = False
    logging.info("cloud geometry condition:", geom_condition)
    return geom_condition, geom_condition_LCL_used, lf_ma


def get_afterglow_time(lat, today, distance_below_threshold, lf_ma, cloud_base_lvl, z_lvl):
    
    day_of_year = today.timetuple().tm_yday
    
    R = 6371  # Earth's radius in km
    T = 1440  # minutes in a day

    if np.isnan(cloud_base_lvl):
        cloud_base_lvl = z_lvl
    if lf_ma >= distance_below_threshold:
        # Convert latitude to radians
        phi = np.deg2rad(lat)

        # Approximate solar declination angle (in degrees then radians)
        decl_deg = 23.44 * np.sin(np.deg2rad((360 / 365) * (day_of_year - 81)))
        delta = np.deg2rad(decl_deg)

        # Linear speed of sunrays at given latitude and time of year based on https://doi.org/10.1016/j.asr.2023.08.036
        speed = (2 * np.pi * R * np.cos(phi)) / T * np.cos(delta) #In km/minutes
        
        total_afterglow_time = np.sqrt((R*cloud_base_lvl))/speed #minutes
        
        # Make a straight line between origin and point (total_afterglow_time, lf_ma)
        
        t1 =  -distance_below_threshold * (total_afterglow_time/lf_ma) # rearrange purple y=mx
        t2 =  total_afterglow_time + (-distance_below_threshold)*(2*total_afterglow_time/lf_ma) # rearrange purple y=mx
        
        overhead_afterglow_time = np.abs(t1-t2)
        
        actual_afterglow_time = total_afterglow_time - overhead_afterglow_time
        
        actual_afterglow_time = actual_afterglow_time + ((cloud_base_lvl/np.tan(np.deg2rad(15)))/(21*1000/60)) # Accounting for the clouds in visual contact assuming 15 deg elevation
        
        logging.info(f"Total Afterglow time: {total_afterglow_time} seconds")
        logging.info(f"Overhead Afterglow time: {overhead_afterglow_time} seconds")
        logging.info(f"Actual Afterglow time: {actual_afterglow_time} seconds")
    else:
        logging.info(f"Sun ray extent lf_max is less than cloud extent. No afterglow is possible.")
        actual_afterglow_time = 0
    return actual_afterglow_time


def possible_colours(cloud_base_lvl, lcl_lvl, total_aod_550, key):
    """
    Determine the possible colours for the afterglow based on cloud base level, inferred cloud cover and AOD_550.
    Cloud cover is inferred through RH of the cloud base level. 
    
    Parameters:
    """
    color = ('none',)
    if np.isnan(cloud_base_lvl):
        if np.isnan(lcl_lvl):
            color = ('none',)
        else:
            cloud_base_lvl = lcl_lvl
    if cloud_base_lvl <= 2000.0:
        if total_aod_550 <= 0.2:
            color = ('orange-red',)
        if key == 'lcc':
            color = ('orange-red',)
        elif key == 'mcc':
            color = ('orange-red', 'dark-red', 'crimson',)
        elif key == 'hcc':
            color = ('orange-red', 'dark-red', 'magenta',)
        else:
            color = ('dirty-orange',)
    elif cloud_base_lvl > 2000.0 and cloud_base_lvl <= 6000.0:
        if key == 'lcc':
            color = ('orange-red', 'dark-red',)
        if key == 'mcc':
            color = ('orange-red', 'dark-red', 'crimson',)
        if key == 'hcc':
            color = ('orange-yellow', 'golden-yellow', 'magenta',)
    elif cloud_base_lvl > 6000.0:
        color = ('golden-yellow', 'crimson', 'magenta',)
        
    return color

def get_elevation_afterglow(cloud_base_lvl, distance_below_threshold, lf_ma, lcl):
    """
    Calculate the elevation angle for afterglow based on cloud base height and distance below threshold.
    
    Parameters:
    - cloud_base_lvl: Cloud base level (m)
    - distance_below_threshold: Distance below threshold (m)
    - lf_ma: Distance to the cloud base (m)
    - lcl: Lifted condensation level (m)
    
    returns:
    - theta: Elevation angle (radians)

    """
    #Constants
    R = 6371*10**3  # Radius of the Earth in meters
    
    if distance_below_threshold == False or np.isnan(distance_below_threshold):
        logging.info("Cloud cover and elevation estimated using tcc")
        theta = np.arctan((cloud_base_lvl-(lf_ma**2/(2*R)))/lf_ma)
        logging.info(f"Elevation angle: {np.rad2deg(theta)}°")
        return theta
    elif np.isnan(cloud_base_lvl):
        logging.info("Cloud cover and elevation estimated using tcc and LCL")
        theta = np.arctan((lcl-((distance_below_threshold**2)/(2*R)))/distance_below_threshold)
        logging.info(f"Elevation angle: {np.rad2deg(theta)}°")
        return theta
    else:
        theta = np.arctan((cloud_base_lvl-(distance_below_threshold**2/(2*R)))/distance_below_threshold)
        logging.info(f"Elevation angle: {np.rad2deg(theta)}°")
        return theta


def signed_power(x, p):
    return np.sign(x) * (abs(x) ** p)

# Color fill based on index using the 'plasma' colormap
def color_fill(index):
    # Normalize the index to a value between 0 and 1 for the colormap
    norm = plt.Normalize(vmin=0, vmax=100) #type: ignore
    cmap = plt.get_cmap('magma')  # 
    return cmap(norm(index))


def create_dashboard(today, today_str, run, index_today, index_tomorrow, city, latitude, longitude,
                     azimuth, afterglow_length_tdy, afterglow_length_tmr, possible_colors_tdy,
                     possible_colors_tmr, cloud_base_lvl_tdy, cloud_base_lvl_tmr, 
                     z_lcl_tdy, z_lcl_tmr, cloud_present_tdy, cloud_present_tmr, total_aod550,
                     sunset, sunset_tmr, fcst_hr_tdy: str, fcst_hr_tmr: str):
    if np.isnan(cloud_base_lvl_tdy):
        cloud_base_lvl_tdy = z_lcl_tdy
    if np.isnan(cloud_base_lvl_tmr):
        cloud_base_lvl_tmr = z_lcl_tmr
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Adjust size as needed

    # Set background colour for each subplot
    for a in ax:
        a.set_facecolor('black')  # Each 'a' is an individual axis
        a.set_xticks([])
        a.set_yticks([])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.spines['bottom'].set_visible(False)

    # Also set the figure background
    fig.patch.set_facecolor('black')
    
    plt.suptitle("ACSAF: Aerosol and Cloud geometry based Sunset cloud Afterglow forecaster", fontsize=16, weight='bold', color='white')
        
    # Box settings for today and tomorrow
    # Larger, square boxes with index numbers only in bold
    ax[0].text(0.1, 0.5, f"{index_today}", fontsize=40, fontweight='bold', ha='center', va='center', 
            color='white', bbox=dict(facecolor=color_fill(index_today), edgecolor='black', boxstyle='round,pad=2'))

    ax[0].text(1.0, 0.5, f"{index_tomorrow}", fontsize=40, fontweight='bold', ha='right', va='center',
            color='white', bbox=dict(facecolor=color_fill(index_tomorrow), edgecolor='black', boxstyle='round,pad=2'))

    # Add text outside the box for "Today" and "Tomorrow"
    ax[0].text(0.1, 0.1, "Today", fontsize=18, ha='center', va='center', color='white', fontweight='bold')
    ax[0].text(0.9, 0.1, "Tomorrow", fontsize=18, ha='center', va='center', color='white', fontweight='bold')

    if cloud_present_tdy is False:
        ax[0].text(0.1, 0.3, "No cloud cover", fontsize=15, ha='center', va='center', color='white')
    if cloud_present_tmr is False:
        ax[0].text(0.9, 0.3, "No cloud cover", fontsize=15, ha='center', va='center', color='white')

    # Title for the left subplot
    ax[0].axis('off')  # Turn off axis for this subplot


    sunset_tz = pytz.timezone("Europe/London")  # Desired timezone for Reading (UK)

    # Convert the sunset time from UTC to local time
    sunset_local = sunset.astimezone(sunset_tz)
    sunset_local_42 = sunset_tmr.astimezone(sunset_tz)
    
    # Info text on the right

    info_text = (
        f"Today: {today.strftime('%Y-%m-%d')}\n"
        f"Location: {city.name}\n"
        f"Sunset Time: {sunset_local.strftime('%H:%M:%S')} \n"
        f"Sunset Azimuth: {round(azimuth)}°\n"
        f"Length: {round(afterglow_length_tdy)} s\n"
        f"Cloud Height : {int(round(cloud_base_lvl_tdy, -2))} m\n"
        f"Aerosol OD(550nm): {total_aod550[0]:.2f} \n"
        f"Colors: {', '.join(possible_colors_tdy) }\n"
    )
    
    ax[1].text(0.7, 0.6, info_text, fontsize=15, ha='center', va='center', color='black', 
           bbox=dict(facecolor='lightgray', alpha=0.8))
    
    # Info text on the right
    info_text = (
        f"Tomorrow: {(today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')}\n"
        f"Length: {round(afterglow_length_tmr)} s\n"
        f"Cloud Height : {int(round(cloud_base_lvl_tmr, -2))} m\n"
        f"Aerosol OD(550nm): {total_aod550[1]:.2f} \n"
        f"Colors: {', '.join(possible_colors_tmr)}\n"
    )
    ax[1].text(0.7, 0.15, info_text, fontsize=15, ha='center', va='center', color='black', 
           bbox=dict(facecolor='lightgray', alpha=0.8))

    ax[1].axis('off')  # Turn off axis for this subplot
        
    # Title
    fig.text(0.01, 0.01, "Index ranges 0-100. Higher indicates more favourable cloud afterglow conditions for occurence and intense colors.\n"
        "Based on daily 00z ECMWF AIFS and Copernicus Atmosphere Monitoring Service forecasts. Valid only for stratiform cloud layer. See supplementary figures for details.",
         ha='left', va='bottom', color='white', fontsize=8)
    
    fig.text(0.89, 0.01, f"Plots by A350XWBoy. V2025.9.7", color='white',fontsize=8)

    plt.savefig(f'{output_path}/{today_str}{run}0000_afterglow_dashboard_{city.name}.png', dpi=400)

# Weighted likelihod index
def weighted_likelihood_index(geom_condition, aod, dust_aod_ratio, cloud_base_lvl, lcl_lvl, theta, avg_first_three, avg_path):
    """
    Calculate the weighted likelihood index based on AOD, afterglow time, cloud base level, geometric condition, and cloud extent.
    
    Parameters:
    - geom_condition: Geometric condition for afterglow (True/False)
    - aod: AOD value
    - cloud_base_lvl: Cloud base level (m)
    - max_RH: Maximum relative humidity (%)
    - theta: Elevation angle (radians)
    
    Returns:
    - likelihood_index: Weighted likelihood index for afterglow
    """
    #Initialize scores
    cloud_cover_score, cloud_base_score, aod_score, dust_ratio_score, theta_score = np.nan, np.nan, np.nan, np.nan, np.nan

    if np.isnan(cloud_base_lvl) or cloud_base_lvl <= 0:
        cloud_base_lvl = lcl_lvl
        logging.info("Used LCL level for computation")
        max_lvl = 6000.0
        norm_lvl = min(cloud_base_lvl, max_lvl) / max_lvl
        cloud_base_score = (norm_lvl)
        logging.info(f"cloud base score using LCL: {cloud_base_score}")
    else:
        max_lvl = 6000.0
        norm_lvl = min(cloud_base_lvl, max_lvl) / max_lvl
        cloud_base_score = (norm_lvl)
    
    if aod >= 0 and aod <= 0.3:
        aod_score = 1
    elif aod > 0.3 and aod <= 0.5:
        aod_score = 0.8
    elif aod > 0.5 and aod <= 0.7:
        aod_score = 0.2
    elif aod > 0.7:
        aod_score = -0.5
    
    if dust_aod_ratio >= 0 and dust_aod_ratio <= 0.2:
        dust_ratio_score = 0.2
    elif dust_aod_ratio > 0.2 and dust_aod_ratio <= 0.4:
        dust_ratio_score = 0.8
    elif dust_aod_ratio > 0.4:
        dust_ratio_score = 1
        
    if theta >= 0 and theta <= np.deg2rad(5):
        theta_score = 0.4
    elif theta > np.deg2rad(5):
        theta_score = 1
    elif theta < 0 and theta > np.deg2rad(-0.5):
        theta_score = -0.8
    elif theta <= np.deg2rad(-0.5):
        theta_score = -1
    elif np.isnan(theta):
        theta_score = 0 
    
    if np.isnan(cloud_base_lvl):
        cloud_base_lvl = lcl_lvl

    if (cloud_base_lvl < 2000 and cloud_base_lvl > 0 ):
        avg_first_three = min(avg_first_three, 100)
        avg_path = min(avg_path, 100)

        x = avg_first_three / 100.0  # normalised
        y = avg_path / 100.0         # normalised
        
        cloud_cover_score = 0.5 * x - 0.5 * y
    
    if cloud_base_lvl < 4000 and cloud_base_lvl >= 2000:
        avg_first_three = min(avg_first_three, 100)
        avg_path = min(avg_path, 100)

        x = avg_first_three / 100.0  # normalised
        y = avg_path / 100.0         # normalised
        
        cloud_cover_score = 0.5 * x - 0.5 * y
        
    if cloud_base_lvl >= 4000:
        avg_first_three = min(avg_first_three, 100)
        avg_path = min(avg_path, 100)

        x = avg_first_three / 100.0  # normalised
        y = avg_path / 100.0         # normalised

        logging.info(f"x_score: {x}, y_score: {y}")
        
        cloud_cover_score = 0.4 * x - 0.6 * y
    
    logging.info(f"Cloud cover score: {cloud_cover_score}")

    # import pdb; pdb.set_trace()  # Debugging point to inspect variables
    
    if geom_condition == True:
        # Constants
        geom_condition_weight = 0.1
        aod_weight = 0.15
        dust_aod_ratio_weight = 0.05
        cloud_cover_weight = 0.4
        cloud_base_lvl_weight = 0.25
        theta_weight = 0.05   
    else:
        # Constants
        geom_condition_weight = 0.5
        aod_weight = 0.05
        dust_aod_ratio_weight = 0.05
        cloud_cover_weight = 0.2
        cloud_base_lvl_weight = 0.15
        theta_weight = 0.05
        
    # Binary flag for geom_condition
    geom_flag = 1 if geom_condition else 0
    # Compute weighted index
    likelihood_index = (
    signed_power(geom_flag, 1) * geom_condition_weight +
    signed_power(aod_score, 2) * aod_weight +
    signed_power(dust_ratio_score, 3) * dust_aod_ratio_weight +
    signed_power(cloud_cover_score, 1) * cloud_cover_weight +
    signed_power(cloud_base_score, 2) * cloud_base_lvl_weight +
    signed_power(theta_score, 2) * theta_weight # Higher power for less importance 
    )
    logging.info(f"geom_flag: {geom_flag}, contribution: {signed_power(geom_flag, 1) * geom_condition_weight}")
    logging.info(f"aod_score: {aod_score}, contribution: {signed_power(aod_score, 2) * aod_weight}")
    logging.info(f"dust_ratio_score: {dust_ratio_score}, contribution: {signed_power(dust_ratio_score, 3) * dust_aod_ratio_weight}")
    logging.info(f"cloud_cover_score: {cloud_cover_score}, contribution: {signed_power(cloud_cover_score, 1) * cloud_cover_weight}")
    logging.info(f"cloud_base_score: {cloud_base_score}, contribution: {signed_power(cloud_base_score, 2) * cloud_base_lvl_weight}")
    logging.info(f"theta_score: {theta_score}, contribution: {signed_power(theta_score, 2) * theta_weight}")

    likelihood_index = np.clip(likelihood_index, 0, 1)  # Ensure it's between 0 and 1
    
    # Scale to 0-100 and round to whole number
    likelihood_index = round(likelihood_index * 100)
    return likelihood_index

def process_city(city_name: str, country: str, lat: float, lon: float, timezone_str: str, today, run, today_str, input_path, output_path, create_dashboard_flag: bool):
    tz = pytz.timezone(timezone_str)
    #current_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    current_utc = datetime.datetime(2025, 11, 20, 10, 0, 0, tzinfo=datetime.timezone.utc)  # Fixed date for testing
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

    try:
        s_tdy = sun(city.observer, date=local_today, tzinfo=tz)
        sunset_tdy = s_tdy['sunset']
        sunrise_tdy = s_tdy['sunrise']

        s_tmr = sun(city.observer, date=local_today + datetime.timedelta(days=1), tzinfo=tz)
        sunrise_tmr = s_tmr['sunrise']
        sunset_tmr = s_tmr['sunset']
    except Exception as e:
        logging.warning(f"Sun calculation failed for {city.name}: {e}")
        # handle polar/no-sun or skip city
        sunrise_tdy = sunset_tdy = sunrise_tmr = sunset_tmr = None

    #run_dt = latest_forecast_hours_run_to_download()
    run_dt = datetime.datetime(2025, 11, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)  # Fixed date for testing
    picks = determine_city_forecast_hours_time_to_use(city, sunrise_tdy, sunset_tdy, run_dt)
    print("run_dt:", run_dt)
    print("picsks:", picks)

    sunset_fh_tdy = picks.get(f'sunset_{today_str}') if today_str else None
    sunset_fh_tmr = picks.get(f'sunset_{tomorrow_str}') if tomorrow_str else None

    # check for day-after-tomorrow pick and allow shifting when today is missing
    day_after_str = (datetime.datetime.strptime(today_str, "%Y%m%d").date() + datetime.timedelta(days=2)).strftime("%Y%m%d")
    sunset_fh_day_after = picks.get(f"sunset_{day_after_str}") if picks else None

    # if today's pick missing: use tomorrow's as today's, and shift tomorrow -> day-after if available
    if sunset_fh_tdy is None and sunset_fh_tmr is not None:
        logging.info(f"No pick for sunset_{today_str} for {city.name}; using tomorrow's pick ({sunset_fh_tmr}) as today's.")
        sunset_fh_tdy = sunset_fh_tmr
        sunset_fh_tmr = sunset_fh_day_after if sunset_fh_day_after is not None else None
    
    logging.info(f"Final picks for {city.name}: today +{sunset_fh_tdy}h, tomorrow +{sunset_fh_tmr}h")

    logging.info(city.name)

    max_elev = max_solar_elevation(city, datetime.date.today())
    logging.info(f"Maximum solar elevation in {city.name}: {max_elev:.2f}°")

    sunset_azimuth = get__sunset_azimuth(city, today)
    sunset_azimuth_tmr = get__sunset_azimuth(city, today + datetime.timedelta(days=1))
    logging.info(f"Sunset azimuth angle in {city.name}: {sunset_azimuth:.2f}°")
    logging.info(f"sunrise time in {city.name}: {sunrise_tdy}")
    logging.info(f"sunset time in {city.name}: {sunset_tdy}")

    # Define a square box enclosing the region of interest
    lat_min, lat_max = lat - 3, lat + 3
    lon_min, lon_max = lon - 5, lon + 1

    if sunset_fh_tdy is None:
        logging.info(f"No pick for sunset_{today_str} for {city.name}; leaving None")
    if sunset_fh_tmr is None:
        logging.info(f"No pick for sunset_{tomorrow_str} for {city.name}; leaving None")
    
    # final guard: if still missing, skip city
    if sunset_fh_tdy is None:
        logging.warning(f"No forecast-hour file available for {city.name} (today); skipping city.")
        return {"city": city.name, "error": "missing forecast-hour file"}

    ds_tdy = xr.open_dataset(f'{input_path}/{today_str}{run}0000-{sunset_fh_tdy}h-oper-fc.grib2', engine = 'cfgrib')
    ds_tmr = xr.open_dataset(f'{input_path}/{today_str}{run}0000-{sunset_fh_tmr}h-oper-fc.grib2', engine = 'cfgrib')

    # t2m at 2m
    # Please fix the cfgrib datasetbuilderror: key present and new value is different: key='heightAboveGround' value=Variable(dimensions=(), data=np.float64(10.0)) new_value=Variable(dimensions=(), data=np.float64(100.0))
    ds_tdy_2m = cfgrib.open_dataset(f'{input_path}/{today_str}{run}0000-{sunset_fh_tdy}h-oper-fc.grib2', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
    ds_tmr_2m = cfgrib.open_dataset(f'{input_path}/{today_str}{run}0000-{sunset_fh_tmr}h-oper-fc.grib2', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})

    ds_tdy_lcc = extract_variable(ds_tdy, "lcc", lat_min, lat_max, lon_min, lon_max)
    ds_tdy_mcc = extract_variable(ds_tdy, "mcc", lat_min, lat_max, lon_min, lon_max)
    ds_tdy_hcc = extract_variable(ds_tdy, "hcc", lat_min, lat_max, lon_min, lon_max)
    ds_tdy_tcc = extract_variable(ds_tdy, "tcc", lat_min, lat_max, lon_min, lon_max)

    ds_tmr_tcc = extract_variable(ds_tmr, "tcc", lat_min, lat_max, lon_min, lon_max)
    ds_tmr_lcc = extract_variable(ds_tmr, "lcc", lat_min, lat_max, lon_min, lon_max)
    ds_tmr_mcc = extract_variable(ds_tmr, "mcc", lat_min, lat_max, lon_min, lon_max)
    ds_tmr_hcc = extract_variable(ds_tmr, "hcc", lat_min, lat_max, lon_min, lon_max)

    cloud_vars_tdy = {
        "tcc": ds_tdy_tcc,
        "lcc": ds_tdy_lcc,
        "mcc": ds_tdy_mcc,
        "hcc": ds_tdy_hcc
    }

    cloud_vars_tmr = {
        "tcc": ds_tmr_tcc,
        "lcc": ds_tmr_lcc,
        "mcc": ds_tmr_mcc,
        "hcc": ds_tmr_hcc
    }

    if create_dashboard_flag:
        plot_cloud_cover_map(cloud_vars_tdy, city, lon, lat, today_str, run,
                        f'{today_str} {run}z +{sunset_fh_tdy}h EC AIFS cloud cover (today sunset)',
                        sunset_fh_tdy, sunset_azimuth, save_path= output_path, cmap='gray')

        plot_cloud_cover_map(cloud_vars_tmr, city, lon, lat, tomorrow_str, run,
                            f'{tomorrow_str} {run}z +{sunset_fh_tmr}h EC AIFS cloud cover (tomorrow sunset)',
                            sunset_fh_tmr, sunset_azimuth, save_path= output_path, cmap='gray')

    RH_tdy, p_18 = specific_to_relative_humidity(ds_tdy.q, ds_tdy.t, ds_tdy.isobaricInhPa, lat, lon)
    cloud_base_lvl_tdy, z_lcl_tdy, RH_cb_tdy = calc_cloud_base(ds_tdy_2m["t2m"], ds_tdy_2m["d2m"], ds_tdy.t, RH_tdy, ds_tdy.isobaricInhPa, lat, lon)

    RH_tmr, p_42 = specific_to_relative_humidity(ds_tmr.q, ds_tmr.t, ds_tmr.isobaricInhPa, lat, lon)
    cloud_base_lvl_tmr, z_lcl_tmr, RH_cb_tmr = calc_cloud_base(ds_tmr_2m["t2m"], ds_tmr_2m["d2m"], ds_tmr.t, RH_tmr, ds_tmr.isobaricInhPa, lat, lon)

    logging.info(f'tdy cloud_base:{cloud_base_lvl_tdy}, z_lcl:{z_lcl_tdy}, RH_cb:{RH_cb_tdy}')
    logging.info(f'tmr cloud_base: {cloud_base_lvl_tmr}, z_lcl:{z_lcl_tmr}, RH_cb:{RH_cb_tmr}')

    # cloud_cover_data = extract_cloud_cover_along_azimuth(ds_tdy_tcc, lon, lat, sunset_azimuth, 500, num_points=20)
    # cloud_cover_data_42 = extract_cloud_cover_along_azimuth(ds_tmr_tcc, lon, lat, sunset_azimuth_42, 500, num_points=20)

    # logging.info(cloud_cover_data)
    # logging.info(cloud_cover_data_42)

    # cloud_cover_data_tdy_all = {
    #     key: extract_cloud_cover_along_azimuth(data, lon, lat, sunset_azimuth, 500, num_points=20)
    #     for key, data in cloud_vars_tdy.items()
    # }

    # # Apply the function to all datasets in cloud_vars_tmr
    # cloud_cover_data_tmr_all = {
    #     key: extract_cloud_cover_along_azimuth(data, lon, lat, sunset_azimuth_tmr, 500, num_points=20)
    #     for key, data in cloud_vars_tmr.items()
    # }

    # # Print the results for verification
    # logging.info("Cloud cover data for +18h forecast_hours:")
    # logging.info(cloud_cover_data_18_all)

    distance_below_threshold_tdy, key_tdy, avg_first_three_tdy, avg_path_tdy, cloud_present_tdy, cloud_base_lvl_tdy  = get_cloud_extent(cloud_vars_tdy, city, lon, lat, sunset_azimuth, cloud_base_lvl_tdy, sunset_fh_tdy, 
                                                                                                                                        create_dashboard_flag, today_str, run)
    distance_below_threshold_tmr, key_tmr, avg_first_three_tmr, avg_path_tmr, cloud_present_tmr, cloud_base_lvl_tmr  = get_cloud_extent(cloud_vars_tmr, city, lon, lat, sunset_azimuth_tmr, cloud_base_lvl_tmr, sunset_fh_tmr, 
                                                                                                                                        create_dashboard_flag, tomorrow_str, run)

    geom_cond_tdy, geom_condition_LCL_used_tdy, lf_ma_tdy = geom_condition(cloud_base_lvl_tdy, distance_below_threshold_tdy, z_lcl_tdy)
    geom_cond_tmr, geom_condition_LCL_used_tmr, lf_ma_tmr = geom_condition(cloud_base_lvl_tmr, distance_below_threshold_tmr, z_lcl_tmr)

    theta_tdy = get_elevation_afterglow(cloud_base_lvl_tdy, distance_below_threshold_tdy, lf_ma_tdy, z_lcl_tdy)
    theta_tmr = get_elevation_afterglow(cloud_base_lvl_tmr, distance_below_threshold_tmr, lf_ma_tmr, z_lcl_tmr)

    actual_afterglow_time_tdy = get_afterglow_time(lat, today, distance_below_threshold_tdy, lf_ma_tdy, cloud_base_lvl_tdy, z_lcl_tdy)
    actual_afterglow_time_tmr = get_afterglow_time(lat, tomorrow, distance_below_threshold_tmr, lf_ma_tmr, cloud_base_lvl_tmr, z_lcl_tmr)

    # One method is to use Equivalent cloud heigh  = cloud base level - equivalent surface height as highlighted in the paper
    # But uncertainty is high and the exact mechanism is not clear
    # Here we use a simplier and quicker method, but yet to be verified
    # The AOD value directly controls the value of the afterglow liklihood index in the weighted equation

    # Incorporate AOD

    dust_aod550, total_aod550, dust_aod550_ratio = calc_aod(run, today_str, input_path) #Array of shape (2,) first is 18h , second is 42h

    likelihood_index_tdy = weighted_likelihood_index(geom_cond_tdy, total_aod550[0], dust_aod550_ratio[0], cloud_base_lvl_tdy, z_lcl_tdy, theta_tdy, avg_first_three_tdy, avg_path_tdy)
    likelihood_index_tmr = weighted_likelihood_index(geom_cond_tmr, total_aod550[1], dust_aod550_ratio[1], cloud_base_lvl_tmr, z_lcl_tmr, theta_tmr, avg_first_three_tmr, avg_path_tmr)
    logging.info(f"{city.name} likelihood_index_tdy: {likelihood_index_tdy}")
    logging.info(f"{city.name} likelihood_index_tmr: {likelihood_index_tmr}")

    possible_colors_tdy = possible_colours(cloud_base_lvl_tdy, z_lcl_tdy, total_aod550[0], key_tdy)
    possible_colors_tmr = possible_colours(cloud_base_lvl_tmr, z_lcl_tmr, total_aod550[1], key_tmr)
    logging.info(f"Possible colors for afterglow {sunset_fh_tdy}h: {possible_colors_tdy}")
    logging.info(f"Possible colors for afterglow {sunset_fh_tmr}h: {possible_colors_tmr}")

    logging.info(cloud_base_lvl_tdy)
    logging.info(cloud_base_lvl_tmr)

    if create_dashboard_flag:
        create_dashboard(
            today, today_str, run, likelihood_index_tdy, likelihood_index_tmr, city, lat, lon, sunset_azimuth, actual_afterglow_time_tdy, actual_afterglow_time_tmr, possible_colors_tdy, possible_colors_tmr, cloud_base_lvl_tdy, cloud_base_lvl_tmr,
            z_lcl_tdy, z_lcl_tmr, cloud_present_tdy, cloud_present_tmr, total_aod550, sunset_tdy, sunset_tmr, fcst_hr_tdy=sunset_fh_tdy, fcst_hr_tmr=sunset_fh_tmr
        )
    
    return {
        "city": city.name,
        "likelihood_index_tdy": likelihood_index_tdy,
        "likelihood_index_tmr": likelihood_index_tmr,
        "possible_colors_tdy": possible_colors_tdy,
        "possible_colors_tmr": possible_colors_tmr,
        "actual_afterglow_time_tdy": actual_afterglow_time_tdy,
        "actual_afterglow_time_tmr": actual_afterglow_time_tmr,
        "cloud_base_lvl_tdy": cloud_base_lvl_tdy,
        "cloud_base_lvl_tmr": cloud_base_lvl_tmr,
        "geom_condition_LCL_used_tdy": geom_condition_LCL_used_tdy,
        "geom_condition_LCL_used_tmr": geom_condition_LCL_used_tmr,
        "cloud_present_tdy": cloud_present_tdy,
        "cloud_present_tmr": cloud_present_tmr,
        "total_aod550_tdy": total_aod550[0],
        "total_aod550_tmr": total_aod550[1],
        "dust_aod550_tdy": dust_aod550[0],
        "dust_aod550_tmr": dust_aod550[1],
        "sunset_time_tdy": sunset_tdy,
    }

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

    # Write out the forecast_hours times from run initialization time in 6-hour increments
    offsets = list(range(0,61,6)) # 0 to 60 hours in 6-hour increments
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
    
    forecast_hours = {"city": city.name}

    if sunrise_time_tdy is not None:
        date_tdy = sunrise_time_tdy.date().strftime("%Y%m%d")
        date_tmr = (sunrise_time_tdy + datetime.timedelta(days=1)).date().strftime("%Y%m%d")
        date_day_next = (sunrise_time_tdy + datetime.timedelta(days=2)).date().strftime("%Y%m%d")
        forecast_hours[f"sunrise_{date_tdy}"] = pick_for_event(sunrise_time_tdy, f"sunrise_{date_tdy}")
        forecast_hours[f"sunrise_{date_tmr}"] = pick_for_event(sunrise_time_tmr, f"sunrise_{date_tmr}")
        if run_datetime < sunrise_time_tdy <= threshold_time:
            forecast_hours[f"sunrise_{date_day_next}"] = pick_for_event(sunrise_time_tmr + datetime.timedelta(days=1), f"sunrise_{date_day_next}")
        
    else:
        logging.error(f"Sunrise times missing for {city.name}")
    
    if sunset_time_tdy is not None:
        date_tdy = sunset_time_tdy.date().strftime("%Y%m%d")
        date_tmr = (sunset_time_tdy + datetime.timedelta(days=1)).date().strftime("%Y%m%d")
        date_day_next = (sunset_time_tdy + datetime.timedelta(days=2)).date().strftime("%Y%m%d")
        forecast_hours[f"sunset_{date_tdy}"] = pick_for_event(sunset_time_tdy, f"sunset_{date_tdy}")
        forecast_hours[f"sunset_{date_tmr}"] = pick_for_event(sunset_time_tmr, f"sunset_{date_tmr}")
        if run_datetime < sunset_time_tdy <= threshold_time:
            forecast_hours[f"sunset_{date_day_next}"] = pick_for_event(sunset_time_tmr + datetime.timedelta(days=1), f"sunset_{date_day_next}")
    else:
        logging.error(f"Sunset times missing for {city.name}")

    return forecast_hours

def main():
    args = parse_args()

    if args.date:
        today = datetime.datetime.strptime(args.date, "%Y%m%d").date()
    else:
        today = datetime.date.today()

    run = str(latest_forecast_hours_run_to_download().hour)
    run = run.zfill(2)
    print(run)
    today_str = today.strftime("%Y%m%d")

    # fcst_hrs_to_download = set(forecast_hours.values()) # In order to do this, we need to access the process
    # city function first. So Instead, we download all time steps between 0 and 60 hours.

    # Get these outside the loop to download global dataset

    for fcst_hour in range(0, 61, 6):
        url = f"https://data.ecmwf.int/forecast/{today_str}/{run}z/aifs-single/0p25/oper/{today_str}{run}0000-{fcst_hour}h-oper-fc.grib2"   
        download_file(url, f"{input_path}/{today_str}{run}0000-{fcst_hour}h-oper-fc.grib2")

    # Need global dataset for this
    get_cams_aod(today, run, today_str, input_path) # type: ignore

    # Load city data
    df = pd.read_csv('worldcities_info_wtimezone.csv', header=0, delimiter=',')

    results = []
    for _, row in df.iterrows():
        try:
            result = process_city(
                row['city'], 
                row['country'],
                row['lat'], 
                row['lng'], 
                row['timezone'],
                today, run, today_str, input_path, output_path,
                row['create_dashboard']
            )
            results.append(result)
        except (ValueError, KeyError, FileNotFoundError) as e:
            logging.error(f"Error processing {row['city']}: {e}", exc_info=True)
            continue # skips to next city, don't crash the script

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_path}/all_cities_summary_{today_str}.csv', index=False)
    
    return results_df


if __name__ == "__main__":
    results = main()


# Work on the case with ecloud extent too large
# The case where cloud cover increases midway along the azimuth path?