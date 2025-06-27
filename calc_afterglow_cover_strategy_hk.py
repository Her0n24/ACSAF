"""
Project ACSAF: Aerosols & Cloud geometry based Sunset/Sunrise cloud Afterglow Forecaster

This script uses the ECMWF AIFS cloud cover data and CAMS AOD550 data to visualize the cloud cover maps and calculate various parameters related to afterglow.

Data availability (HH:MM)
CAMS Global analyses and forecasts:

00 UTC forecast data availability guaranteed by 10:00 UTC

12 UTC forecast data availability guaranteed by 22:00 UTC

For UTC +8, Hong Kong sunset time.
Around 18 to 19 depending on the time of the year, which is around 10 to 11 UTC.
So for today's sunset, we need 12Z run of the previous day, available by 22Z. 
Hence, optimally we need +18h or +24h. 


"""
import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from astral.sun import sun, elevation, azimuth
from astral import LocationInfo
import pytz
import cfgrib
from calc_cloudbase import specific_to_relative_humidity, calc_cloud_base
from get_aifs import download_file
from get_cds import get_cams_aod
from geopy import Nominatim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
font_path = '/home/tsing/.local/share/fonts/SourceHanSerifTC-Bold.otf'  # Update this path as needed
output_path = '/home/tsing/Documents/dev/Afterglow/output'
input_path = '/home/tsing/Documents/dev/Afterglow/input'

# Register the font
font_prop = font_manager.FontProperties(fname=font_path)
# Set as default font for all text
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False


run = "12"
run = run.zfill(2)
today = datetime.date.today() + datetime.timedelta(days=1) # Remember to uncomment after fixing the script
today_str = today.strftime("%Y%m%d")
yesterday = today - datetime.timedelta(days=1)
yesterday_str = yesterday.strftime("%Y%m%d")

def get_or_save_latlon(location_name, filename="location_latlon.txt"):
    """
    Get lat/lon for a location, saving to a file for future use.
    If the file exists, read lat/lon from it.
    Otherwise, geocode and save to file.
    """
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            name, lat, lon = line.split(",", 2)
            if name == location_name:
                return float(lat), float(lon)
    # If not found or name mismatch, geocode and save
    geolocator = Nominatim(user_agent="acsaf_afterglow_forecaster")
    location = geolocator.geocode(location_name)
    lat, lon = location.latitude, location.longitude # type: ignore
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{location_name},{lat},{lon}\n")
    return lat, lon

# Usage:
lat, lon = get_or_save_latlon("HongKong")
lat, lon = round(lat, 1), round(lon, 1)  # Round to 1 decimal place

# print(location): print the name of the location

city = LocationInfo("HongKong", "HK", "Asia/Hong_Kong", lat, lon)  # Hong Kong
s_tdy = sun(city.observer, date=today)
sunset = s_tdy['sunset']
s_tmr = sun(city.observer, date=today + datetime.timedelta(days=1))
sunrise = s_tmr['sunrise']
sunset_tmr = s_tmr['sunset']

print(city.name)

url = f"https://data.ecmwf.int/forecasts/{yesterday_str}/{run}z/aifs-single/0p25/oper/{yesterday_str}{run}0000-24h-oper-fc.grib2"   
download_file(url, f"{input_path}/{yesterday_str}{run}0000-24h-oper-fc.grib2")

url = f"https://data.ecmwf.int/forecasts/{yesterday_str}/{run}z/aifs-single/0p25/oper/{yesterday_str}{run}0000-48h-oper-fc.grib2"   
download_file(url, f"{input_path}/{yesterday_str}{run}0000-48h-oper-fc.grib2")

get_cams_aod(yesterday, run, city.name, yesterday_str, input_path)

def max_solar_elevation(city, date):
    tz = pytz.timezone(city.timezone)
    observer = city.observer

    # Sample every 5 minutes across the day
    times = [datetime.datetime.combine(date, datetime.datetime.min.time()) + datetime.timedelta(minutes=5*i) for i in range(288)]
    times = [tz.localize(t) for t in times]

    max_angle = max(elevation(observer, t) for t in times)
    return max_angle

max_elev = max_solar_elevation(city, datetime.date.today())
print(f"Maximum solar elevation in {city.name}: {max_elev:.2f}°")

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

sunset_azimuth = get__sunset_azimuth(city, today)
sunset_azimuth_42 = get__sunset_azimuth(city, today + datetime.timedelta(days=1))
print(f"Sunset azimuth angle in {city.name}: {sunset_azimuth:.2f}°")
print(f"sunset time in {city.name}: {sunset}")

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

# Define a square box enclosing the region of interest
lat_min, lat_max = lat - 3, lat + 3
lon_min, lon_max = lon - 5, lon + 1

ds_18 = xr.open_dataset(f'{input_path}/{yesterday_str}{run}0000-24h-oper-fc.grib2', engine = 'cfgrib')
ds_42 = xr.open_dataset(f'{input_path}/{yesterday_str}{run}0000-48h-oper-fc.grib2', engine = 'cfgrib')

# t2m at 2m
# Please fix the cfgrib datasetbuilderror: key present and new value is different: key='heightAboveGround' value=Variable(dimensions=(), data=np.float64(10.0)) new_value=Variable(dimensions=(), data=np.float64(100.0))
ds_18_2m = cfgrib.open_dataset(f'{input_path}/{yesterday_str}{run}0000-24h-oper-fc.grib2', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
ds_42_2m = cfgrib.open_dataset(f'{input_path}/{yesterday_str}{run}0000-48h-oper-fc.grib2', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})

def extract_variable(ds, var_name, lat_min, lat_max, lon_min, lon_max, verbose=False):
    var = getattr(ds, var_name)
    var = var.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).squeeze()
    if verbose:
        print(f"{var_name}:")
        print(var.values)
    return var

ds_18_lcc = extract_variable(ds_18, "lcc", lat_min, lat_max, lon_min, lon_max)
ds_18_mcc = extract_variable(ds_18, "mcc", lat_min, lat_max, lon_min, lon_max)
ds_18_hcc = extract_variable(ds_18, "hcc", lat_min, lat_max, lon_min, lon_max)
ds_18_tcc = extract_variable(ds_18, "tcc", lat_min, lat_max, lon_min, lon_max)

ds_42_tcc = extract_variable(ds_42, "tcc", lat_min, lat_max, lon_min, lon_max)
ds_42_lcc = extract_variable(ds_42, "lcc", lat_min, lat_max, lon_min, lon_max)
ds_42_mcc = extract_variable(ds_42, "mcc", lat_min, lat_max, lon_min, lon_max)
ds_42_hcc = extract_variable(ds_42, "hcc", lat_min, lat_max, lon_min, lon_max)

cloud_vars_18 = {
    "tcc": ds_18_tcc,
    "lcc": ds_18_lcc,
    "mcc": ds_18_mcc,
    "hcc": ds_18_hcc
}

cloud_vars_42 = {
    "tcc": ds_42_tcc,
    "lcc": ds_42_lcc,
    "mcc": ds_42_mcc,
    "hcc": ds_42_hcc
}


def plot_cloud_cover_map(data_dict, city, lon, lat, title_prefix, fcst_hr, sunset_azimuth, save_path= output_path, cmap='gray'):
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

    plt.savefig(f'{save_path}' + f"/{yesterday_str}{run}0000-{fcst_hr}h-AIFS_cloud_cover_{city.name}.png")
    plt.close()
    
plot_cloud_cover_map(cloud_vars_18, city, lon, lat,
                     f'{yesterday_str} {run}z +24h EC AIFS cloud cover (today sunset)',
                     '24', sunset_azimuth, save_path=output_path, cmap='gray')

plot_cloud_cover_map(cloud_vars_42, city, lon, lat,
                     f'{yesterday_str} {run}z +48h EC AIFS cloud cover (tomorrow sunset)',
                     '48', sunset_azimuth, save_path= output_path, cmap='gray')

RH_18, p_18 = specific_to_relative_humidity(ds_18.q, ds_18.t, ds_18.isobaricInhPa, lat, lon)
cloud_base_lvl_18, z_lcl_18, RH_cb_18 = calc_cloud_base(ds_18_2m["t2m"], ds_18_2m["d2m"], ds_18.t, RH_18, ds_18.isobaricInhPa, lat, lon)

RH_42, p_42 = specific_to_relative_humidity(ds_42.q, ds_42.t, ds_42.isobaricInhPa, lat, lon)
cloud_base_lvl_42, z_lcl_42, RH_cb_42 = calc_cloud_base(ds_42_2m["t2m"], ds_42_2m["d2m"], ds_42.t, RH_42, ds_42.isobaricInhPa, lat, lon)

print('18')
print(cloud_base_lvl_18, z_lcl_18, RH_cb_18)
print('42')
print(cloud_base_lvl_42, z_lcl_42, RH_cb_42)

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

def plot_cloud_cover_along_azimuth(cloud_cover_data, azimuth, distance_km, fcst_hr, threshold, cloud_lvl_used, save_path=output_path):
    """
    Plot cloud cover data along the azimuth line.
    
    Parameters:
    - cloud_cover_data: Array containing cloud cover data along the azimuth line
    - azimuth: Azimuth angle in degrees
    - distance_km: Distance along the azimuth line (in km)
    - save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the cloud cover data
    ax.plot(np.linspace(0, distance_km, len(cloud_cover_data)), cloud_cover_data, label=f"Cloud Cover ({cloud_lvl_used})")
    
    # Check for the first distance where cloud cover falls below the threshold
    below_threshold_index = np.argmax(cloud_cover_data <= threshold)
    
    if below_threshold_index > 0:  # Ensure that the threshold is met somewhere in the data
        distance_below_threshold = np.linspace(0, distance_km, len(cloud_cover_data))[below_threshold_index]
        avg_first_three = np.mean(cloud_cover_data[:3])
        avg_path = np.mean(cloud_cover_data[4:])
        print(f"Local cloud cover is {avg_first_three}%. Average path cloud cover is {avg_path}%.")
        print(f"The cloud cover falls below {threshold}% at {distance_below_threshold} km.")
    else:
        print(f"Cloud cover does not fall below {threshold}%.")
        distance_below_threshold = np.nan
        avg_first_three = np.mean(cloud_cover_data[:3])
        avg_path = np.mean(cloud_cover_data[4:])
        if avg_first_three > 10 and avg_first_three < threshold and avg_path < threshold: # We still think there are cloud if local above 10% total cloud cover
            distance_below_threshold = 300
            print(f"Local cloud cover is {avg_first_three}%. Average path cloud cover is {avg_path}%. Meet Criteria even threshold requirement not met.")
            print(f"There is cloud cover above, we assume disance below thres is {distance_below_threshold} km.")
            
    # Plot the threshold line
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold}% threshold')
    
    if avg_first_three > 10 and avg_first_three < threshold and avg_path < threshold:
        ax.axhline(y=avg_first_three, color='orange', linestyle='--', label=f'Local cloud cover {avg_first_three}%')
        
    if ~np.isnan(distance_below_threshold):
        ax.axvline(x=distance_below_threshold, color='r', linestyle='--', label=f'{threshold}% threshold')
    ax.legend()
    
    ax.set_ylim(0,100)
    ax.set_xlabel('Distance along Azimuth (km)')
    ax.set_ylabel('Cloud Cover (%)')
    ax.set_title(f'{yesterday_str} {run}z +{fcst_hr}h EC AIFS cloud cover Along Azimuth path ')
    #set subtitle
    ax.text(0.5, 1.02, f"Azimuth {azimuth}°", fontsize=10)
    ax.legend()
    
    plt.savefig(save_path + f"/{yesterday_str}{run}0000_{fcst_hr}h_AIFS_cloud_cover_azimuth_{city.name}.png")
    plt.close()
    return distance_below_threshold, avg_first_three, avg_path
    
# cloud_cover_data = extract_cloud_cover_along_azimuth(ds_18_tcc, lon, lat, sunset_azimuth, 500, num_points=20)
# cloud_cover_data_42 = extract_cloud_cover_along_azimuth(ds_42_tcc, lon, lat, sunset_azimuth_42, 500, num_points=20)

# print(cloud_cover_data)
# print(cloud_cover_data_42)

cloud_cover_data_18_all = {
    key: extract_cloud_cover_along_azimuth(data, lon, lat, sunset_azimuth, 500, num_points=20)
    for key, data in cloud_vars_18.items()
}

# Apply the function to all datasets in cloud_vars_42
cloud_cover_data_42_all = {
    key: extract_cloud_cover_along_azimuth(data, lon, lat, sunset_azimuth_42, 500, num_points=20)
    for key, data in cloud_vars_42.items()
}

# # Print the results for verification
# print("Cloud cover data for +18h forecast:")
# print(cloud_cover_data_18_all)


def get_cloud_extent(data_dict, lon, lat, azimuth, cloud_base_lvl: float, fcst_hr, distance_km = 500, num_points=20, threshold=60.0):
    priority_order = ["lcc", "mcc", "hcc"]
    cloud_lvl_used = None
    cloud_present = True
    
    for key in priority_order:
        if key == 'lcc' or key == 'mcc':
            #Extract cloud_cover data along the azimuth
            cloud_cover_data = extract_cloud_cover_along_azimuth(
                data_dict[key], lon, lat, azimuth, distance_km, num_points
            )
            # Calculate the average of the first 3 indices
            avg_first_three = np.nanmean(cloud_cover_data[:3])
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
            avg_path = np.nanmean(cloud_cover_data[4:])
            if avg_first_three > threshold:
                cloud_lvl_used = key 
                data = data_dict[key]
                avg_first_three_met = True
                hcc_condition = False
            if avg_first_three < avg_path:
                avg_first_three_met = True
                print(f"Average first three cloud cover {avg_first_three}% is less than average path cloud cover {avg_path}%.")
                # IF avg first three is above 10% and below 90%
                if avg_first_three > 10.00 and avg_first_three < 90.00 and avg_path < 70.00:
                    hcc_condition = True
                    cloud_lvl_used = key 
                    data = data_dict[key]
                else:
                    hcc_condition = False
                    avg_first_three_met = False

    # Check if data is empty
    if avg_first_three_met is False:
        cloud_lvl_used = 'tcc' 
        data = data_dict['tcc']
        print(f"Cloud base level {cloud_base_lvl} is invalid. There is no cloud cover. Afterglow not probable.")
        print("Trying to use tcc for cloud cover data")
        data = data_dict['tcc']
        cloud_lvl_used = 'tcc'
        try:
            cloud_cover_data = extract_cloud_cover_along_azimuth(data, lon, lat, azimuth, distance_km, num_points)
            distance_below_threshold, avg_first_three, avg_path  = plot_cloud_cover_along_azimuth(cloud_cover_data, azimuth, distance_km, fcst_hr, threshold, cloud_lvl_used, save_path= output_path)
        except:
            print(f"Cloud cover data is not available for forecast hour {fcst_hr}.")
            cloud_present = False
    # # Check if data is associated with a value or a NaN
    # if data is None:
    #     print(f"Cloud cover NaN for forecast hour {fcst_hr}. There is no stratiform cloud cover. Afterglow not probable.")
    #     cloud_present = False
    #     return cloud_present
    elif hcc_condition is False:
        cloud_cover_data = extract_cloud_cover_along_azimuth(data, lon, lat, azimuth, distance_km, num_points)
        distance_below_threshold, avg_first_three, avg_path = plot_cloud_cover_along_azimuth(cloud_cover_data, azimuth, distance_km, fcst_hr, threshold, cloud_lvl_used, save_path= output_path)
    elif hcc_condition is True:
        print('hcc condition is True. We will assume a distance below threshold of 300 km.')
        cloud_cover_data = extract_cloud_cover_along_azimuth(data, lon, lat, azimuth, distance_km, num_points)
        distance_below_threshold = 300
        avg_first_three = np.nanmean(cloud_cover_data[:3])
        avg_path = np.nanmean(cloud_cover_data[4:])
    distance_below_threshold = distance_below_threshold * 1000 # convert to meters

    if avg_first_three < 10.0:
        cloud_present = False

    return distance_below_threshold, key, avg_first_three, avg_path, cloud_present

distance_below_threshold_18, key_18, avg_first_three_18, avg_path_18, cloud_present_18 = get_cloud_extent(cloud_vars_18, lon, lat, sunset_azimuth, cloud_base_lvl_18, '24')
distance_below_threshold_42, key_42, avg_first_three_42, avg_path_42, cloud_present_42 = get_cloud_extent(cloud_vars_42, lon, lat, sunset_azimuth, cloud_base_lvl_42, '48')

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
        print(f"lf_ma: {round(lf_ma)} m")
        geom_condition_LCL_used = False
    except ValueError as e:
        print(f"Error: {e}")
        print(f"lf_ma is {lf_ma}")
        print('We will assume lf_ma using LCL')
        lf_ma = 2*np.sqrt(2*R*LCL)
        print(f"lf_ma: {round(lf_ma)} m")
        geom_condition_LCL_used = True
        geom_condition = False
    # Compare with cloud_extent
    if lf_ma > cloud_extent:
        geom_condition = True
    else:
        geom_condition = False
    print("cloud geometry condition:", geom_condition)
    return geom_condition, geom_condition_LCL_used, lf_ma

geom_cond_18, geom_condition_LCL_used_18, lf_ma_18 = geom_condition(cloud_base_lvl_18, distance_below_threshold_18, z_lcl_18)
geom_cond_42, geom_condition_LCL_used_42, lf_ma_42 = geom_condition(cloud_base_lvl_42, distance_below_threshold_42, z_lcl_42)

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
        print("Cloud cover and elevation estimated using tcc")
        theta = np.arctan((cloud_base_lvl-(lf_ma**2/(2*R)))/lf_ma)
        print(f"Elevation angle: {np.rad2deg(theta)}°")
        return theta
    elif np.isnan(cloud_base_lvl):
        print("Cloud cover and elevation estimated using tcc and LCL")
        theta = np.arctan((lcl-((distance_below_threshold**2)/(2*R)))/distance_below_threshold)
        print(f"Elevation angle: {np.rad2deg(theta)}°")
        return theta
    else:
        theta = np.arctan((cloud_base_lvl-(distance_below_threshold**2/(2*R)))/distance_below_threshold)
        print(f"Elevation angle: {np.rad2deg(theta)}°")
        return theta

theta_18 = get_elevation_afterglow(cloud_base_lvl_18, distance_below_threshold_18, lf_ma_18, z_lcl_18)
theta_42 = get_elevation_afterglow(cloud_base_lvl_42, distance_below_threshold_42, lf_ma_42, z_lcl_42)

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
        
        actual_afterglow_time = actual_afterglow_time + ((cloud_base_lvl/np.tan(np.deg2rad(15)))/(21*1000/60)) # Accounting for the clouds in visual contact assuming 15 deg elevation (increased from 5 deg)
        
        print(f"Total Afterglow time: {total_afterglow_time} seconds")
        print(f"Overhead Afterglow time: {overhead_afterglow_time} seconds")
        print(f"Actual Afterglow time: {actual_afterglow_time} seconds")
    else:
        print(f"Sun ray extent lf_max is less than cloud extent. No afterglow is possible.")
        actual_afterglow_time = 0
    return actual_afterglow_time

actual_afterglow_time_18 = get_afterglow_time(lat, today, distance_below_threshold_18, lf_ma_18, cloud_base_lvl_18, z_lcl_18)
actual_afterglow_time_42 = get_afterglow_time(lat, today, distance_below_threshold_42, lf_ma_42, cloud_base_lvl_42, z_lcl_42)

# One method is to use Equivalent cloud heigh  = cloud base level - equivalent surface height as highlighted in the paper
# But uncertainty is high and the exact mechanism is not clear
# Here we use a simplier and quicker method, but yet to be verified
# The AOD value directly controls the value of the afterglow liklihood index in the weighted equation

# Incorporate AOD
from calc_aod import calc_aod
dust_aod550, total_aod550, dust_aod550_ratio = calc_aod(run, yesterday_str, city.name, input_path) #Array of shape (2,) first is 18h , second is 42h

def signed_power(x, p):
    return np.sign(x) * (abs(x) ** p)

# Weighted likelihod index
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
    if np.isnan(cloud_base_lvl) or cloud_base_lvl <= 0:
        cloud_base_lvl = lcl_lvl
        print("Used LCL level for computation")
        max_lvl = 6000.0
        norm_lvl = min(cloud_base_lvl, max_lvl) / max_lvl
        cloud_base_score = (norm_lvl)
        print(f"cloud base score using LCL: {cloud_base_score}")
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

        print(f"x_score: {x}, y_score: {y}")
        
        cloud_cover_score = 0.4 * x - 0.6 * y
    
    print(f"Cloud cover score: {cloud_cover_score}")

    # import pdb; pdb.set_trace()  # Debugging point to inspect variables
    
    if geom_condition == True:
        # Constants
        geom_condition_weight = 0.1
        aod_weight = 0.15
        dust_aod_ratio_weight = 0.05
        cloud_cover_weight = 0.4
        cloud_base_lvl_weight = 0.2
        theta_weight = 0.1    
    else:
        # Constants
        geom_condition_weight = 0.5
        aod_weight = 0.05
        dust_aod_ratio_weight = 0.05
        cloud_cover_weight = 0.2
        cloud_base_lvl_weight = 0.1
        theta_weight = 0.1
        
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
    print(f"geom_flag: {geom_flag}, contribution: {signed_power(geom_flag, 1) * geom_condition_weight}")
    print(f"aod_score: {aod_score}, contribution: {signed_power(aod_score, 2) * aod_weight}")
    print(f"dust_ratio_score: {dust_ratio_score}, contribution: {signed_power(dust_ratio_score, 3) * dust_aod_ratio_weight}")
    print(f"cloud_cover_score: {cloud_cover_score}, contribution: {signed_power(cloud_cover_score, 1) * cloud_cover_weight}")
    print(f"cloud_base_score: {cloud_base_score}, contribution: {signed_power(cloud_base_score, 2) * cloud_base_lvl_weight}")
    print(f"theta_score: {theta_score}, contribution: {signed_power(theta_score, 2) * theta_weight}")

    likelihood_index = np.clip(likelihood_index, 0, 1)  # Ensure it's between 0 and 1
    
    # Scale to 0-100 and round to whole number
    likelihood_index = round(likelihood_index * 100)
    return likelihood_index


likelihood_index_18 = weighted_likelihood_index(geom_cond_18, total_aod550[0], dust_aod550_ratio[0], cloud_base_lvl_18, z_lcl_18, theta_18, avg_first_three_18, avg_path_18)
likelihood_index_42 = weighted_likelihood_index(geom_cond_42, total_aod550[1], dust_aod550_ratio[1], cloud_base_lvl_42, z_lcl_42, theta_42, avg_first_three_42, avg_path_42)
print(likelihood_index_18)
print(likelihood_index_42)

def possible_colours(cloud_base_lvl, lcl_lvl, total_aod_550, key):
    """
    Determine the possible colours for the afterglow based on cloud base level, inferred cloud cover and AOD_550.
    Cloud cover is inferred through RH of the cloud base level. 
    
    Parameters:
    """
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
            color = ('orange-red', 'crimson', 'dark-red',)
        else:
            color = ('dirty-orange',)
    elif cloud_base_lvl > 2000.0 and cloud_base_lvl <= 6000.0:
        if key == 'lcc':
            color = ('orange-red', 'dark-red',)
        if key == 'mcc':
            color = ('orange-yellow', 'orange-red', 'dark-red',) 
        if key == 'hcc':
            color = ('orange-yellow', 'golden-yellow', 'magenta',)
    elif cloud_base_lvl > 6000.0:
        color = ('golden-yellow', 'crimson', 'magenta',)
    
    return color

possible_colors_18 = possible_colours(cloud_base_lvl_18, z_lcl_18, total_aod550[0], key_18)
possible_colors_42 = possible_colours(cloud_base_lvl_42, z_lcl_42, total_aod550[1], key_42)
print(f"Possible colors for afterglow 18: {possible_colors_18}")
print(f"Possible colors for afterglow 42: {possible_colors_42}")

# Color fill based on index using the 'plasma' colormap
def color_fill(index):
    # Normalize the index to a value between 0 and 1 for the colormap
    norm = plt.Normalize(vmin=0, vmax=100) # type: ignore
    cmap = plt.get_cmap('magma')  
    return cmap(norm(index))

print(cloud_base_lvl_18)
print(cloud_base_lvl_42)

def create_dashboard(index_today, index_tomorrow, city, latitude, longitude,
                     azimuth, afterglow_length_18, afterglow_length_42, possible_colors_18,
                     possible_colors_42, cloud_base_lvl_18, cloud_base_lvl_42, 
                     z_lcl_18, z_lcl_42, cloud_present_18, cloud_present_42,):
    if np.isnan(cloud_base_lvl_18):
        cloud_base_lvl_18 = z_lcl_18
    if np.isnan(cloud_base_lvl_42):
        cloud_base_lvl_42 = z_lcl_42
    
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
    
    # Remove fontproperties from suptitle, use only for English
    plt.suptitle("ACSAF: Aerosol and Cloud geometry based Sunset cloud Afterglow Forecaster", 
                fontsize=16, weight='bold', color='white')

    # Add the Chinese part as a separate text, using font_prop_bold
    plt.gcf().text(0.5, 0.91, "火燒雲觀賞指數預報", 
                fontsize=16, color='white', fontproperties=font_prop, ha='center', va='center')

    # Box settings for today and tomorrow
    # Larger, square boxes with index numbers only in bold
    ax[0].text(0.1, 0.5, f"{index_today}", fontsize=40, fontweight='bold', ha='center', va='center', 
            color='white', bbox=dict(facecolor=color_fill(index_today), edgecolor='black', boxstyle='round,pad=2'))

    ax[0].text(1.0, 0.5, f"{index_tomorrow}", fontsize=40, fontweight='bold', ha='right', va='center',
            color='white', bbox=dict(facecolor=color_fill(index_tomorrow), edgecolor='black', boxstyle='round,pad=2'))

    # Add text outside the box for "Today" and "Tomorrow"
    ax[0].text(0.1, 0.1, "Today", fontsize=18, ha='center', va='center', color='white', fontweight='bold')
    ax[0].text(0.9, 0.1, "Tomorrow", fontsize=18, ha='center', va='center', color='white', fontweight='bold')

    if cloud_present_18 is False:
        ax[0].text(0.1, 0.2, "No cloud cover", fontsize=15, ha='center', va='center', color='white', fontweight='bold')
    if cloud_present_42 is False:
        ax[0].text(0.9, 0.2, "No cloud cover", fontsize=15, ha='center', va='center', color='white', fontweight='bold')

    # Title for the left subplot
    ax[0].axis('off')  # Turn off axis for this subplot


    sunset_tz = pytz.timezone("HongKong")  # Desired timezone for Reading (UK)

    # Convert the sunset time from UTC to local time
    sunset_local = sunset.astimezone(sunset_tz)
    sunset_local_42 = sunset_tmr.astimezone(sunset_tz)
    
    # Info text on the right

    info_text = (
        f"Today 今日: {today.strftime('%Y-%m-%d')}\n"
        f"Location 地點: {city.name}\n"
        f"Sunset Time 日落時間(HKT): {sunset_local.strftime('%H:%M:%S')} \n"
        f"Sunset Azimuth 太陽方位角: {round(azimuth)}°\n"
        f"Length 火燒雲時長: {round(afterglow_length_18)} s\n"
        f"Cloud Base 雲底: {int(round(cloud_base_lvl_18, -2))} m\n"
        f"Aerosol OD 氣溶膠光學厚度(550nm) : {total_aod550[0]:.2f} \n"
        f"Colors 顏色: {', '.join(possible_colors_18) }\n"
    )
    
    ax[1].text(0.7, 0.6, info_text, fontsize=15, ha='center', va='center', color='black', 
           bbox=dict(facecolor='lightgray', alpha=0.8), fontproperties=font_prop)
    
    # Info text on the right
    info_text = (
        f"Tomorrow 明日: {(today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')}\n"
        f"Length 火燒雲時長: {round(afterglow_length_42)} s\n"
        f"Cloud Base 雲底: {int(round(cloud_base_lvl_42, -2))} m\n"
        f"Aerosol OD 氣溶膠光學厚度(550nm) : {total_aod550[1]:.2f} \n"
        f"Colors 顏色: {', '.join(possible_colors_42)}\n"
    )
    ax[1].text(0.7, 0.15, info_text, fontsize=15, ha='center', va='center', color='black', 
           bbox=dict(facecolor='lightgray', alpha=0.8), fontproperties=font_prop)

    ax[1].axis('off')  # Turn off axis for this subplot
        
    # Chinese part
    fig.text(
        0.01, 0.055,
        "指數範圍為0-100，數值越高越好。數值高代表有條件形成大範圍色彩鮮豔且持續較長的火燒雲或餘輝。\n"
        "基於昨日12z 歐洲中期預報中心人工智慧氣象預報系統和哥白尼大氣監測服務的預報。僅對成層狀分佈的雲有效。雲況詳情請參閱補充圖表。",
        ha='left', va='bottom', color='white', fontsize=8, fontproperties=font_prop
    )
    # English part (no fontproperties, uses default/bold)
    fig.text(
        0.01, 0.015,
        "Index ranges 0-100. Higher indicates more favourable cloud afterglow conditions for occurence and intense colors.\n"
        "Based on daily 00z ECMWF AIFS and Copernicus Atmosphere Monitoring Service forecasts. Valid only for stratiform cloud layer. See supplementary figures for details.",
        ha='left', va='bottom', color='white', fontsize=8
    )
    fig.text(0.89, 0.01, f"Her0n24. V2025.6.27", color='white', fontsize=8)
    
    plt.savefig(f'{output_path}/{yesterday_str}{run}0000_afterglow_dashboard_{city.name}_hk.png', dpi=400)
    print(f"Dashboard saved to {output_path}/{yesterday_str}{run}0000_afterglow_dashboard_{city.name}_hk.png")

create_dashboard(
    likelihood_index_18, likelihood_index_42, city, lat, lon, sunset_azimuth, actual_afterglow_time_18, actual_afterglow_time_42, possible_colors_18, possible_colors_42, cloud_base_lvl_18, cloud_base_lvl_42,
    z_lcl_18, z_lcl_42, cloud_present_18, cloud_present_42 
)

# Work on the case with ecloud extent too large
# The case where cloud cover increases midway along the azimuth path?