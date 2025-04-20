import xarray as xr
import numpy as np
import pytz

def specific_to_relative_humidity(q,T,p, lat, lon):
    """
    Approximate relative humidity with specific humidity and temperature.

    Parameters:
    - q: Specific humidity (kg/kg)
    - T: Temperature (K)
    - p: Pressure (hPa)

    Returns:
    - RH: Relative humidity (%)
    """
    # Point base location
    q = q.sel(latitude=lat, longitude=lon, method='nearest').values
    T = T.sel(latitude=lat, longitude=lon, method='nearest').values
    p = p.values
    # Convert temperature from kelvin to celsius
    T_C = T - 273.15
    
    # Saturation vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))

    # Actual vapor pressure (hPa)
    e = (q * p) / (0.622 + 0.378 * q)

    # Relative humidity (%)
    RH = 100 * (e / es)
    return np.clip(RH, 0, 100), p
    

def calc_cloud_base(T2m, Td2m, T, RH, p, lat, lon):
    """
    Calculate most likely cloud base height using the vertical profile of relative humidity derived.
    Also calculate the theorhetical lifted condensation level for convective cloud base using surface temperature and lapse rates.
    
    Parameters:
    - T2m: 2m temperature (K)
    - Td2m: 2m dew point temperature (K)
    - T: Temperature (K)
    - RH: Relative humidity (%)
    - p: Pressure (hPa)
    
    Returns:
    - cloud_base_pressure: Cloud base pressure_lvl (m)
    - z_lcl: Lifted condensation level (m)
    - RH_cb : Relative humidity at cloud base (%)
    """
    T2m = T2m.sel(latitude=lat, longitude=lon, method='nearest').values
    Td2m = Td2m.sel(latitude=lat, longitude=lon, method='nearest').values
    T = T.sel(latitude=lat, longitude=lon, method='nearest').values
    p = p.values
    
    # Constants
    gamma_d = 9.8 / 1000  # Dry adiabatic lapse rate (K/m)

    # Bolton's equation for T_lcl
    T_lcl = 1 / (1 / (Td2m - 56) + np.log(T2m / Td2m) / 800) + 56

    # Approximate LCL height
    z_lcl = (T2m - T_lcl) / gamma_d
    
    # Find the cloud base, the lowest pressure level with RH > 85% for temperature above 0 and with RH > 80% for temperature below 0.
    T_C = T - 273.15
    
    # Define mask based on temperature-dependent RH threshold to account for ice clouds
    condition = ((T_C > 0) & (RH >= 85)) | ((T_C <= 0) & (RH >= 75))
    
    print("RH")
    print(RH)
    
    # Check if any True values exist in the condition
    if np.any(condition):
        # Find the first index where the condition is True
        first_idx = np.argmax(condition, axis=0)
        cloud_base_pressure = p[first_idx]
        RH_cb = RH[first_idx]
        T_cb = T[first_idx] # Kelvin

        # Convert cloud base height from hPa to approximately m assuming standard atmosphere
        # Constants
        R = 287.05  # Gas constant for dry air (J/(kg·K))
        g = 9.81    # Acceleration due to gravity (m/s²)
        p0 = 1013.25  # Reference pressure (hPa)
        
        # Apply the hypsometric equation
        cloud_base_height = (R * T_cb / g) * np.log(p0 / cloud_base_pressure)
    else:
        # If no True values, set cloud base pressure to NaN
        cloud_base_height = np.nan
        RH_cb = np.nan
    
    return cloud_base_height, z_lcl, RH_cb