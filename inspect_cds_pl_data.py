# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from get_cds_global import get_cams_aod_lwc, get_cams_cloud_cover

today = pd.to_datetime("2026-03-29T15:00:00")  # Simulate today as selected date for testing
today_str = today.strftime("%Y%m%d")
run = "00".zfill(2) # Change this when simulating as well!

# get_cams_aod_lwc(today.date(), run, today_str, "input") # type: ignore
get_cams_cloud_cover(today.date(), run, today_str, "input") # type:

# file_path = f"input/cams_data_{today_str}{run}0000.grib"
cloud_file_path = f"input/cams_cloud_cover_data_{today_str}{run}0000.grib"

# ds = xr.open_dataset(file_path, engine='cfgrib')
ds_cloud = xr.open_dataset(cloud_file_path, engine='cfgrib')


# %%
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

def _indexpath(grib_path: str) -> str:
    # Reuse ecCodes index to speed repeated opens
    return f"{grib_path}.idx"
aifs_path = f"input/{today_str}{run}0000-0h-oper-fc.grib2"
aifs_ds = open_cloud_da(aifs_path, ['lcc', 'mcc', 'hcc'])  # type: ignore

# %%
ds_cloud = open_cloud_da(cloud_file_path, ['lcc', 'mcc', 'hcc'])  # type: ignore

# %%
from calc_afterglow_realistic_path_lwc_global import combine_cloud_layers
ds_cloud_test = combine_cloud_layers(cloud_file_path, 48)
# %%
# file_path = f"input/cams_data_{today_str}{run}0000.grib"
file_path = f"input/cams_data_{today_str}{run}0000.grib"

# ds = xr.open_dataset(file_path, engine='cfgrib')
ds = xr.open_dataset(cloud_file_path, engine='cfgrib')


# %%

variable = 'clwc' # 'clwc' or 'ciwc' or 'aerext532'
# If you have more dimensions, select or squeeze them:
latitude_sel = 30.0
hpa_sel = 900.0
da = ds[variable].sel(latitude=30.0, method="nearest")
# If 'time' or 'step' is still present, select a value:
da = da.isel(step=0)
da = da.squeeze()
# Now plot:
da.plot(x='longitude', y='isobaricInhPa')
plt.gca().invert_yaxis()  # Invert y-axis for pressure levels
plt.savefig(f"demo_figures/{variable}_vertical_profile_at_{latitude_sel}_{today_str}{run}.png")
plt.show()
# lat lon plot at surface at time 0
# If you have more dimensions, select or squeeze them:
da = ds[variable].sel(isobaricInhPa=hpa_sel, method="nearest")
# If 'time' or 'step' is still present, select a value:
da = da.isel(step=0)  # or .sel(time=your_time_value)
# Remove any size-1 dimensions:
da = da.squeeze()
# Now plot:
da.plot(x='longitude', y='latitude')
plt.savefig(f"demo_figures/{variable}_latlon_{round(hpa_sel)}hpa_{today_str}{run}.png")
plt.show()
da = ds[variable].sel(isobaricInhPa=hpa_sel, method="nearest")
# If 'time' or 'step' is still present, select a value:
step_size = 6
da = da.isel(step=step_size)  # or .sel(time=your_time_value)
# Remove any size-1 dimensions:
da = da.squeeze()
# Now plot:
da.plot(x='longitude', y='latitude')
plt.savefig(f"demo_figures/{variable}_latlon_{round(hpa_sel)}hpa_{today_str}{run}_{step_size}step.png")
plt.show()


# %%
