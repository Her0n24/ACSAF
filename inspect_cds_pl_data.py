# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

today = pd.to_datetime("2026-03-27T15:00:00")  # Simulate today as selected date for testing
today_str = today.strftime("%Y%m%d")
run = "00".zfill(2) # Change this when simulating as well!

file_path = f"input/cams_data_{today_str}{run}0000.grib"
aifs_path = f"input/{today_str}{run}0000-6h-oper-fc.grib2"

ds = xr.open_dataset(file_path, engine='cfgrib')

# %%
aifs_ds = xr.open_dataset(aifs_path, engine='cfgrib')

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
