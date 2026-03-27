# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "input/cams_data_20260327000000.grib"
aifs_path = "input/20260327000000-6h-oper-fc.grib2"

ds = xr.open_dataset(file_path, engine='cfgrib')

# %%
aifs_ds = xr.open_dataset(aifs_path, engine='cfgrib')

# %%
np.quantile