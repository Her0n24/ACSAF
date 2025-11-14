import pandas as pd
from timezonefinder import TimezoneFinder
from datetime import datetime
import os
import numpy

df = pd.read_csv('/Users/hng/Documents/dev/Afterglow/worldcities_info_wtimezone.csv', header=0, delimiter=',')
tf = TimezoneFinder()
print(df)
print(df.columns)

# Check if the lat and lon are in the correct format from -90 to 90 and -180 to 180 respectively

# Check if the lat and lon are in the correct format from -90 to 90 and -180 to 180 respectively
# invalid_coords = df[(df['lat'] < -90) | (df['lat'] > 90) | (df['lng'] < -180) | (df['lng'] > 180)]

# if not invalid_coords.empty:
#     print(f"Warning: Found {len(invalid_coords)} rows with invalid coordinates:")
#     print(invalid_coords)
# else:
#     print("All coordinates are valid.")

# df['timezone'] = df.apply(lambda row: tf.timezone_at(lng=row['lng'], lat=row['lat']), axis=1)
df['create_dashboard'] = False

df.to_csv('/Users/hng/Documents/dev/Afterglow/worldcities_info_wtimezone.csv', index=False)