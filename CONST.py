"""Centralized constants for Afterglow project."""
import numpy as np

# Physical / geometric
MAX_CLOUD_HEIGHT = 9000  # meters
VIEW_ELEVATION_ANGLE = 5  # degrees
R_EARTH_M = 6.371e6  # Earth radius in meters
R_DRY = 287.05  # Specific gas constant for dry air (J/(kg*K))
T0 = 288.15      # Sea level standard temperature (K)
P0 = 101325.0    # Sea level standard pressure (Pa)
L = 0.0065       # Standard lapse rate (K/m)
R_CLOUDDROP = 10e-6  # Effective radius of cloud droplets (m)
R_ICECRYSTAL = 50e-6  # Effective radius of ice crystals (m)
RHO_WATER = 1000  # kg/m^3 Density of liquid water
RHO_ICE = 917    # kg/m^3 Density of ice

# Ray sampling / temporal
TIMESTEP_SECONDS = 60
TIMESTEP_ARRAY = np.arange(0, 1081, TIMESTEP_SECONDS)
ALPHA_COEFF = -5.14e-5

# Scoring
RAY_NORM_CONSANT = 0.7  # Max I_ray value for normalization in scoring
RAY_NORM_CONSANT_LWC = 0.05

## Cloud layer definitions
# Cloud layer height bins (meters)
LCC_HMIN, LCC_HMAX = 0, 1999
MCC_HMIN, MCC_HMAX = 2000, 5999
HCC_HMIN, HCC_HMAX = 6000, 9000

# Cloud cover thresholds (percentage)
MIN_CLOUD_COVER_THRESHOLD = 10  # % Minimum cloud cover fraction to consider a layer as present
MAX_LCC_THRESHOLD = 50  # % Max LCC cover to consider it "low" for scoring
MAX_MCC_THRESHOLD = 60  # % Max MCC cover to consider it "low" for scoring

# Representative layer heights (meters)
LCC_HEIGHT = 1000
MCC_HEIGHT = 4000
HCC_HEIGHT = 7500

# Resampling and optical depths
HORIZONTAL_GRID_RESOLUTION = 0.4 # degrees
HORIZONTAL_GRID_RESOLUTION_KM_APPROX = HORIZONTAL_GRID_RESOLUTION * 111 # Approximate conversion from degrees to km
DESIRED_NUM_POINTS = 100 ## NOTE THAT THIS NUMBER DEPENDS ON THE HORIZONTAL RESOLUTION OF THE DATA #DECREASED FROM 25 TO 
TAU_EFF_MAP = {
    'lcc': 2.0,
    'mcc': 1.0,
    'hcc': 0.3,
}

# Aerosol layer
H_AERO_KM_DEFAULT = 3.0

# Defaults for plotting / selection
DEFAULT_AZIMUTH_LINE_DISTANCE_KM = 700
ASSUMED_DISTANCE_BELOW_THRESHOLD_KM = 250
DEBUG_TS = 14
I_RAY_THRESHOLD_DEFAULT = 0.05
