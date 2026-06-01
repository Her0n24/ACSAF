# ACSAF: Aerosols & Cloud geometry based Sunset/Sunrise cloud Afterglow Forecaster

The ASCAF project attempts to provide a short-term (today and tomorrow) forecast for the occurrence of a vibrant sunset cloud afterglow. The computed index ranges from 0-100, with 0 indicating a dull, short-lived and less appreciable cloud afterglow is probable while 100 indicating a vilvid, long lasting and very visible cloud afterglow is probable.

The index is based on the ECMWF IFS Cloud and Ice Water Content and CAMS atmosphere Total Aerosol Optical Depth forecast data for the next two days with 3 hours temporal resolution, 0.4 degrees horizontal resolution and 13 pressure levels. Various parameters are chosen when computing the index. While the computation is physically based on Beer-Lambert's law and as physical as possible, the final value of the index itself has limited physical meaning as it is normalised and weighted with other factors. 


This index is provided for reference purposes only and what being regarded as "vibrant" display is quite subjective. It will not be reliable due to algorithm limitations and model erre. It should not be relied upon for legal, financial, or operational decisions. The creators accept no responsibility for any loss or liability arising from its use.

The algorithm ignores physical limitations in the sun- Earth geometry, and will not work in mountaneous high latitude cities. The index also will not work when the presence of convective or broken cloud is predominant. The index is also 

Further work is appreciated to calibrate the parameters based on global webcams and self learning algorithms. 


### Quick Usage

visit afterglow.top 

or run locally with dependencies

python calc_afterglow_realistic_path_lwc_global.py

to generate a json schema of results.

# ACSAF: Aerosols & Cloud geometry-based Sunset/Sunrise cloud Afterglow Forecaster

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

ACSAF is a short-term forecasting tool designed to predict the occurrence and intensity of vibrant sunset and sunrise cloud afterglows for the current and following day. 

The tool computes a normalized **Afterglow Index (0–100)**:
* **0:** Dull, short-lived, or unappreciable cloud afterglow display.
* **100:** Vivid, long-lasting, and highly visible cloud afterglow display.

---

## 🛰️ How It Works

The ACSAF index is physically rooted in **Beer-Lambert's Law**, tracking light attenuation through the atmosphere. However, the final index is normalized and weighted alongside empirical factors for human perception.

The engine processes data across **13 pressure levels** at a **3-hour temporal resolution** and **0.4-degree horizontal resolution**, utilizing:
* **ECMWF IFS:** Cloud and Ice Water Content forecast data.
* **CAMS:** Atmospheric Total Aerosol Optical Depth (AOD) forecast data.

---

## ⚡ Quick Start

### Web Interface
The easiest way to view the forecast is to visit the live app:
👉 **[afterglow.top](https://afterglow.top)**

### Local Installation & Execution
To generate the raw forecast JSON schema locally, clone the repository, install your required dependencies, and run the calculation script:

```bash
git clone [https://github.com/Her0n24/ACSAF.git](https://github.com/Her0n24/ACSAF.git)
cd ACSAF
# Install your dependencies here (e.g., environment.yml)
python calc_afterglow_realistic_path_lwc_global.py

## Gallery

[afterglow-1 (dragged).tiff](https://github.com/user-attachments/files/28464819/afterglow-1.dragged.tiff)
Why such information is useful?

[afterglow-3 (dragged).tiff](https://github.com/user-attachments/files/28464823/afterglow-3.dragged.tiff)
The principle of cloud afterglow forecasting

![20250418000000_afterglow_dashboard_Reading](https://github.com/user-attachments/assets/bb6f6f12-2cab-4159-9596-dcbc0c741ce1)
Index and details of the events in the legacy ASCAF Dashboard 

![20250419000000-18h-AIFS_cloud_cover_Reading](https://github.com/user-attachments/assets/9b85c79e-5903-4509-8dce-e9cc8fe50421)
Legacy supplementary figures displaying raw model output cloud coves for the selected region
