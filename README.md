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
```

## Gallery

<img width="3000" height="1687" alt="afterglow1-1 (dragged)" src="https://github.com/user-attachments/assets/fbe6848b-7fce-4eca-b987-ce4adde06b8c" />
Why such information is useful?


<img width="3000" height="1687" alt="afterglow1-3 (dragged)" src="https://github.com/user-attachments/assets/3b485d9a-eada-469f-98c8-97d2c28be7c3" />
The principle of cloud afterglow forecasting


![20250418000000_afterglow_dashboard_Reading](https://github.com/user-attachments/assets/bb6f6f12-2cab-4159-9596-dcbc0c741ce1)
Index and details of the events in the legacy ASCAF Dashboard 


![20250419000000-18h-AIFS_cloud_cover_Reading](https://github.com/user-attachments/assets/9b85c79e-5903-4509-8dce-e9cc8fe50421)
Legacy supplementary figures displaying raw model output cloud coves for the selected region
