# ACSAF: Aerosols & Cloud geometry based Sunset/Sunrise cloud Afterglow Forecaster

The ASCAF project attempts to provide a near-term (today and tomorrow) forecast for the occurrence of a vibrant sunset cloud afterglow. The computed index ranges from 0-100, with 0 indicating a dull, short-lived and less appreciable cloud afterglow is probable while 100 indicating a vilvid, long lasting and very visible cloud afterglow is probable.

The index is based on the 00Z ECMWF AIFS cloud cover and CAMS atmosphere optical depth forecast data for the next two days made available daily at 10Z. Various Atmospheric parameters are taken into condition when computing the index but the value of the index itself has no physical meaning.
The script can be applied to elsewhere on the planet and for the 12Z run but due to physical limitations in the sun- Earth geometry, it will not work in high latitude regions. The index also will not work when the presence of convective or broken cloud is predominant.

Further work is required to calibrate the index from observation and factoring in more parameters such as columnal total water content, total ice content, coincidental cloud cover at various layers, and types of cloud.

This index is provided for reference purposes only and will not be reliable due to algorithm limitations and model erreo. It should not be relied upon for legal, financial, or operational decisions. The creators accept no responsibility for any loss or liability arising from its use.

### Quick Usage

python calc_afterglow_cover_strategy.py

## Gallery
![20250418000000_afterglow_dashboard_Reading](https://github.com/user-attachments/assets/bb6f6f12-2cab-4159-9596-dcbc0c741ce1)
Index and details of the events in ASCAF Dashboard 

![20250419000000-18h-AIFS_cloud_cover_Reading](https://github.com/user-attachments/assets/9b85c79e-5903-4509-8dce-e9cc8fe50421)
Supplementary figures displaying raw model output cloud coves for the selected region
