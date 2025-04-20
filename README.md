# ACSAF: Aerosol & Cloud geometry based Sunset/Sunrise cloud Afterglow Forecaster

The ASCAF project attempts to provide a near-term (today and tomorrow) forecast for the occurrence of a vibrant sunset cloud afterglow. The computed index ranges from 0-100, with 0 indicating a dull, short-lived and less appreciable cloud afterglow is probable while 100 indicating a vilvid, long lasting and very visible cloud afterglow is probable.

The index is based on the 00Z ECMWF AIFS cloud cover and CAMS AOD550 forecast data for the next two days made available daily at 10Z. Various Atmospheric parameters are taken into condition when computing the index but the value of the index itself has no physical meaning.
The script can be applied to elsewhere on the planet and for the 12Z run but due to physical limitations in the sun- Earth geometry, it will not work in high latitude regions. The index also will not work when the presence of convective or broken cloud is predominant.

Further work is required to calibrate the index from observation and factoring in more parameters such as columnal total water content, total ice content, coincidental cloud cover at various layers, and types of cloud.
