from pydantic import BaseModel, Field
from datetime import datetime

class AfterglowResult(BaseModel):
    city: str
    country: str
    run_time: datetime
    run: str                # '00' or '12'
    forecast_hour: int
    sunrise: bool
    sunset: bool
    sunset_azimuth: float | None
    sunrise_azimuth: float | None
    max_solar_elev: float | None
    cloud_stats: dict       # aggregated values
    aod: float | None
    score: float | None