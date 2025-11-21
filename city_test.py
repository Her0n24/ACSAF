
cities = [
{"name": "London",   "country": "UK", "tz": "Europe/London",    "lat": 51.5074, "lon": -0.1278, "create_dashboard": True},
{"name": "New_York","country": "US", "tz": "America/New_York",  "lat": 40.7128, "lon": -74.0060, "create_dashboard": True},
{"name": "Tokyo",    "country": "JP", "tz": "Asia/Tokyo",       "lat": 35.6895, "lon": 139.6917, "create_dashboard": True},
]

# def run_tests():
#     run_dt = datetime.datetime(2025, 11, 20, 12, 0, tzinfo=datetime.timezone.utc)
#     current_utc = run_dt + datetime.timedelta(hours=22) #simulate current time at 10 UTC or 22 UTC
#     cities = [
#         {"name": "London", "country": "UK", "tz": "Europe/London", "lat": 51.5074, "lon": -0.1278},
#         {"name": "New_York", "country": "US", "tz": "America/New_York", "lat": 40.7128, "lon": -74.0060},
#         {"name": "Tokyo", "country": "JP", "tz": "Asia/Tokyo", "lat": 35.6895, "lon": 139.6917},
#         {"name": "Sydney", "country": "AU", "tz": "Australia/Sydney", "lat": -33.8688, "lon": 151.2093},
#         {"name": "Los_Angeles", "country": "US", "tz": "America/Los_Angeles", "lat": 34.0522, "lon": -118.2437},
#     ]

#     print("=== Real-city tests ===")
#     for c in cities:
#         city = LocationInfo(c["name"], c["country"], c["tz"], c["lat"], c["lon"])
#         tz = pytz.timezone(c["tz"])

#         # local date anchored to actual run-time (10Z) so we compute the coming events
#         local_ref_date = current_utc.astimezone(tz).date()

#         s_today = sun(city.observer, date=local_ref_date, tzinfo=tz)
#         sunrise_today = s_today["sunrise"]
#         sunset_today  = s_today["sunset"]

#         # also get tomorrow's events (coming) for completeness
#         s_tom = sun(city.observer, date=local_ref_date + datetime.timedelta(days=1), tzinfo=tz)
#         sunrise_tom = s_tom["sunrise"]
#         sunset_tom  = s_tom["sunset"]

#         # choose which event datetimes to pass to determine_city_forecast_time_to_use:
#         # pass the event datetimes for the coming local day (today & tomorrow)
#         res = determine_city_forecast_time_to_use(city, sunrise_today, sunset_today, run_dt, time_threshold_h=15)
#         print(c["name"], "local_ref_date:", local_ref_date, "->", res)


import datetime
from astral import LocationInfo
from city_test import cities           # must be a module-level list as above
from calc_afterglow_cover_strategy_global import process_city, latest_forecast_hours_run_to_download
from get_cds_global import get_cams_aod
from get_aifs import download_file
import requests
import logging
import os


def main():

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Afterglow', 'output'))
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Afterglow', 'input'))
    today = datetime.date.today() - datetime.timedelta(days=1)
    today_str = today.strftime("%Y%m%d")

    print("today_str:", today_str)
    run = "00".zfill(2)
    # run_dt = latest_forecast_hours_run_to_download()
    # run = str(run_dt.hour).zfill(2)

    failed_count = 0
    for fcst_hour in range(0, 61, 6):
        try:
            if failed_count >= 5:
                print("Too many failed downloads, aborting.")
                quit()
            url = f"https://data.ecmwf.int/forecasts/{today_str}/{run}z/aifs-single/0p25/oper/{today_str}{run}0000-{fcst_hour}h-oper-fc.grib2"   
            download_file(url, f"{input_path}/{today_str}{run}0000-{fcst_hour}h-oper-fc.grib2")
        except requests.HTTPError as e:
            logging.warning(f"HTTP error downloading hour {fcst_hour}: {e}")
            failed_count += 1
        except Exception as e:
            logging.exception(f"Unexpected error downloading hour {fcst_hour}")
            failed_count += 1

    # Need global dataset for this
    get_cams_aod(today, run, today_str, input_path) # type: ignore

    results = []
    for c in cities:
        try:
            res = process_city(
                c["name"],
                c.get("country", ""),
                float(c["lat"]),
                float(c["lon"]),
                c["tz"],
                today,
                run,
                today_str,
                input_path,  # input_path: process_city uses module-level input_path if None (or adjust)
                output_path,  # output_path: same as above
                c.get("create_dashboard", False)
            )
            results.append(res)
            print("OK:", c["name"], "->", res.get("sunset_time_tdy"))
        except Exception as e:
            print("ERROR:", c["name"], e)

    # optional: save results to CSV
    try:
        import pandas as pd
        pd.DataFrame(results).to_csv(f"subset_results_{today_str}.csv", index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()