import pymongo
from pymongo import MongoClient
import os, sys 
# Add parent folder (/Users/hng/Documents/dev/Afterglow) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from pymongo.errors import OperationFailure
password = os.environ.get("MONGODB_PWD")
user = os.environ.get("MONGODB_USER")

def latest_forecast_hours_run_to_download() -> datetime.datetime:
    """
    Determine the latest forecast_hours run to download based on current UTC time.
    ECMWF AIFS runs are at 00z and 12z, with a 10-hour delay for data availability.
    
    Logic:
    - If current UTC time >= 10:00: Download today's 00z run
    - If current UTC time < 10:00: Download yesterday's 12z run
    - If current UTC time >= 22:00 : Download today's 12z run
    Returns:
        forecast_hours initialization time (datetime in UTC)
    """

    now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    threshold_00z = now_utc.replace(hour=10, minute=0, second=0, microsecond=0)
    threshold_12z = now_utc.replace(hour=22, minute=0, second=0, microsecond=0)
    today_end_of_day = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    if now_utc >= threshold_12z and now_utc <= today_end_of_day:
        # Return datetime for downloading today's 12z forecast_hours
        return now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    elif now_utc >= threshold_00z and now_utc < threshold_12z:
        # Return datetime for downloading today's 00z forecast_hours
        return now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Return datetime for downloading yesterday's 12z forecast_hours
        return now_utc.replace(hour=12, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)


#today = datetime.utcnow().strftime("%Y-%m-%d")
today = datetime.datetime(2025, 11, 27, 11, 0)
run = latest_forecast_hours_run_to_download().strftime("%H")
print(run)

connection_string = f"mongodb+srv://{user}:{password}@cluster0.fpibh6i.mongodb.net/?appName=Cluster0"
client = MongoClient(connection_string)
afterglow_db = client["ACSAF"]
collections = afterglow_db.list_collection_names()
print(collections)
afterglow_collection = afterglow_db["forecast_data"]

try:
    # Verify auth
    client.admin.command("ping")
    print("Mongo ping OK")
except OperationFailure as e:
    print(f"Auth failed: {e}")
    raise

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
forecast_path = os.path.join(parent_dir, f"output/all_cities_summary_{today.strftime('%Y%m%d')}_{run}Z.json")
if not os.path.exists(forecast_path):
    raise FileNotFoundError(f"Not found: {forecast_path}")

with open(forecast_path, "r") as f:
    data = json.load(f)

# Attach metadata to each city doc
if isinstance(data, list):
    for d in data:
        d["run"] = run
        d["run_time"] = today.isoformat()
    if data:
        res = afterglow_collection.insert_many(data, ordered=False)
        print(f"Inserted {len(res.inserted_ids)} documents.")
    else:
        print("No documents to insert.")
else:
    data["run"] = run
    data["run_time"] = today.isoformat()
    res = afterglow_collection.insert_one(data)
    print(f"Inserted document id: {res.inserted_id}")

print("Collections:", afterglow_db.list_collection_names())
