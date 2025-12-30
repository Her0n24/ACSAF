import dns.resolver
resolver = dns.resolver.Resolver()
resolver.timeout = 5       # per try (s)
resolver.lifetime = 30     # total (s)
dns.resolver.default_resolver = resolver

import argparse
import pymongo
from pymongo import MongoClient, UpdateOne
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError, ConfigurationError
from pymongo.write_concern import WriteConcern
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

# Optional environment variables to control direct hosts fallback and DB settings
direct_hosts_env = os.environ.get("MONGODB_DIRECT_HOSTS")  # comma-separated host:port list
replica_set = os.environ.get("MONGODB_REPLICA_SET")
db_name = os.environ.get("MONGODB_DB", "ACSAF")
auth_source = os.environ.get("MONGODB_AUTH_SOURCE", "admin")
use_tls = os.environ.get("MONGODB_USE_TLS", "true").lower() in ("1", "true", "yes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload forecast JSON into MongoDB (defaults to latest run unless overridden)."
    )
    parser.add_argument(
        "--date",
        help="UTC date for the forecast run in YYYYMMDD format (e.g., 20251128).",
    )
    parser.add_argument(
        "--run",
        dest="run",
        help="Run hour in HH format (00 or 12).",
    )

    args = parser.parse_args()
    if bool(args.date) ^ bool(args.run):
        parser.error("--date and --run must be provided together or omitted together.")
    if args.run and args.run not in {"00", "12"}:
        parser.error("--run must be '00' or '12'.")
    if args.date:
        try:
            datetime.datetime.strptime(args.date, "%Y%m%d")
        except ValueError as exc:
            parser.error(f"--date must be YYYYMMDD. Received {args.date!r}: {exc}.")
    return args

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


cli_args = parse_args()
if cli_args.date and cli_args.run:
    run_datetime = datetime.datetime.strptime(
        f"{cli_args.date}{cli_args.run}", "%Y%m%d%H"
    ).replace(tzinfo=datetime.timezone.utc)
else:
    run_datetime = latest_forecast_hours_run_to_download()
run = run_datetime.strftime("%H")
run_date = run_datetime.strftime("%Y%m%d")
print(f"Using run {run}Z on {run_date}")

# Build SRV (default) and direct connection strings
srv_connection_string = f"mongodb+srv://{user}:{password}@cluster0.fpibh6i.mongodb.net/?appName=Cluster0"
direct_connection_string = None
if direct_hosts_env:
    hosts = ",".join([h.strip() for h in direct_hosts_env.split(",") if h.strip()])
    tls_param = "true" if use_tls else "false"
    replica_part = f"&replicaSet={replica_set}" if replica_set else ""
    direct_connection_string = (
        f"mongodb://{user}:{password}@{hosts}/{db_name}?authSource={auth_source}&tls={tls_param}&retryWrites=true{replica_part}"
    )

# Try SRV first, then fallback to direct if provided
client = None
last_exception = None
attempts = []
for uri_label, uri in (("SRV", srv_connection_string), ("Direct", direct_connection_string)):
    if not uri:
        continue
    try:
        print(f"Attempting MongoDB connection using: {uri_label} URI")
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=120000,
            connectTimeoutMS=60000,
            socketTimeoutMS=120000,
        )
        client.admin.command("ping")
        print(f"{uri_label} Mongo ping OK")
        attempts.append((uri_label, True, None))
        break
    except (ServerSelectionTimeoutError, ConfigurationError, OperationFailure, Exception) as exc:
        print(f"{uri_label} connection failed: {exc}")
        last_exception = exc
        attempts.append((uri_label, False, str(exc)))

if client is None:
    print("All connection attempts failed:")
    for label, ok, msg in attempts:
        print(f" - {label}: {'OK' if ok else 'FAILED'} {msg or ''}")
    raise last_exception

afterglow_db = client[db_name]
collections = afterglow_db.list_collection_names()
print(collections)
# Use a write concern with a higher wtimeout to avoid transient write timeouts
wc = WriteConcern(w=1, wtimeout=120000)
afterglow_collection = afterglow_db.get_collection("forecast_data").with_options(write_concern=wc)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
forecast_path = os.path.join(parent_dir, f"output/all_cities_summary_{run_date}_{run}Z.json")
if not os.path.exists(forecast_path):
    raise FileNotFoundError(f"Not found: {forecast_path}")

with open(forecast_path, "r") as f:
    data = json.load(f)

def attach_metadata(doc: dict) -> dict:
    doc["run"] = run
    doc["run_time"] = run_datetime.isoformat()
    doc["uploaded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return doc


def build_filter(doc: dict) -> dict:
    city = doc.get("city")
    if not city:
        raise ValueError("Each document must include a 'city' field for upsert filtering.")
    return {"city": city, "run_time": doc["run_time"]}


# Attach metadata to each city doc
if isinstance(data, list):
    bulk_ops: list[UpdateOne] = []
    for doc in data:
        doc_with_meta = attach_metadata(doc)
        bulk_ops.append(
            UpdateOne(
                build_filter(doc_with_meta),
                {"$set": doc_with_meta},
                upsert=True,
            )
        )

    if bulk_ops:
        result = afterglow_collection.bulk_write(bulk_ops, ordered=False)
        print(
            f"Upserts: {result.upserted_count}, "
            f"Modified: {result.modified_count}"
        )
    else:
        print("No documents to insert.")
else:
    doc_with_meta = attach_metadata(data)
    res = afterglow_collection.update_one(
        build_filter(doc_with_meta),
        {"$set": doc_with_meta},
        upsert=True,
    )
    if res.upserted_id:
        print(f"Upserted new document id: {res.upserted_id}")
    else:
        print("Updated existing document.")

print("Collections:", afterglow_db.list_collection_names())
