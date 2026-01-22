import dns.resolver
resolver = dns.resolver.Resolver()
resolver.timeout = 5
resolver.lifetime = 30
dns.resolver.default_resolver = resolver

import os
import sys
import argparse
import datetime
import json
from dotenv import load_dotenv, find_dotenv
import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError, OperationFailure

# Add project root so env helpers in parent can be imported if needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

load_dotenv(find_dotenv())

password = os.environ.get("MONGODB_PWD")
user = os.environ.get("MONGODB_USER")
db_name = os.environ.get("MONGODB_DB", "ACSAF")
auth_source = os.environ.get("MONGODB_AUTH_SOURCE", "admin")
direct_hosts_env = os.environ.get("MONGODB_DIRECT_HOSTS")
replica_set = os.environ.get("MONGODB_REPLICA_SET")
use_tls = os.environ.get("MONGODB_USE_TLS", "true").lower() in ("1", "true", "yes")


def parse_args():
	p = argparse.ArgumentParser(description="Delete oldest forecast run from MongoDB collection 'forecast_data'.")
	p.add_argument("--yes", action="store_true", help="Actually perform deletion. Otherwise does a dry-run.")
	p.add_argument("--limit", type=int, default=1, help="Number of oldest runs to delete (default: 1)")
	return p.parse_args()


def build_connection():
	srv_connection_string = f"mongodb+srv://{user}:{password}@cluster0.fpibh6i.mongodb.net/?appName=Cluster0"
	direct_connection_string = None
	if direct_hosts_env:
		hosts = ",".join([h.strip() for h in direct_hosts_env.split(",") if h.strip()])
		tls_param = "true" if use_tls else "false"
		replica_part = f"&replicaSet={replica_set}" if replica_set else ""
		direct_connection_string = (
			f"mongodb://{user}:{password}@{hosts}/{db_name}?authSource={auth_source}&tls={tls_param}&retryWrites=true{replica_part}"
		)

	client = None
	last_exc = None
	for label, uri in (("SRV", srv_connection_string), ("Direct", direct_connection_string)):
		if not uri:
			continue
		try:
			print(f"Attempting MongoDB connection using: {label} URI")
			client = MongoClient(uri, serverSelectionTimeoutMS=120000, connectTimeoutMS=60000, socketTimeoutMS=120000)
			client.admin.command("ping")
			print(f"{label} Mongo ping OK")
			return client
		except (ServerSelectionTimeoutError, ConfigurationError, OperationFailure, Exception) as exc:
			print(f"{label} connection failed: {exc}")
			last_exc = exc

	raise last_exc or RuntimeError("No MongoDB URI available")


def main():
	args = parse_args()
	client = build_connection()
	db = client[db_name]
	coll = db.get_collection("forecast_data")

	# Find up to N distinct oldest run_time values, in ascending order
	# We sort by run_time (ISO strings sort lexicographically correctly) then collect distinct run_time values
	cursor = coll.find({}, {"run_time": 1}).sort("run_time", pymongo.ASCENDING)
	oldest_run_times = []
	for doc in cursor:
		rt = doc.get("run_time")
		if not rt:
			continue
		if not oldest_run_times or oldest_run_times[-1] != rt:
			oldest_run_times.append(rt)
		if len(oldest_run_times) >= args.limit:
			break

	if not oldest_run_times:
		print("No documents with 'run_time' found in collection 'forecast_data'. Nothing to delete.")
		return

	# Summary
	print("Oldest run_time values to remove:")
	for rt in oldest_run_times:
		count = coll.count_documents({"run_time": rt})
		print(f" - {rt}  (documents: {count})")

	if not args.yes:
		print("Dry-run mode: no documents were deleted. Re-run with --yes to delete the above runs.")
		return

	# Perform deletions
	total_deleted = 0
	for rt in oldest_run_times:
		res = coll.delete_many({"run_time": rt})
		print(f"Deleted {res.deleted_count} documents with run_time={rt}")
		total_deleted += res.deleted_count

	print(f"Total documents deleted: {total_deleted}")


if __name__ == "__main__":
	main()

