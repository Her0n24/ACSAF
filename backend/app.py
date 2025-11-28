from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, "mongo_db", ".env")
load_dotenv(ENV_PATH)

user = os.getenv("MONGODB_USER")
password = os.getenv("MONGODB_PWD")
if not (user and password):
    raise ValueError("MONGODB_USER and MONGODB_PWD must be set")
mongo_uri = f"mongodb+srv://{user}:{password}@cluster0.fpibh6i.mongodb.net/ACSAF?retryWrites=true&w=majority&appName=Cluster0"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MONGO_URI"] = mongo_uri
mongo = PyMongo(app)

def ensure_indexes():
    mongo.db.forecast_data.create_index("city", background=True)
    mongo.db.forecast_data.create_index([("city", 1), ("run_time", -1)], background=True)

ensure_indexes()

@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "afterglow-api", "version": "v1", "endpoints": ["/cities", "/forecast"]})

@app.route("/cities", methods=["GET"])
def list_cities():
    cities = mongo.db.forecast_data.distinct("city")
    return jsonify({"count": len(cities), "cities": sorted(cities)})

@app.route("/debug_count", methods=["GET"])
def debug_count():
    return jsonify({
        "db": mongo.db.name,
        "collection": "forecast_data",
        "count": mongo.db.forecast_data.count_documents({}),
        "sample": mongo.db.forecast_data.find_one({}, {"city": 1, "_id": 0})
    })

@app.route("/forecast", methods=["GET", "POST"])
def get_forecast():
    city = None
    run_time = None

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        city = payload.get("city")
        run_time = payload.get("run_time")
    else:
        city = request.args.get("city")
        run_time = request.args.get("run_time")

    if not city:
        return jsonify({"error": "city parameter required"}), 400

    query = {"city": {"$regex": f"^{city}$", "$options": "i"}}
    if run_time:
        query["run_time"] = run_time

    # Latest doc if run_time not provided
    cursor = mongo.db.forecast_data.find(query).sort("run_time", -1).limit(1)
    doc = next(cursor, None)

    if not doc:
        return jsonify({"error": f"No forecast found for {city}"}), 404

    doc["_id"] = str(doc["_id"])
    return jsonify(doc)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)