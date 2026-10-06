"""Flask API.  Run:  python app.py   then open http://127.0.0.1:5000"""
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, redirect
from pathlib import Path
from src.risk_logic import assess_risk

BASE = Path(__file__).parent
app = Flask(__name__)

model, meta = None, None
mp = BASE / "models" / "accident_model.joblib"
if mp.exists():
    model = joblib.load(mp)
    meta = json.loads((BASE / "models" / "road_model_metadata.json").read_text())


@app.route("/")
def home():
    return redirect("/dashboard")


@app.route("/risk", methods=["POST"])
def risk():
    d = request.get_json(force=True)
    return jsonify(assess_risk(
        speed_kmh=float(d.get("speed_kmh", 40)),
        visibility_m=float(d.get("visibility_m", 1000)),
        rainfall_mm=float(d.get("rainfall_mm", 0)),
        hazard=d.get("hazard", "none"),
        hazard_distance_m=d.get("hazard_distance_m"),
        nearby_vehicles=int(d.get("nearby_vehicles", 0))))


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet. Run src/04_train_model.py"}), 503
    d = request.get_json(force=True)
    cols = meta["numeric_features"] + meta["categorical_features"]
    row = pd.DataFrame([{c: d.get(c) for c in cols}])
    for c in meta["numeric_features"]:
        row[c] = pd.to_numeric(row[c], errors="coerce")
    return jsonify({"target": meta["target"], "prediction": str(model.predict(row)[0]),
                    "expected_fields": cols})


@app.route("/dashboard")
def dashboard():
    return send_from_directory(BASE / "dashboard", "index.html")


@app.route("/readme")
def readme():
    try:
        return (BASE / "README.md").read_text(encoding="utf-8"), 200, {"Content-Type": "text/plain; charset=utf-8"}
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
