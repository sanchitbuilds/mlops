"""Flask REST API that serves the model (Experiment 7)."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

FEATURES = ["hour", "failed_attempts", "new_device", "foreign_country",
            "vpn_used", "hours_since_last_login"]

app = Flask(__name__)
model = joblib.load("models/model.joblib")

def risk_level(score):
    return "LOW" if score < 30 else "MEDIUM" if score < 70 else "HIGH"

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify(error=f"missing fields: {missing}"), 400
    row = pd.DataFrame([{f: data[f] for f in FEATURES}])
    score = round(float(model.predict_proba(row)[0][1]) * 100, 1)
    return jsonify(risk_score=score, risk_level=risk_level(score),
                   is_anomaly=score >= 50)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
