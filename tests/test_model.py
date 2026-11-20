import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

def setup_module(_):
    subprocess.run([sys.executable, "src/generate_data.py"], check=True)
    subprocess.run([sys.executable, "src/train.py"], check=True)

def client():
    from src.app import app
    return app.test_client()

def test_model_quality():
    m = json.load(open("models/metrics.json"))
    assert m["tuned_model"]["f1"] > 0.6

def test_high_risk_login():
    r = client().post("/predict", json={"hour": 3, "failed_attempts": 5,
        "new_device": 1, "foreign_country": 1, "vpn_used": 1,
        "hours_since_last_login": 200})
    assert r.status_code == 200 and r.get_json()["risk_level"] == "HIGH"

def test_low_risk_login():
    r = client().post("/predict", json={"hour": 14, "failed_attempts": 0,
        "new_device": 0, "foreign_country": 0, "vpn_used": 0,
        "hours_since_last_login": 20})
    assert r.get_json()["risk_level"] == "LOW"

def test_bad_input():
    assert client().post("/predict", json={"hour": 3}).status_code == 400
