import asyncio
import os
from fastapi.testclient import TestClient
from api.main import app, load_model
from src.ml.train import train_and_save

client = TestClient(app)

# Force model load before tests


asyncio.run(load_model())

# Ensure a model exists before API startup
if not os.path.exists("models/model_v1.joblib"):
    os.makedirs("models", exist_ok=True)
    train_and_save("models/model_v1.joblib")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    payload = {
        "MedInc": 3.5,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
