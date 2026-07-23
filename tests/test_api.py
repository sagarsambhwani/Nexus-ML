import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure models are trained before API test execution
from scripts.train_all import train_all
train_all()

from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["total_models"] == 12

def test_pipeline_status_endpoint():
    response = client.get("/api/v1/pipelines/status")
    assert response.status_code == 200
    assert response.json()["total_pipelines"] == 12

def test_predict_fraud():
    payload = {
        "amount": 100.0, "time_hour": 14, "velocity_1h": 1, "location_risk": 0.2,
        "v1": 0.1, "v2": 0.1, "v3": 0.1, "v4": 0.1, "v5": 0.1
    }
    response = client.post("/api/v1/predict/fraud-detection", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "fraud_probability" in response.json()["result"]

def test_predict_house_prices():
    payload = {
        "sqft": 1500, "bedrooms": 3, "bathrooms": 2.0, "location_score": 5.0,
        "house_age": 10, "garage_cars": 1, "dist_city_km": 12.0
    }
    response = client.post("/api/v1/predict/house-prices", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()["result"]

def test_predict_sentiment():
    payload = {"text": "Awesome product, great build quality and fast shipping."}
    response = client.post("/api/v1/predict/sentiment-analysis", json=payload)
    assert response.status_code == 200
    assert response.json()["result"]["sentiment"] == "POSITIVE"
