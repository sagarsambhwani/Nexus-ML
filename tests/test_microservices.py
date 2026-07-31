import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import microservices
from api.course_service.main import app as course_app
from api.ml_service.main import app as ml_app

course_client = TestClient(course_app)
ml_client = TestClient(ml_app)


# --- Course Microservice Tests ---

def test_course_service_health():
    response = course_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "Course Microservice"
    assert response.json()["total_chapters"] > 0

def test_course_service_list_courses():
    response = course_client.get("/api/v1/courses")
    assert response.status_code == 200
    data = response.json()
    assert data["total_courses"] > 0
    assert any(c["key"] == "01_data_cleaning_and_preprocessing.md" for c in data["courses"])

def test_course_service_get_content():
    response = course_client.get("/api/v1/courses/01_data_cleaning_and_preprocessing.md")
    assert response.status_code == 200
    data = response.json()
    assert "Data Cleaning" in data["content"]


# --- ML Dashboard Microservice Tests ---

def test_ml_service_health():
    response = ml_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "ML Dashboard Microservice"
    assert response.json()["total_models"] == 12

def test_ml_service_pipeline_status():
    response = ml_client.get("/api/v1/pipelines/status")
    assert response.status_code == 200
    assert response.json()["total_pipelines"] == 12

def test_ml_service_predict_fraud():
    payload = {
        "amount": 100.0, "time_hour": 14, "velocity_1h": 1, "location_risk": 0.2,
        "v1": 0.1, "v2": 0.1, "v3": 0.1, "v4": 0.1, "v5": 0.1
    }
    response = ml_client.post("/api/v1/predict/fraud-detection", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "fraud_probability" in response.json()["result"]
