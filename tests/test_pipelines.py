import sys
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.fraud_detection.pipeline import FraudDetectionPipeline
from src.credit_risk.pipeline import CreditRiskPipeline
from src.customer_churn.pipeline import CustomerChurnPipeline
from src.house_prices.pipeline import HousePricePipeline
from src.recommendation.pipeline import RecommendationPipeline
from src.demand_forecasting.pipeline import DemandForecastingPipeline
from src.predictive_maintenance.pipeline import PredictiveMaintenancePipeline
from src.medical_diagnosis.pipeline import MedicalDiagnosisPipeline
from src.sentiment_analysis.pipeline import SentimentAnalysisPipeline
from src.document_classification.pipeline import DocumentClassificationPipeline
from src.defect_detection.pipeline import DefectDetectionPipeline
from src.customer_segmentation.pipeline import CustomerSegmentationPipeline

def test_fraud_detection():
    pipe = FraudDetectionPipeline()
    metrics = pipe.train()
    assert "roc_auc" in metrics
    
    pred = pipe.predict({
        "amount": 500.0, "time_hour": 3, "velocity_1h": 5, "location_risk": 0.8,
        "v1": 1.5, "v2": -0.5, "v3": 0.2, "v4": 1.1, "v5": -0.8
    })
    assert "fraud_probability" in pred
    assert "alert_status" in pred

def test_credit_risk():
    pipe = CreditRiskPipeline()
    metrics = pipe.train()
    assert "roc_auc" in metrics

    pred = pipe.predict({
        "credit_score": 720, "annual_income": 85000, "dti_ratio": 0.22,
        "loan_amount": 15000, "delinquencies_2yr": 0, "employment_years": 5
    })
    assert "default_probability" in pred
    assert pred["underwriting_decision"] in ["APPROVED", "MANUAL_REVIEW", "REJECTED"]

def test_customer_churn():
    pipe = CustomerChurnPipeline()
    metrics = pipe.train()
    assert "accuracy" in metrics

    pred = pipe.predict({
        "tenure": 12, "monthly_charges": 65.0, "total_charges": 780.0,
        "contract_type": 1, "support_tickets": 1, "paperless_billing": 0
    })
    assert "churn_probability" in pred
    assert "risk_level" in pred

def test_house_prices():
    pipe = HousePricePipeline()
    metrics = pipe.train()
    assert "rmse" in metrics

    pred = pipe.predict({
        "sqft": 2000, "bedrooms": 3, "bathrooms": 2.0, "location_score": 7.0,
        "house_age": 10, "garage_cars": 2, "dist_city_km": 10.0
    })
    assert pred["predicted_price"] > 0
    assert "valuation_range" in pred

def test_recommendation():
    pipe = RecommendationPipeline()
    metrics = pipe.train()
    assert "explained_variance" in metrics

    pred = pipe.predict({"user_id": "USER_001", "category": "All", "top_n": 3})
    assert len(pred["recommendations"]) > 0

def test_demand_forecasting():
    pipe = DemandForecastingPipeline()
    metrics = pipe.train()
    assert "mae" in metrics

    pred = pipe.predict({"store_id": "STORE_101", "horizon_days": 5, "is_promo": 1})
    assert len(pred["daily_forecast"]) == 5

def test_predictive_maintenance():
    pipe = PredictiveMaintenancePipeline()
    metrics = pipe.train()
    assert "recall" in metrics

    pred = pipe.predict({
        "vibration_hz": 40.0, "temperature_c": 65.0, "pressure_psi": 50.0,
        "rpm": 2000, "sensor_noise_std": 1.0, "operating_hours": 3000
    })
    assert "failure_probability" in pred
    assert "health_status" in pred

def test_medical_diagnosis():
    pipe = MedicalDiagnosisPipeline()
    metrics = pipe.train()
    assert "roc_auc" in metrics

    pred = pipe.predict({
        "age": 45, "glucose": 110.0, "blood_pressure": 82.0, "bmi": 26.5,
        "hba1c": 5.8, "family_history": 0, "smoker": 0
    })
    assert "disease_risk_probability" in pred

def test_sentiment_analysis():
    pipe = SentimentAnalysisPipeline()
    metrics = pipe.train()
    assert "accuracy" in metrics

    pred = pipe.predict({"text": "Super incredible quality and amazing fast shipping!"})
    assert pred["sentiment"] in ["POSITIVE", "NEUTRAL", "NEGATIVE"]

def test_document_classification():
    pipe = DocumentClassificationPipeline()
    metrics = pipe.train()
    assert "accuracy" in metrics

    pred = pipe.predict({"text": "Senior Python Backend Developer with FastAPI Docker Kubernetes experience"})
    assert pred["predicted_category"] == "ENGINEERING_RESUME"

def test_defect_detection():
    pipe = DefectDetectionPipeline()
    metrics = pipe.train()
    assert "accuracy" in metrics

    pred = pipe.predict({
        "mean_intensity": 150.0, "std_intensity": 10.0, "edge_pixel_density": 0.05,
        "contrast_ratio": 1.5, "surface_roughness": 1.0, "anomaly_patch_max": 0.1
    })
    assert "defect_type" in pred
    assert "quality_control_passed" in pred

def test_customer_segmentation():
    pipe = CustomerSegmentationPipeline()
    metrics = pipe.train()
    assert "n_clusters" in metrics

    pred = pipe.predict({
        "annual_income_k": 100.0, "spending_score": 85.0,
        "frequency_purchases": 25.0, "recency_days": 10.0
    })
    assert "cluster_id" in pred
    assert "persona_name" in pred
