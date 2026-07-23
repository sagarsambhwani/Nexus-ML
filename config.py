import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
DATA_DIR = ARTIFACTS_DIR / "data"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Pipeline Identifiers
PIPELINES = [
    "fraud_detection",
    "credit_risk",
    "customer_churn",
    "house_prices",
    "recommendation",
    "demand_forecasting",
    "predictive_maintenance",
    "medical_diagnosis",
    "sentiment_analysis",
    "document_classification",
    "defect_detection",
    "customer_segmentation"
]
