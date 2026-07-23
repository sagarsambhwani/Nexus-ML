from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 01 Fraud Detection ---
class FraudDetectionRequest(BaseModel):
    amount: float = Field(..., json_schema_extra={"example": 250.50})
    time_hour: int = Field(..., ge=0, le=23, json_schema_extra={"example": 2})
    velocity_1h: int = Field(..., json_schema_extra={"example": 4})
    location_risk: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.85})
    v1: float = Field(default=0.0, json_schema_extra={"example": 1.8})
    v2: float = Field(default=0.0, json_schema_extra={"example": -0.5})
    v3: float = Field(default=0.0, json_schema_extra={"example": 0.2})
    v4: float = Field(default=0.0, json_schema_extra={"example": 1.1})
    v5: float = Field(default=0.0, json_schema_extra={"example": -0.8})

# --- 02 Credit Risk ---
class CreditRiskRequest(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, json_schema_extra={"example": 640})
    annual_income: float = Field(..., json_schema_extra={"example": 55000})
    dti_ratio: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.42})
    loan_amount: float = Field(..., json_schema_extra={"example": 25000})
    delinquencies_2yr: int = Field(..., json_schema_extra={"example": 1})
    employment_years: int = Field(..., json_schema_extra={"example": 3})

# --- 03 Customer Churn ---
class CustomerChurnRequest(BaseModel):
    tenure: int = Field(..., json_schema_extra={"example": 6})
    monthly_charges: float = Field(..., json_schema_extra={"example": 89.90})
    total_charges: float = Field(..., json_schema_extra={"example": 539.40})
    contract_type: int = Field(..., description="0: Month-to-month, 1: 1-Year, 2: 2-Year", json_schema_extra={"example": 0})
    support_tickets: int = Field(..., json_schema_extra={"example": 4})
    paperless_billing: int = Field(..., json_schema_extra={"example": 1})

# --- 04 House Prices ---
class HousePriceRequest(BaseModel):
    sqft: float = Field(..., json_schema_extra={"example": 2200})
    bedrooms: int = Field(..., json_schema_extra={"example": 3})
    bathrooms: float = Field(..., json_schema_extra={"example": 2.5})
    location_score: float = Field(..., ge=1.0, le=10.0, json_schema_extra={"example": 8.5})
    house_age: int = Field(..., json_schema_extra={"example": 10})
    garage_cars: int = Field(..., json_schema_extra={"example": 2})
    dist_city_km: float = Field(..., json_schema_extra={"example": 6.2})

# --- 05 Recommendation ---
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., json_schema_extra={"example": "USER_005"})
    category: str = Field(default="All", json_schema_extra={"example": "Electronics"})
    top_n: int = Field(default=5, ge=1, le=20, json_schema_extra={"example": 5})

# --- 06 Demand Forecasting ---
class DemandForecastingRequest(BaseModel):
    store_id: str = Field(default="STORE_101", json_schema_extra={"example": "STORE_101"})
    horizon_days: int = Field(default=7, ge=1, le=30, json_schema_extra={"example": 7})
    is_promo: int = Field(default=0, json_schema_extra={"example": 1})

# --- 07 Predictive Maintenance ---
class PredictiveMaintenanceRequest(BaseModel):
    vibration_hz: float = Field(..., json_schema_extra={"example": 68.5})
    temperature_c: float = Field(..., json_schema_extra={"example": 92.3})
    pressure_psi: float = Field(..., json_schema_extra={"example": 78.0})
    rpm: float = Field(..., json_schema_extra={"example": 2800})
    sensor_noise_std: float = Field(..., json_schema_extra={"example": 3.2})
    operating_hours: float = Field(..., json_schema_extra={"example": 6500})

# --- 08 Medical Diagnosis ---
class MedicalDiagnosisRequest(BaseModel):
    age: int = Field(..., json_schema_extra={"example": 54})
    glucose: float = Field(..., json_schema_extra={"example": 145.0})
    blood_pressure: float = Field(..., json_schema_extra={"example": 95.0})
    bmi: float = Field(..., json_schema_extra={"example": 32.4})
    hba1c: float = Field(..., json_schema_extra={"example": 6.8})
    family_history: int = Field(..., json_schema_extra={"example": 1})
    smoker: int = Field(..., json_schema_extra={"example": 0})

# --- 09 Sentiment Analysis ---
class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "The product exceeded my expectations! Super fast delivery and great quality."})

# --- 10 Document Classification ---
class DocumentClassificationRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "Senior Software Engineer with 6 years experience in Python, Docker, Kubernetes, microservices, and FastAPI backend development."})

# --- 11 Defect Detection ---
class DefectDetectionRequest(BaseModel):
    mean_intensity: float = Field(..., json_schema_extra={"example": 110.0})
    std_intensity: float = Field(..., json_schema_extra={"example": 35.2})
    edge_pixel_density: float = Field(..., json_schema_extra={"example": 0.22})
    contrast_ratio: float = Field(..., json_schema_extra={"example": 5.1})
    surface_roughness: float = Field(..., json_schema_extra={"example": 4.2})
    anomaly_patch_max: float = Field(..., json_schema_extra={"example": 0.82})

# --- 12 Customer Segmentation ---
class CustomerSegmentationRequest(BaseModel):
    annual_income_k: float = Field(..., json_schema_extra={"example": 105.0})
    spending_score: float = Field(..., ge=1, le=100, json_schema_extra={"example": 88.0})
    frequency_purchases: float = Field(..., json_schema_extra={"example": 22.0})
    recency_days: float = Field(..., json_schema_extra={"example": 14.0})

# --- Standard Generic API Response ---
class StandardResponse(BaseModel):
    pipeline: str
    status: str = "success"
    result: Dict[str, Any]
