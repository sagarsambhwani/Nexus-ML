from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NEXUS_ML_DIR = BASE_DIR / "nexus_ml"

for p in [BASE_DIR, NEXUS_ML_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from api.schemas import (
    FraudDetectionRequest, CreditRiskRequest, CustomerChurnRequest,
    HousePriceRequest, RecommendationRequest, DemandForecastingRequest,
    PredictiveMaintenanceRequest, MedicalDiagnosisRequest, SentimentAnalysisRequest,
    DocumentClassificationRequest, DefectDetectionRequest, CustomerSegmentationRequest,
    StandardResponse
)
from nexus_ml.src.fraud_detection.pipeline import FraudDetectionPipeline
from nexus_ml.src.credit_risk.pipeline import CreditRiskPipeline
from nexus_ml.src.customer_churn.pipeline import CustomerChurnPipeline
from nexus_ml.src.house_prices.pipeline import HousePricePipeline
from nexus_ml.src.recommendation.pipeline import RecommendationPipeline
from nexus_ml.src.demand_forecasting.pipeline import DemandForecastingPipeline
from nexus_ml.src.predictive_maintenance.pipeline import PredictiveMaintenancePipeline
from nexus_ml.src.medical_diagnosis.pipeline import MedicalDiagnosisPipeline
from nexus_ml.src.sentiment_analysis.pipeline import SentimentAnalysisPipeline
from nexus_ml.src.document_classification.pipeline import DocumentClassificationPipeline
from nexus_ml.src.defect_detection.pipeline import DefectDetectionPipeline
from nexus_ml.src.customer_segmentation.pipeline import CustomerSegmentationPipeline

router = APIRouter(prefix="/api/v1", tags=["ML Inference Microservice Endpoints"])

PIPELINES_CACHE: Dict[str, Any] = {}

def get_pipeline(name: str):
    if name not in PIPELINES_CACHE:
        if name == "fraud_detection":
            pipe = FraudDetectionPipeline()
        elif name == "credit_risk":
            pipe = CreditRiskPipeline()
        elif name == "customer_churn":
            pipe = CustomerChurnPipeline()
        elif name == "house_prices":
            pipe = HousePricePipeline()
        elif name == "recommendation":
            pipe = RecommendationPipeline()
        elif name == "demand_forecasting":
            pipe = DemandForecastingPipeline()
        elif name == "predictive_maintenance":
            pipe = PredictiveMaintenancePipeline()
        elif name == "medical_diagnosis":
            pipe = MedicalDiagnosisPipeline()
        elif name == "sentiment_analysis":
            pipe = SentimentAnalysisPipeline()
        elif name == "document_classification":
            pipe = DocumentClassificationPipeline()
        elif name == "defect_detection":
            pipe = DefectDetectionPipeline()
        elif name == "customer_segmentation":
            pipe = CustomerSegmentationPipeline()
        else:
            raise HTTPException(status_code=404, detail=f"Unknown pipeline {name}")
        
        pipe.load()
        PIPELINES_CACHE[name] = pipe

    return PIPELINES_CACHE[name]


@router.post("/predict/fraud-detection", response_model=StandardResponse)
def predict_fraud(req: FraudDetectionRequest):
    pipe = get_pipeline("fraud_detection")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="fraud_detection", result=res)


@router.post("/predict/credit-risk", response_model=StandardResponse)
def predict_credit_risk(req: CreditRiskRequest):
    pipe = get_pipeline("credit_risk")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="credit_risk", result=res)


@router.post("/predict/customer-churn", response_model=StandardResponse)
def predict_customer_churn(req: CustomerChurnRequest):
    pipe = get_pipeline("customer_churn")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="customer_churn", result=res)


@router.post("/predict/house-prices", response_model=StandardResponse)
def predict_house_prices(req: HousePriceRequest):
    pipe = get_pipeline("house_prices")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="house_prices", result=res)


@router.post("/predict/recommendation", response_model=StandardResponse)
def predict_recommendation(req: RecommendationRequest):
    pipe = get_pipeline("recommendation")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="recommendation", result=res)


@router.post("/predict/demand-forecasting", response_model=StandardResponse)
def predict_demand_forecasting(req: DemandForecastingRequest):
    pipe = get_pipeline("demand_forecasting")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="demand_forecasting", result=res)


@router.post("/predict/predictive-maintenance", response_model=StandardResponse)
def predict_predictive_maintenance(req: PredictiveMaintenanceRequest):
    pipe = get_pipeline("predictive_maintenance")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="predictive_maintenance", result=res)


@router.post("/predict/medical-diagnosis", response_model=StandardResponse)
def predict_medical_diagnosis(req: MedicalDiagnosisRequest):
    pipe = get_pipeline("medical_diagnosis")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="medical_diagnosis", result=res)


@router.post("/predict/sentiment-analysis", response_model=StandardResponse)
def predict_sentiment_analysis(req: SentimentAnalysisRequest):
    pipe = get_pipeline("sentiment_analysis")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="sentiment_analysis", result=res)


@router.post("/predict/document-classification", response_model=StandardResponse)
def predict_document_classification(req: DocumentClassificationRequest):
    pipe = get_pipeline("document_classification")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="document_classification", result=res)


@router.post("/predict/defect-detection", response_model=StandardResponse)
def predict_defect_detection(req: DefectDetectionRequest):
    pipe = get_pipeline("defect_detection")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="defect_detection", result=res)


@router.post("/predict/customer-segmentation", response_model=StandardResponse)
def predict_customer_segmentation(req: CustomerSegmentationRequest):
    pipe = get_pipeline("customer_segmentation")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="customer_segmentation", result=res)


@router.get("/pipelines/status")
def get_all_pipelines_status():
    status_list = []
    pipeline_names = [
        ("fraud_detection", "Fraud Detection", "Classification"),
        ("credit_risk", "Credit Risk Prediction", "Classification"),
        ("customer_churn", "Customer Churn Prediction", "Classification"),
        ("house_prices", "House Price Prediction", "Regression"),
        ("recommendation", "Recommendation System", "Hybrid Matrix Factorization"),
        ("demand_forecasting", "Demand Forecasting", "Time Series"),
        ("predictive_maintenance", "Predictive Maintenance", "Telemetry / Classification"),
        ("medical_diagnosis", "Medical Diagnosis Support", "Healthcare Classification"),
        ("sentiment_analysis", "Sentiment Analysis", "NLP Classification"),
        ("document_classification", "Document Classification", "NLP Classification"),
        ("defect_detection", "Defect Detection", "Computer Vision"),
        ("customer_segmentation", "Customer Segmentation", "Clustering")
    ]
    for key, display_name, task_type in pipeline_names:
        pipe = get_pipeline(key)
        metrics = pipe.model.get("metrics", {}) if pipe.model else {}
        status_list.append({
            "key": key,
            "name": display_name,
            "task_type": task_type,
            "artifact": pipe.artifact_name,
            "metrics": metrics
        })
    return {"total_pipelines": len(status_list), "pipelines": status_list}


@router.get("/pipelines/{pipeline_key}/readme")
def get_pipeline_readme(pipeline_key: str):
    readme_path = BASE_DIR / "nexus_ml" / "src" / pipeline_key / "README.md"
    if not readme_path.exists():
        readme_path = BASE_DIR / "src" / pipeline_key / "README.md"
    if not readme_path.exists():
        raise HTTPException(status_code=404, detail=f"README for pipeline '{pipeline_key}' not found.")
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return {"key": pipeline_key, "readme": content}
