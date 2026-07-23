from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from api.schemas import (
    FraudDetectionRequest, CreditRiskRequest, CustomerChurnRequest,
    HousePriceRequest, RecommendationRequest, DemandForecastingRequest,
    PredictiveMaintenanceRequest, MedicalDiagnosisRequest, SentimentAnalysisRequest,
    DocumentClassificationRequest, DefectDetectionRequest, CustomerSegmentationRequest,
    StandardResponse
)
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

router = APIRouter(prefix="/api/v1", tags=["ML Inference Endpoints"])

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
    return StandardResponse(pipeline="Fraud Detection", result=res)

@router.post("/predict/credit-risk", response_model=StandardResponse)
def predict_credit_risk(req: CreditRiskRequest):
    pipe = get_pipeline("credit_risk")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Credit Risk Prediction", result=res)

@router.post("/predict/customer-churn", response_model=StandardResponse)
def predict_churn(req: CustomerChurnRequest):
    pipe = get_pipeline("customer_churn")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Customer Churn Prediction", result=res)

@router.post("/predict/house-prices", response_model=StandardResponse)
def predict_house_prices(req: HousePriceRequest):
    pipe = get_pipeline("house_prices")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="House Price Prediction", result=res)

@router.post("/predict/recommendation", response_model=StandardResponse)
def predict_recommendation(req: RecommendationRequest):
    pipe = get_pipeline("recommendation")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Recommendation System", result=res)

@router.post("/predict/demand-forecasting", response_model=StandardResponse)
def predict_demand(req: DemandForecastingRequest):
    pipe = get_pipeline("demand_forecasting")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Demand Forecasting", result=res)

@router.post("/predict/predictive-maintenance", response_model=StandardResponse)
def predict_maintenance(req: PredictiveMaintenanceRequest):
    pipe = get_pipeline("predictive_maintenance")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Predictive Maintenance", result=res)

@router.post("/predict/medical-diagnosis", response_model=StandardResponse)
def predict_medical(req: MedicalDiagnosisRequest):
    pipe = get_pipeline("medical_diagnosis")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Medical Diagnosis Support", result=res)

@router.post("/predict/sentiment-analysis", response_model=StandardResponse)
def predict_sentiment(req: SentimentAnalysisRequest):
    pipe = get_pipeline("sentiment_analysis")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Sentiment Analysis", result=res)

@router.post("/predict/document-classification", response_model=StandardResponse)
def predict_document(req: DocumentClassificationRequest):
    pipe = get_pipeline("document_classification")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Document Classification", result=res)

@router.post("/predict/defect-detection", response_model=StandardResponse)
def predict_defect(req: DefectDetectionRequest):
    pipe = get_pipeline("defect_detection")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Defect Detection", result=res)

@router.post("/predict/customer-segmentation", response_model=StandardResponse)
def predict_segmentation(req: CustomerSegmentationRequest):
    pipe = get_pipeline("customer_segmentation")
    res = pipe.predict(req.model_dump())
    return StandardResponse(pipeline="Customer Segmentation", result=res)

@router.get("/pipelines/status")
def list_pipelines():
    status_list = []
    pipeline_names = [
        ("fraud_detection", "Fraud Detection", "Classification"),
        ("credit_risk", "Credit Risk Prediction", "Classification"),
        ("customer_churn", "Customer Churn Prediction", "Classification"),
        ("house_prices", "House Price Prediction", "Regression"),
        ("recommendation", "Recommendation System", "Hybrid Recommendation"),
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
