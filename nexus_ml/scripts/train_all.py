import sys
import time
from pathlib import Path

# Add project root to sys.path
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

def train_all():
    print("=" * 70)
    print("      ENTERPRISE ML SUITE - AUTOMATED MODEL TRAINING PIPELINE")
    print("=" * 70)

    pipelines = [
        ("fraud_detection", FraudDetectionPipeline()),
        ("credit_risk", CreditRiskPipeline()),
        ("customer_churn", CustomerChurnPipeline()),
        ("house_prices", HousePricePipeline()),
        ("recommendation", RecommendationPipeline()),
        ("demand_forecasting", DemandForecastingPipeline()),
        ("predictive_maintenance", PredictiveMaintenancePipeline()),
        ("medical_diagnosis", MedicalDiagnosisPipeline()),
        ("sentiment_analysis", SentimentAnalysisPipeline()),
        ("document_classification", DocumentClassificationPipeline()),
        ("defect_detection", DefectDetectionPipeline()),
        ("customer_segmentation", CustomerSegmentationPipeline()),
    ]

    summary = []
    start_total = time.time()

    for idx, (name, pipeline) in enumerate(pipelines, 1):
        print(f"\n[{idx}/12] Training Pipeline: {pipeline.name} ({name})...")
        t0 = time.time()
        metrics = pipeline.train()
        elapsed = round(time.time() - t0, 2)
        print(f"     [OK] Model Artifact Saved: {pipeline.artifact_name}")
        print(f"     [OK] Metrics: {metrics} (took {elapsed}s)")
        summary.append({
            "id": name,
            "name": pipeline.name,
            "metrics": metrics,
            "time_sec": elapsed
        })

    total_time = round(time.time() - start_total, 2)
    print("\n" + "=" * 70)
    print(f"SUCCESS: All 12 ML models trained & serialized in {total_time} seconds!")
    print("=" * 70)

if __name__ == "__main__":
    train_all()
