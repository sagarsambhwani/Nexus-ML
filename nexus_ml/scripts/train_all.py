import sys
import time
from pathlib import Path

# Add workspace root and nexus_ml root to sys.path
NEXUS_ML_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = NEXUS_ML_DIR.parent

for p in [WORKSPACE_ROOT, NEXUS_ML_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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
    print(f"Training Complete! Total Time: {total_time} seconds.")
    print("=" * 70)
    return summary

if __name__ == "__main__":
    train_all()
