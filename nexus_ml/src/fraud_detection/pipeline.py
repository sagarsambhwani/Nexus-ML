import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from config import RANDOM_SEED
from src.common.base_model import BasePipeline

class FraudDetectionPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Fraud Detection", artifact_name="fraud_detection_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)
        
        amount = np.random.exponential(scale=100, size=n_samples)
        time_hour = np.random.randint(0, 24, size=n_samples)
        velocity_1h = np.random.poisson(lam=2, size=n_samples)
        location_risk = np.random.uniform(0, 1, size=n_samples)
        
        v1 = np.random.normal(0, 1, size=n_samples)
        v2 = np.random.normal(0, 1, size=n_samples)
        v3 = np.random.normal(0, 1, size=n_samples)
        v4 = np.random.normal(0, 1, size=n_samples)
        v5 = np.random.normal(0, 1, size=n_samples)

        fraud_score = (
            0.015 * amount +
            1.2 * velocity_1h +
            2.5 * location_risk +
            0.8 * (v1 > 1.5).astype(int) +
            1.0 * (time_hour < 4).astype(int) +
            np.random.normal(0, 0.5, size=n_samples)
        )
        is_fraud = (fraud_score > np.percentile(fraud_score, 88)).astype(int)

        return pd.DataFrame({
            "amount": amount, "time_hour": time_hour, "velocity_1h": velocity_1h,
            "location_risk": location_risk, "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5,
            "is_fraud": is_fraud
        })

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df.drop(columns=["is_fraud"])
        y = df["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = float(roc_auc_score(y_test, y_prob))
        f1 = float(f1_score(y_test, y_pred))

        self.model = {
            "classifier": model,
            "feature_names": list(X.columns),
            "metrics": {"roc_auc": round(auc, 4), "f1_score": round(f1, 4)}
        }
        self.save()
        return self.model["metrics"]

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self.load()

        feature_names = self.model["feature_names"]
        df_input = pd.DataFrame([input_data])[feature_names]

        model_cls = self.model["classifier"]
        prob = float(model_cls.predict_proba(df_input)[0, 1])
        risk_score = int(round(prob * 100))
        
        status = "HIGH_RISK" if prob >= 0.5 else ("MEDIUM_RISK" if prob >= 0.25 else "LOW_RISK")

        importances = model_cls.feature_importances_
        top_indices = np.argsort(importances)[::-1][:3]
        top_factors = [feature_names[i] for i in top_indices]

        return {
            "fraud_probability": round(prob, 4),
            "risk_score": risk_score,
            "alert_status": status,
            "top_risk_factors": top_factors
        }
