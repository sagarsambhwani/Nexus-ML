import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score
from config import RANDOM_SEED
from src.common.base_model import BasePipeline

class PredictiveMaintenancePipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Predictive Maintenance", artifact_name="predictive_maintenance_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        vibration_hz = np.random.uniform(10.0, 90.0, size=n_samples)
        temperature_c = np.random.uniform(40.0, 110.0, size=n_samples)
        pressure_psi = np.random.uniform(20.0, 100.0, size=n_samples)
        rpm = np.random.uniform(800.0, 3500.0, size=n_samples)
        sensor_noise_std = np.random.uniform(0.1, 5.0, size=n_samples)
        operating_hours = np.random.uniform(100.0, 10000.0, size=n_samples)

        failure_score = (
            0.05 * (vibration_hz - 40) +
            0.08 * (temperature_c - 70) +
            0.04 * (pressure_psi - 50) +
            0.8 * sensor_noise_std +
            0.0003 * operating_hours +
            np.random.normal(0, 1.0, size=n_samples)
        )
        is_failure = (failure_score > 3.2).astype(int)

        return pd.DataFrame({
            "vibration_hz": vibration_hz, "temperature_c": temperature_c,
            "pressure_psi": pressure_psi, "rpm": rpm,
            "sensor_noise_std": sensor_noise_std, "operating_hours": operating_hours,
            "is_failure": is_failure
        })

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df.drop(columns=["is_failure"])
        y = df["is_failure"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

        model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = float(roc_auc_score(y_test, y_prob))
        recall = float(recall_score(y_test, y_pred))

        self.model = {
            "classifier": model,
            "feature_names": list(X.columns),
            "metrics": {"roc_auc": round(auc, 4), "recall": round(recall, 4)}
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

        est_rul_hours = int(round(max(10, (1.0 - prob) * 1200)))

        if prob >= 0.65:
            status = "CRITICAL"
            recommendation = "Immediate shutdown required. Schedule technician for component overhaul."
        elif prob >= 0.35:
            status = "WARNING"
            recommendation = "Schedule preventative maintenance within 48 hours. Inspect thermal & vibration dampers."
        else:
            status = "HEALTHY"
            recommendation = "System operating within optimal parameters. Next routine check standard schedule."

        return {
            "failure_probability": round(prob, 4),
            "health_status": status,
            "estimated_rul_hours": est_rul_hours,
            "recommended_action": recommendation
        }
