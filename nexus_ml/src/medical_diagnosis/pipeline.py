import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from nexus_ml.config import RANDOM_SEED
from nexus_ml.src.common.base_model import BasePipeline

class MedicalDiagnosisPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Medical Diagnosis Support", artifact_name="medical_diagnosis_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        age = np.random.randint(18, 85, size=n_samples)
        glucose = np.random.uniform(70.0, 240.0, size=n_samples)
        blood_pressure = np.random.uniform(60.0, 160.0, size=n_samples)
        bmi = np.random.uniform(18.0, 45.0, size=n_samples)
        hba1c = np.random.uniform(4.5, 11.0, size=n_samples)
        family_history = np.random.choice([0, 1], p=[0.65, 0.35], size=n_samples)
        smoker = np.random.choice([0, 1], p=[0.75, 0.25], size=n_samples)

        logit = (
            0.03 * (age - 45) +
            0.025 * (glucose - 100) +
            0.015 * (blood_pressure - 80) +
            0.08 * (bmi - 25) +
            0.6 * (hba1c - 5.7) +
            0.8 * family_history +
            0.5 * smoker +
            np.random.normal(0, 0.4, size=n_samples)
        )
        prob = 1 / (1 + np.exp(-logit))
        disease_positive = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)

        return pd.DataFrame({
            "age": age, "glucose": glucose, "blood_pressure": blood_pressure,
            "bmi": bmi, "hba1c": hba1c, "family_history": family_history, "smoker": smoker,
            "disease_positive": disease_positive
        })

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df.drop(columns=["disease_positive"])
        y = df["disease_positive"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        auc = float(roc_auc_score(y_test, y_prob))
        f1 = float(f1_score(y_test, y_pred))

        self.model = {
            "scaler": scaler,
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

        scaler = self.model["scaler"]
        model_cls = self.model["classifier"]

        X_scaled = scaler.transform(df_input)
        prob = float(model_cls.predict_proba(X_scaled)[0, 1])

        if prob >= 0.60:
            risk_category = "HIGH_RISK"
            clinical_guidance = "Comprehensive diagnostic panel recommended. Consult endocrinologist for confirmation."
        elif prob >= 0.30:
            risk_category = "ELEVATED_RISK"
            clinical_guidance = "Lifestyle modifications advised (diet & exercise). Repeat HbA1c screening in 3 months."
        else:
            risk_category = "LOW_RISK"
            clinical_guidance = "Biomarkers within normal limits. Routine annual wellness checkup."

        elevated_biomarkers = []
        if input_data.get("glucose", 0) > 125:
            elevated_biomarkers.append("Fasting Glucose (>125 mg/dL)")
        if input_data.get("hba1c", 0) > 6.4:
            elevated_biomarkers.append("HbA1c (>6.4%)")
        if input_data.get("bmi", 0) > 30:
            elevated_biomarkers.append("BMI (>30 Obesity Class I)")

        return {
            "disease_risk_probability": round(prob, 4),
            "risk_category": risk_category,
            "elevated_biomarkers": elevated_biomarkers,
            "clinical_guidance": clinical_guidance
        }
