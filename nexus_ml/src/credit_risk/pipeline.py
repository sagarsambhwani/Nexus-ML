import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from nexus_ml.config import RANDOM_SEED
from nexus_ml.src.common.base_model import BasePipeline

class CreditRiskPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Credit Risk Prediction", artifact_name="credit_risk_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        credit_score = np.random.randint(300, 850, size=n_samples)
        annual_income = np.random.uniform(20000, 180000, size=n_samples)
        dti_ratio = np.random.uniform(0.05, 0.60, size=n_samples)
        loan_amount = np.random.uniform(2000, 50000, size=n_samples)
        delinquencies_2yr = np.random.poisson(lam=0.4, size=n_samples)
        employment_years = np.random.randint(0, 25, size=n_samples)

        logit = (
            -0.012 * (credit_score - 600) +
            -0.00002 * (annual_income - 50000) +
            4.5 * dti_ratio +
            0.00005 * loan_amount +
            0.8 * delinquencies_2yr +
            -0.08 * employment_years +
            np.random.normal(0, 0.5, size=n_samples)
        )
        prob_default = 1 / (1 + np.exp(-logit))
        is_default = (np.random.uniform(0, 1, size=n_samples) < prob_default).astype(int)

        return pd.DataFrame({
            "credit_score": credit_score, "annual_income": annual_income,
            "dti_ratio": dti_ratio, "loan_amount": loan_amount,
            "delinquencies_2yr": delinquencies_2yr, "employment_years": employment_years,
            "is_default": is_default
        })

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df.drop(columns=["is_default"])
        y = df["is_default"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        auc = float(roc_auc_score(y_test, y_prob))
        acc = float(accuracy_score(y_test, y_pred))

        self.model = {
            "scaler": scaler,
            "classifier": model,
            "feature_names": list(X.columns),
            "metrics": {"roc_auc": round(auc, 4), "accuracy": round(acc, 4)}
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

        if prob < 0.10:
            tier, decision = "AAA (Prime)", "APPROVED"
        elif prob < 0.25:
            tier, decision = "AA (Near Prime)", "APPROVED"
        elif prob < 0.45:
            tier, decision = "B (Subprime)", "MANUAL_REVIEW"
        else:
            tier, decision = "CCC (High Risk)", "REJECTED"

        return {
            "default_probability": round(prob, 4),
            "risk_tier": tier,
            "underwriting_decision": decision,
            "credit_score_provided": input_data.get("credit_score", 0)
        }
