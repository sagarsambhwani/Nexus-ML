import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from config import RANDOM_SEED
from src.common.base_model import BasePipeline

class CustomerChurnPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Customer Churn Prediction", artifact_name="customer_churn_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        tenure = np.random.randint(1, 72, size=n_samples)
        monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
        contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20])
        support_tickets = np.random.poisson(lam=1.5, size=n_samples)
        paperless_billing = np.random.choice([0, 1], size=n_samples)
        total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=n_samples)

        logit = (
            -0.04 * tenure +
            0.02 * monthly_charges +
            -1.2 * contract_type +
            0.5 * support_tickets +
            0.3 * paperless_billing +
            np.random.normal(0, 0.4, size=n_samples)
        )
        prob = 1 / (1 + np.exp(-logit))
        churn = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)

        return pd.DataFrame({
            "tenure": tenure, "monthly_charges": monthly_charges, "total_charges": total_charges,
            "contract_type": contract_type, "support_tickets": support_tickets, "paperless_billing": paperless_billing,
            "churn": churn
        })

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df.drop(columns=["churn"])
        y = df["churn"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

        model = GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        self.model = {
            "classifier": model,
            "feature_names": list(X.columns),
            "metrics": {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}
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

        risk_level = "HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.3 else "LOW")
        
        if prob > 0.5:
            if input_data.get("contract_type", 0) == 0:
                retention_action = "Offer 15% discount for 12-month contract lock-in"
            elif input_data.get("support_tickets", 0) >= 3:
                retention_action = "Assign dedicated customer success manager to resolve tickets"
            else:
                retention_action = "Send personalized customer appreciation gift & loyalty perk"
        else:
            retention_action = "Standard nurturing newsletter & upsell campaign"

        return {
            "churn_probability": round(prob, 4),
            "risk_level": risk_level,
            "recommended_action": retention_action
        }
