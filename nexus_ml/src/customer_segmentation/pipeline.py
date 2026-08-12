import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import RANDOM_SEED
from src.common.base_model import BasePipeline

class CustomerSegmentationPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Customer Segmentation", artifact_name="customer_segmentation_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        n0 = n_samples // 4
        inc0 = np.random.normal(110, 15, n0)
        spend0 = np.random.normal(85, 8, n0)
        freq0 = np.random.normal(24, 4, n0)
        rec0 = np.random.normal(12, 5, n0)

        n1 = n_samples // 4
        inc1 = np.random.normal(105, 14, n1)
        spend1 = np.random.normal(25, 10, n1)
        freq1 = np.random.normal(8, 3, n1)
        rec1 = np.random.normal(45, 12, n1)

        n2 = n_samples // 4
        inc2 = np.random.normal(35, 10, n2)
        spend2 = np.random.normal(78, 10, n2)
        freq2 = np.random.normal(18, 5, n2)
        rec2 = np.random.normal(20, 8, n2)

        n3 = n_samples - (n0 + n1 + n2)
        inc3 = np.random.normal(30, 8, n3)
        spend3 = np.random.normal(20, 9, n3)
        freq3 = np.random.normal(4, 2, n3)
        rec3 = np.random.normal(90, 20, n3)

        income = np.clip(np.concatenate([inc0, inc1, inc2, inc3]), 15, 160)
        spending = np.clip(np.concatenate([spend0, spend1, spend2, spend3]), 1, 100)
        frequency = np.clip(np.concatenate([freq0, freq1, freq2, freq3]), 1, 50)
        recency = np.clip(np.concatenate([rec0, rec1, rec2, rec3]), 1, 180)

        return pd.DataFrame({
            "annual_income_k": income, "spending_score": spending,
            "frequency_purchases": frequency, "recency_days": recency
        })

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        feature_cols = list(df.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        pca.fit(X_scaled)

        centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_personas = {}
        for idx, cent in enumerate(centroids_unscaled):
            inc, spend = cent[0], cent[1]
            if inc > 70 and spend > 50:
                name = "VIP High Rollers"
                strategy = "Exclusive preview invites, concierge service, premium loyalty rewards"
            elif inc > 70 and spend <= 50:
                name = "Selective Wealth Savers"
                strategy = "Value proposition messaging, high-margin quality focus, targeted newsletters"
            elif inc <= 70 and spend > 50:
                name = "Trend Seekers & Impulse Shoppers"
                strategy = "Flash sales, influencer collaborations, social proof & trending product alerts"
            else:
                name = "Budget Conscious & Occasional"
                strategy = "Win-back discount coupons, low-cost essentials bundle, re-engagement emails"
            
            cluster_personas[idx] = {"name": name, "marketing_strategy": strategy}

        inertia = float(kmeans.inertia_)

        self.model = {
            "scaler": scaler,
            "kmeans": kmeans,
            "pca": pca,
            "feature_names": feature_cols,
            "cluster_personas": cluster_personas,
            "metrics": {"inertia": round(inertia, 2), "n_clusters": 4}
        }
        self.save()
        return self.model["metrics"]

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self.load()

        feature_names = self.model["feature_names"]
        df_input = pd.DataFrame([input_data])[feature_names]

        scaler = self.model["scaler"]
        kmeans = self.model["kmeans"]
        pca = self.model["pca"]

        X_scaled = scaler.transform(df_input)
        cluster_id = int(kmeans.predict(X_scaled)[0])

        pca_coords = pca.transform(X_scaled)[0]
        persona_info = self.model["cluster_personas"].get(cluster_id, {
            "name": f"Cluster #{cluster_id}",
            "marketing_strategy": "General personalized recommendation"
        })

        return {
            "cluster_id": cluster_id,
            "persona_name": persona_info["name"],
            "marketing_strategy": persona_info["marketing_strategy"],
            "pca_coordinates": {
                "pc1": round(float(pca_coords[0]), 3),
                "pc2": round(float(pca_coords[1]), 3)
            }
        }
