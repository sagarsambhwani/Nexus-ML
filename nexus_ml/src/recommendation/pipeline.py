import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.decomposition import TruncatedSVD
from config import RANDOM_SEED
from src.common.base_model import BasePipeline

class RecommendationPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Recommendation System", artifact_name="recommendation_model.joblib")

    def generate_data(self, n_users: int = 200, n_items: int = 50) -> Dict[str, pd.DataFrame]:
        np.random.seed(RANDOM_SEED)

        categories = ["Electronics", "Books", "Fashion", "Home & Kitchen", "Sports"]
        items = []
        for i in range(1, n_items + 1):
            items.append({
                "item_id": f"ITEM_{i:03d}",
                "title": f"Product Item #{i}",
                "category": np.random.choice(categories),
                "price": float(np.random.uniform(10, 300))
            })
        df_items = pd.DataFrame(items)

        ratings = []
        for u in range(1, n_users + 1):
            user_id = f"USER_{u:03d}"
            rated_item_ids = np.random.choice(df_items["item_id"], size=np.random.randint(8, 16), replace=False)
            for item_id in rated_item_ids:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.4, 0.25])
                ratings.append({"user_id": user_id, "item_id": item_id, "rating": rating})
        df_ratings = pd.DataFrame(ratings)

        return {"items": df_items, "ratings": df_ratings}

    def train(self) -> Dict[str, Any]:
        data = self.generate_data()
        df_items = data["items"]
        df_ratings = data["ratings"]

        user_item_matrix = df_ratings.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

        svd = TruncatedSVD(n_components=12, random_state=RANDOM_SEED)
        user_factors = svd.fit_transform(user_item_matrix)
        item_factors = svd.components_.T

        reconstructed_ratings = np.dot(user_factors, item_factors.T)
        df_reconstructed = pd.DataFrame(
            reconstructed_ratings, 
            index=user_item_matrix.index, 
            columns=user_item_matrix.columns
        )

        explained_var = float(np.sum(svd.explained_variance_ratio_))

        self.model = {
            "user_item_matrix": user_item_matrix,
            "reconstructed_ratings": df_reconstructed,
            "df_items": df_items,
            "metrics": {"explained_variance": round(explained_var, 4)}
        }
        self.save()
        return self.model["metrics"]

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self.load()

        user_id = input_data.get("user_id", "USER_001")
        category_filter = input_data.get("category", "All")
        top_n = input_data.get("top_n", 5)

        df_items = self.model["df_items"]
        df_reconstructed = self.model["reconstructed_ratings"]
        user_item_matrix = self.model["user_item_matrix"]

        if user_id in df_reconstructed.index:
            user_scores = df_reconstructed.loc[user_id]
            already_rated = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
        else:
            user_scores = df_reconstructed.mean(axis=0)
            already_rated = set()

        recommendations = []
        for item_id, score in user_scores.sort_values(ascending=False).items():
            if item_id in already_rated:
                continue

            item_info = df_items[df_items["item_id"] == item_id].iloc[0]
            item_cat = item_info["category"]

            if category_filter != "All" and item_cat.lower() != category_filter.lower():
                continue

            match_score = float(np.clip(score / 5.0, 0.5, 0.99))

            recommendations.append({
                "item_id": item_id,
                "title": item_info["title"],
                "category": item_cat,
                "price": item_info["price"],
                "match_score": round(match_score, 2),
                "recommendation_reason": f"High affinity score for {item_cat} based on user preference profile"
            })

            if len(recommendations) >= top_n:
                break

        return {
            "user_id": user_id,
            "category_filter": category_filter,
            "recommendations": recommendations
        }
