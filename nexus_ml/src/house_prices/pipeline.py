import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from nexus_ml.config import RANDOM_SEED
from nexus_ml.src.common.base_model import BasePipeline

class HousePricePipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="House Price Prediction", artifact_name="house_price_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        sqft = np.random.uniform(600, 4500, size=n_samples)
        bedrooms = np.random.randint(1, 6, size=n_samples)
        bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], size=n_samples)
        location_score = np.random.uniform(1.0, 10.0, size=n_samples)
        house_age = np.random.randint(0, 50, size=n_samples)
        garage_cars = np.random.randint(0, 4, size=n_samples)
        dist_city_km = np.random.uniform(1.0, 30.0, size=n_samples)

        base_price = (
            250 * sqft +
            15000 * bedrooms +
            22000 * bathrooms +
            35000 * location_score +
            -1800 * house_age +
            12000 * garage_cars +
            -2500 * dist_city_km +
            np.random.normal(0, 35000, size=n_samples)
        )
        price = np.maximum(80000, base_price)

        return pd.DataFrame({
            "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "location_score": location_score, "house_age": house_age,
            "garage_cars": garage_cars, "dist_city_km": dist_city_km, "price": price
        })

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df.drop(columns=["price"])
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        self.model = {
            "regressor": model,
            "feature_names": list(X.columns),
            "metrics": {"rmse": round(rmse, 2), "r2_score": round(r2, 4)}
        }
        self.save()
        return self.model["metrics"]

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self.load()

        feature_names = self.model["feature_names"]
        df_input = pd.DataFrame([input_data])[feature_names]

        model_reg = self.model["regressor"]
        predicted_price = float(model_reg.predict(df_input)[0])

        sqft = input_data.get("sqft", 1000)
        price_per_sqft = predicted_price / sqft if sqft > 0 else 0

        val_lower = predicted_price * 0.93
        val_upper = predicted_price * 1.07

        return {
            "predicted_price": round(predicted_price, 2),
            "price_per_sqft": round(price_per_sqft, 2),
            "valuation_range": {
                "lower_bound": round(val_lower, 2),
                "upper_bound": round(val_upper, 2)
            }
        }
