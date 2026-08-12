import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from config import RANDOM_SEED
from src.common.base_model import BasePipeline

class DemandForecastingPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Demand Forecasting", artifact_name="demand_forecasting_model.joblib")

    def generate_data(self, n_days: int = 500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        start_date = datetime(2025, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]

        records = []
        for d in dates:
            day_of_week = d.weekday()
            month = d.month
            is_weekend = 1 if day_of_week >= 5 else 0
            is_promo = np.random.choice([0, 1], p=[0.85, 0.15])

            base_demand = (
                100 +
                15 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365.0) +
                25 * is_weekend +
                40 * is_promo +
                0.05 * (d - start_date).days +
                np.random.normal(0, 10)
            )
            units_sold = max(10, int(round(base_demand)))

            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "is_promo": is_promo,
                "day_of_year": d.timetuple().tm_yday,
                "units_sold": units_sold
            })

        df = pd.DataFrame(records)
        df["lag_1"] = df["units_sold"].shift(1).fillna(df["units_sold"].mean())
        df["lag_7"] = df["units_sold"].shift(7).fillna(df["units_sold"].mean())
        df["rolling_mean_7"] = df["units_sold"].shift(1).rolling(7).mean().fillna(df["units_sold"].mean())

        return df

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        feature_cols = ["day_of_week", "month", "is_weekend", "is_promo", "day_of_year", "lag_1", "lag_7", "rolling_mean_7"]
        
        X = df[feature_cols]
        y = df["units_sold"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False)

        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        self.model = {
            "regressor": model,
            "feature_names": feature_cols,
            "last_units_mean": float(df["units_sold"].mean()),
            "metrics": {"mae": round(mae, 2), "r2_score": round(r2, 4)}
        }
        self.save()
        return self.model["metrics"]

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self.load()

        horizon_days = input_data.get("horizon_days", 7)
        is_promo = input_data.get("is_promo", 0)
        store_id = input_data.get("store_id", "STORE_101")

        model_reg = self.model["regressor"]
        feature_cols = self.model["feature_names"]

        start_date = datetime.now()
        forecasts = []
        
        running_lag1 = self.model["last_units_mean"]
        running_lag7 = self.model["last_units_mean"]
        running_roll7 = self.model["last_units_mean"]

        total_forecast_units = 0

        for i in range(horizon_days):
            current_date = start_date + timedelta(days=i)
            dow = current_date.weekday()
            month = current_date.month
            is_weekend = 1 if dow >= 5 else 0
            doy = current_date.timetuple().tm_yday

            input_dict = {
                "day_of_week": dow, "month": month, "is_weekend": is_weekend, "is_promo": is_promo,
                "day_of_year": doy, "lag_1": running_lag1, "lag_7": running_lag7, "rolling_mean_7": running_roll7
            }
            df_curr = pd.DataFrame([input_dict])[feature_cols]
            pred_units = float(model_reg.predict(df_curr)[0])
            pred_units = max(5.0, pred_units)

            total_forecast_units += pred_units

            forecasts.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day_name": current_date.strftime("%A"),
                "forecasted_units": round(pred_units, 1),
                "lower_bound": round(pred_units * 0.88, 1),
                "upper_bound": round(pred_units * 1.12, 1)
            })

            running_lag1 = pred_units
            running_roll7 = (running_roll7 * 6 + pred_units) / 7.0

        return {
            "store_id": store_id,
            "forecast_horizon_days": horizon_days,
            "total_predicted_units": round(total_forecast_units, 1),
            "daily_forecast": forecasts
        }
