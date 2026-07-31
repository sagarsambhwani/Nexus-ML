# 📊 Demand Forecasting (Time-Series Sales Forecasting System)

## 📌 System Overview
The **Demand Forecasting System** predicts multi-day future retail demand for inventory optimization and supply chain planning. Accurate time-series forecasting prevents costly out-of-stock events while minimizing excess warehouse holding costs.

This pipeline implements an autoregressive **Lag-Feature Gradient Boosting Model** incorporating calendar seasonality (day of week, month, weekend flags) and promotional events to generate multi-step iterative forecasts.

---

## 🏗️ Deep-Dive Implementation Architecture

Implemented in [`pipeline.py`](file:///e:/Downloads/Nexus-ML/src/demand_forecasting/pipeline.py) as a subclass of `BasePipeline`:

```
Time-Series Sales Stream -> Autoregressive Feature Extraction (Lags & Moving Averages) -> Gradient Boosting Regressor -> Recursive Multi-Step Forecast Horizon
```

### Class Code Structure & Execution Flow:

```python
class DemandForecastingPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Demand Forecasting", artifact_name="demand_forecasting_model.joblib")
```

1. **Data Generation (`generate_data`)**:
   - Synthesizes 500 days of daily sales records incorporating yearly sine-wave seasonality, weekend demand lifts (+25 units), promotional spikes (+40 units), and upward macro trends.
   - Constructs autoregressive lag features:
     - `lag_1`: Sales volume on day $t-1$
     - `lag_7`: Sales volume on day $t-7$ (weekly seasonal cycle)
     - `rolling_mean_7`: 7-day moving average

2. **Model Training & Artifact Serialization (`train`)**:
   - Performs chronological time-series train/test split (80/20, `shuffle=False`) to avoid future data leakage.
   - Fits `GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)`.
   - Computes MAE (~14.5 units) and $R^2$ score (~0.27).
   - Serializes `demand_forecasting_model.joblib`.

3. **Multi-Step Recursive Inference (`predict`)**:
   - Iterates through target forecast horizon days ($1 \dots N$):
     - Predicts sales $\hat{y}_t$ for current day.
     - Updates autoregressive lag buffers dynamically ($\text{lag}_1 \leftarrow \hat{y}_t$, updates rolling mean).
   - Computes total predicted period demand and confidence bounds ($\pm 12\%$).

---

## 🎯 Engineering Decision-Making Process

### 1. Algorithm Selection Rationale: Why Supervised Lag-Regression over ARIMA or Prophet?
- **Decision**: Selected **Autoregressive Lag-Feature Gradient Boosting**.
- **Rationale**:
  - *Vs. Classical ARIMA*: Classical ARIMA cannot easily incorporate external promotional flags (`is_promo`) or complex non-linear calendar interactions.
  - *Vs. Prophet*: Standard regression models allow direct joint training across multiple store locations with exogenous pricing features while maintaining lightweight dependency footprints.

### 2. Hyperparameter Choices & Design Trade-Offs:
| Hyperparameter | Value Chosen | Engineering Rationale & Trade-Off |
|---|---|---|
| `shuffle=False` | Enabled | Mandatory for time-series evaluation to ensure models are validated on strict future time windows. |
| `n_estimators` | `100` | Fits trend and seasonal residual steps without overfitting noise spikes. |

---

## 📊 Autoregressive Feature Engineering Schema

| Feature Name | Type | Mathematical Definition | Purpose |
|---|---|---|---|
| `day_of_week` | Integer | $t \bmod 7 \in [0, 6]$ | Captures weekly cyclical patterns |
| `month` | Integer | $t_{\text{month}} \in [1, 12]$ | Captures monthly/seasonal demand shifts |
| `is_weekend` | Binary | $\mathbb{I}(\text{day\_of\_week} \ge 5)$ | Captures weekend shopping surges |
| `is_promo` | Binary | $\{0, 1\}$ | Models exogenous marketing campaigns |
| `lag_1` | Float | $y_{t-1}$ | Short-term momentum factor |
| `lag_7` | Float | $y_{t-7}$ | Same-day-last-week seasonal anchor |
| `rolling_mean_7` | Float | $\frac{1}{7}\sum_{i=1}^7 y_{t-i}$ | Smoothed local trend baseline |

---

## 🏋️ Evaluation Metrics & Benchmarks

- **Mean Absolute Error (MAE)**: ~14.49 units (Average daily unit error).
- **$R^2$ Score**: ~0.27 (Captured seasonal and promotional variance).

---

## 🚀 Production Deployment Strategy

1. **ERP / WMS Integration**: Connect forecast outputs directly into Enterprise Resource Planning (ERP) databases for automated purchase order drafting.
2. **Automated Retraining**: Retrain models weekly on fresh point-of-sale (POS) transactional data.
