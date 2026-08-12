# ⏳ Chapter 42: Deep Time-Series Forecasting (TFT & N-BEATS)

## 42.1 Deep Learning vs. Classical Forecasting
Classical ARIMA models struggle with multi-variate exogenous features, high-cardinality entity groupings, and non-linear cross-series dependencies. Deep sequential models overcome these limitations.

---

## 42.2 Temporal Fusion Transformer (TFT)
TFT (Lim et al. 2021) integrates:
1. **Variable Selection Networks (VSN)**: Dynamically selects relevant static metadata and time-varying features.
2. **Gated Residual Networks (GRN)**: Provides adaptive non-linear processing with skip connections.
3. **Multi-Head Attention**: Captures long-range temporal dependencies and seasonal patterns across time steps.

---

## ⚓ Repository Code Reference
- See [`src/demand_forecasting/pipeline.py`](file:///e:/Downloads/ML_only/src/demand_forecasting/pipeline.py) for multi-step lag forecasting.
