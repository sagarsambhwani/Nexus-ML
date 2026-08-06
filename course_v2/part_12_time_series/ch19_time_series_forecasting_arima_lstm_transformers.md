# Chapter 19: Time Series Forecasting and Temporal Deep Learning

---

## 1. Big Picture

Time series forecasting predicts future sequence values based on historical observations. Unlike standard tabular ML where samples are assumed i.i.d. (independent and identically distributed), time series data has strong **temporal dependencies, trend, seasonality, autocorrelation, and non-stationarity**.

This chapter covers stationarity tests, ARIMA/SARIMAX, Prophet, feature engineering for time series (lags, rolling stats), and temporal deep learning models (LSTM, Temporal Fusion Transformers - TFT).

---

## 2. Intuition

- **Stationarity**: A time series is stationary if its mean, variance, and autocorrelation structure are constant over time. Non-stationary data (e.g. stock prices with upward trend) must be differenced ($Y_t - Y_{t-1}$) to make statistical modeling reliable.
- **Autocorrelation**: The degree to which today's value $Y_t$ depends on yesterday's value $Y_{t-1}$ or last week's value $Y_{t-7}$.

---

## 3. Visualization

```text
Time Series Decomposition:

  Original Series Y(t) ──► [ Trend Component T(t) ]  (Long-term direction)
                       ──► [ Seasonal Component S(t) ] (Repeating daily/weekly patterns)
                       ──► [ Residual Noise R(t) ]     (Random stationary noise)
```

---

## 4. Mathematics

### 1. ARIMA(p, d, q) Model Equation
For stationary differenced series $y'_t = \Delta^d y_t$:

$$y'_t = c + \sum_{i=1}^p \phi_i y'_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$

Where:
- $p$: Autoregressive (AR) order.
- $d$: Integrated (Differencing) degree.
- $q$: Moving Average (MA) noise order.

### 2. Augmented Dickey-Fuller (ADF) Stationarity Test
Null Hypothesis $H_0$: Unit root exists ($\gamma = 0$, non-stationary).
Test regression: $\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum \delta_i \Delta y_{t-i} + \epsilon_t$. If $p < 0.05$, reject $H_0$ (Series is stationary).

---

## 5. Python (From Scratch Time-Series Lag & Rolling Feature Engine)

```python
import numpy as np
import pandas as pd

def create_time_series_features(df: pd.DataFrame, target_col: str, lags=[1, 7, 14], windows=[7, 30]) -> pd.DataFrame:
    res = df.copy()
    
    # 1. Lag Features (Historical Past Values)
    for lag in lags:
        res[f'{target_col}_lag_{lag}'] = res[target_col].shift(lag)
        
    # 2. Rolling Window Statistics
    for w in windows:
        res[f'{target_col}_roll_mean_{w}'] = res[target_col].shift(1).rolling(window=w).mean()
        res[f'{target_col}_roll_std_{w}'] = res[target_col].shift(1).rolling(window=w).std()
        
    # 3. Differencing Features (Momentum)
    res[f'{target_col}_diff_1'] = res[target_col] - res[target_col].shift(1)
    
    return res

# Test Transformation
dates = pd.date_range("2025-01-01", periods=50, freq="D")
sales = np.sin(np.linspace(0, 10, 50)) * 10 + 50 + np.random.normal(0, 1, 50)
df_sales = pd.DataFrame({"date": dates, "sales": sales})

df_feat = create_time_series_features(df_sales, "sales")
print("Engineered Time Series Features (Last 5 Rows):")
print(df_feat[['sales', 'sales_lag_1', 'sales_roll_mean_7', 'sales_diff_1']].tail())
```

---

## 6. Production Library (Statsmodels ARIMA & Prophet)

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 1. Test Stationarity
adf_res = adfuller(df_sales['sales'])
print(f"ADF Statistic: {adf_res[0]:.4f}, p-value: {adf_res[1]:.4f}")

# 2. Fit Production ARIMA(1, 1, 1) Model
model = ARIMA(df_sales['sales'], order=(1, 1, 1))
fit_model = model.fit()
forecast = fit_model.forecast(steps=7)
print("7-Day Ahead Forecast Values:")
print(forecast.values)
```

---

## 7. Under the Hood

- Note the critical line: `res[target_col].shift(1).rolling(window=w).mean()`. We apply `.shift(1)` *before* computing rolling statistics to prevent **Lookahead Bias**! Omitting `.shift(1)` incorporates today's actual target value into today's prediction feature.

---

## 8. Engineering Perspective

- **Validation Strategy**: Never use standard $K$-Fold Cross Validation on time series! Standard CV randomly shuffles data, training on future data to predict past data. Use **Time Series Split (Rolling Window CV / Expanding Window CV)**.

---

## 9. Common Mistakes

1. **Lookahead Leakage**: Computing rolling means without shifting, leaking future actual values into training feature columns.
2. **Ignoring Stationarity**: Fitting linear autoregressive models directly onto un-differenced trending time series.

---

## 10. Interview Questions

### Q1: How do you evaluate time series models properly without data leakage?
**Answer**: Use TimeSeriesSplit (Walk-Forward Validation / Rolling Window CV). Train on historical window $[0, T_1]$ and test on $[T_1+1, T_1+k]$. Then expand or slide the training window to $[0, T_2]$ and test on $[T_2+1, T_2+k]$. Never shuffle time series samples.

---

## 11. Exercises

1. **Math**: Prove that an AR(1) process $Y_t = \phi Y_{t-1} + \epsilon_t$ is stationary if and only if $|\phi| < 1$.
2. **Coding**: Implement a Temporal Fusion Transformer (TFT) dataset loader in PyTorch.

---

## 12. Mini Project: Automated Demand Forecasting System

Write a script `demand_forecaster.py` using Polars and LightGBM that computes rolling features, performs Walk-Forward Validation, and outputs 30-day demand predictions with confidence intervals.

---

## 13. Capstone Integration

Implemented in `src/demand_forecasting/pipeline.py`.
