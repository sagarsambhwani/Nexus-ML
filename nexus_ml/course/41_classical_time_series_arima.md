# 📊 Chapter 41: Classical Time-Series Decomposition & ARIMA Modeling

## 41.1 Time-Series Component Decomposition
An observed time series $Y_t$ decomposes into trend ($T_t$), seasonality ($S_t$), cyclic variation ($C_t$), and irregular noise ($\epsilon_t$).

```
Additive Model:      Y_t = T_t + S_t + C_t + ε_t
Multiplicative Model: Y_t = T_t * S_t * C_t * ε_t
```

---

## 41.2 Stationarity & Augmented Dickey-Fuller (ADF) Test
A series is stationary if mean $\mu$ and variance $\sigma^2$ are constant over time. The **ADF Test** evaluates unit root presence ($H_0$: Non-stationary). If $p$-value $< 0.05$, differencing $\Delta Y_t = Y_t - Y_{t-1}$ achieves stationarity.

---

## 41.3 Box-Jenkins ARIMA Methodology
ARIMA($p, d, q$) fits Auto-Regressive ($p$), Differencing ($d$), and Moving Average ($q$) orders guided by Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots.

---

## ⚓ Repository Code Reference
- See [`src/demand_forecasting/pipeline.py`](file:///e:/Downloads/ML_only/src/demand_forecasting/pipeline.py) for time-series forecasting.
