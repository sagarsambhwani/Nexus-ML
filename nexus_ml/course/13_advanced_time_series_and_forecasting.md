# 📈 Chapter 13: Advanced Time-Series & Sequential Deep Learning

## 13.1 Stationarity & Statistical Fundamentals

A time series $\{Y_t\}$ is **weakly stationary** if its mean, variance, and autocovariance structure are constant over time:

$$\mathbb{E}[Y_t] = \mu, \quad \text{Var}(Y_t) = \sigma^2, \quad \text{Cov}(Y_t, Y_{t+k}) = \gamma(k)$$

```
        Non-Stationary Series (Trending)                  Stationary Series (Constant μ & σ)
                 ▲     ╱                                         ▲   ╭──╮  ╭──╮
                 │   ╱                                           │  ─┼──┼──┼──┼─ μ
                 │ ╱                                             │   ╰──╯  ╰──╯
                 └───────────────► Time                          └───────────────► Time
```

### 1. Augmented Dickey-Fuller (ADF) Test
Evaluates the presence of a unit root ($H_0$: Non-stationary series):

$$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^p \delta_i \Delta Y_{t-i} + \epsilon_t$$

If $p$-value $< 0.05$, reject $H_0$; the series is stationary.

### 2. Differencing
Converts non-stationary series to stationary by subtracting previous values:

$$\Delta Y_t = Y_t - Y_{t-1}$$

---

## 13.2 Classical Time-Series Models: ARIMA & SARIMAX

$$\text{ARIMA}(p, d, q)$$

- **$p$ (Auto-Regressive order)**: Number of lag observations $Y_{t-k}$.
- **$d$ (Integrated degree)**: Number of non-seasonal differencing iterations required for stationarity.
- **$q$ (Moving Average order)**: Number of lagged forecast error terms $\epsilon_{t-k}$.

### Mathematical Formulation:
$$\left(1 - \sum_{i=1}^p \phi_i B^i\right) (1 - B)^d Y_t = \left(1 + \sum_{j=1}^q \theta_j B^j\right) \epsilon_t$$

where $B$ is the backshift operator ($B^k Y_t = Y_{t-k}$).

### SARIMAX:
Extends ARIMA to include seasonal parameters $(P, D, Q)_s$ and exogenous predictors $X_t$ (e.g. price promotions).

---

## 13.3 Recurrent Neural Networks (RNNs) & LSTMs

Standard feedforward networks process inputs independently. **Recurrent Neural Networks (RNNs)** pass a hidden state vector $h_t$ sequentially:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

```
                               LSTM Cell Architecture
                                          C_{t-1} (Cell State)
                                             │
                       ┌─────────────────────┼─────────────────────┐
                       │                     ▼                     │
                       │           ┌───────────────────┐           │ ──► C_t
                       │           │   Cell State (*)  │           │
                       │           └─────────▲─────────┘           │
                       │                     │                     │
  h_{t-1} ─────────────┼──────────► (x) ─────┼───────► (+) ────────┼───► h_t
                       │             ▲       │          ▲          │
                       │             │       │          │          │
                       │          ┌──┴───┐┌──┴───┐  ┌───┴──┐       │
                       │          │Forget││Input │  │Output│       │
                       │          │Gate  ││Gate  │  │Gate  │       │
                       │          └──▲───┘└──▲───┘  └───▲──┘       │
                       └─────────────┴───────┴──────────┴──────────┘
                                             ▲
                                            x_t (Input)
```

### Long Short-Term Memory (LSTM) Cell Equations:
LSTMs solve vanishing gradients via an additive **Cell State** $C_t$ and three gating mechanisms:

1. **Forget Gate** (Determines what information to discard):
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate** (Determines what new information to store):
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update**:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate & Hidden State**:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

---

## 13.4 Modern Forecasting Transformers: TFT & N-BEATS

### 1. N-BEATS (Neural Basis Expansion Analysis)
Deep neural architecture constructed from stacked backward and forward residual blocks using basis functions (Fourier, Monomials) without requiring step-by-step recurrent iterations.

### 2. Temporal Fusion Transformer (TFT)
Combines multi-head attention with specialized Variable Selection Networks (VSN) and Gated Residual Networks (GRN) to handle static metadata, known future inputs (promotions), and historical time series simultaneously.

---

## ⚓ Repository Code Reference
- See [`src/demand_forecasting/pipeline.py`](file:///e:/Downloads/ML_only/src/demand_forecasting/pipeline.py) for autoregressive lag feature building and multi-step recursive forecasting.
