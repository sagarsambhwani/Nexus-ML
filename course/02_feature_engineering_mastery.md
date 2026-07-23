# 📙 Chapter 2: Feature Engineering Mastery & Signal Extraction

## 2.1 The Role of Feature Engineering
Feature engineering is the process of converting raw variables into quantitative representations that maximize the predictive signal for machine learning algorithms. While deep learning models learn representations automatically from raw pixels or waveforms, tabular enterprise machine learning relies heavily on domain-specific feature extraction.

---

## 2.2 Categorical Feature Encoding

Categorical features contain discrete text values (e.g. `City = ["NYC", "London", "Tokyo"]`). ML models operate exclusively on numerical vectors, requiring transformation.

```
                             Categorical Encoding
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   One-Hot Encoding            Ordinal Encoding             Target Encoding
(Unordered Nominal Data)    (Ordered Sequential Data)   (High-Cardinality Features)
```

### 1. One-Hot Encoding (OHE)
Maps $K$ unique categorical values into $K$ binary columns:

$$\text{City} = \text{"NYC"} \rightarrow [1, 0, 0], \quad \text{"London"} \rightarrow [0, 1, 0]$$

- **When to use**: Nominal variables without intrinsic ordering and low cardinality ($K < 15$).
- **Dummy Variable Trap**: When using linear regression, drop one column ($K-1$) to prevent perfect multicollinearity:
  $$\mathbf{x}_K = 1 - \sum_{i=1}^{K-1} \mathbf{x}_i$$

```python
import pandas as pd
df_ohe = pd.get_dummies(df, columns=["category"], drop_first=True)
```

### 2. Ordinal Encoding
Assigns sequential integers based on natural ordering:
$$\text{Education} = [\text{"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}]$$

### 3. Target (Out-of-Fold) Encoding
Replaces categorical values with the mean target value for that category:
$$\hat{x}_i = \mathbb{E}[y \mid X = c_i]$$

To prevent target leakage and overfitting on small categories, use **smoothed out-of-fold target encoding**:
$$\hat{S}_c = \frac{n_c \cdot \bar{y}_c + m \cdot y_{\text{global}}}{n_c + m}$$
where $n_c$ is the category count, $\bar{y}_c$ is the category target mean, $y_{\text{global}}$ is the overall dataset mean, and $m$ is the smoothing weight.

---

## 2.3 Numerical Feature Scaling

Numerical variables often operate on vast scale disparities (e.g. Income \$100,000 vs. Age 25). Distance-based models ($K$-Means, KNN, SVM) and gradient-descent algorithms (Logistic Regression, Neural Networks) fail without scaling.

### 1. Standard Scaler ($Z$-Score Normalization)
Rescales feature to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
$$Z = \frac{X - \mu}{\sigma}$$
- **When to use**: Normally distributed data, linear models, Logistic Regression, PCA.

### 2. Min-Max Scaler
Rescales feature values into a bounded range $[0, 1]$:
$$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$
- **When to use**: Image pixel intensities, neural network inputs, or algorithms requiring positive values.

### 3. Robust Scaler (IQR-Based)
Uses median and Interquartile Range, making it robust to severe outliers:
$$X_{\text{scaled}} = \frac{X - \text{Median}}{\text{IQR}} = \frac{X - Q_2}{Q_3 - Q_1}$$

---

## 2.4 Mathematical Transformations for Skewed Features

Right-skewed features (e.g. Transaction Amounts, House Prices) create skewed residuals. Applying logarithmic or power transformations shifts the distribution toward Gaussian normality.

```
Right-Skewed Distribution ────────► Log1p Transform ────────► Gaussian Normal Bell Curve
```

### 1. Logarithmic Transformation ($\log(1+x)$)
$$\tilde{x} = \log(1 + x) = \text{np.log1p}(x)$$
Handles zero values safely without division errors ($\log(1 + 0) = 0$).

### 2. Box-Cox & Yeo-Johnson Transformations
Stabilizes variance and normalizes distributions via parameterized power transformations:
$$y^{(\lambda)} = \begin{cases} \frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \log(y) & \text{if } \lambda = 0 \end{cases}$$
*Box-Cox requires strictly positive values ($y > 0$), while Yeo-Johnson supports zero and negative values.*

---

## 2.5 Time-Series Feature Engineering

Time-series demand forecasting relies on converting sequential date timestamps into stationary autoregressive features.

### 1. Autoregressive Lag Features
Includes past values at fixed time lags ($t-k$):
$$\text{Lag}_1 = y_{t-1}, \quad \text{Lag}_7 = y_{t-7} \quad (\text{Weekly Seasonality})$$

### 2. Rolling Moving Window Statistics
Computes rolling summary metrics over historical sliding windows:
$$\text{RollingMean}_7 = \frac{1}{7} \sum_{i=1}^7 y_{t-i}$$
$$\text{RollingStd}_7 = \sqrt{\frac{1}{7} \sum_{i=1}^7 (y_{t-i} - \text{RollingMean}_7)^2}$$

### 3. Cyclical Sine/Cosine Encoding
Standard integer day-of-week encoding ($0 \dots 6$) implies Sunday (6) is far from Monday (0). Cyclical encoding preserves continuity on a continuous circle:

$$\text{Day}_{\sin} = \sin\left(\frac{2 \pi \cdot \text{day}}{7}\right), \quad \text{Day}_{\cos} = \cos\left(\frac{2 \pi \cdot \text{day}}{7}\right)$$

```python
import numpy as np

df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
```

---

## ⚓ Repository Code Reference
- See [`src/demand_forecasting/pipeline.py`](file:///e:/Downloads/ML_only/src/demand_forecasting/pipeline.py) for practical implementation of autoregressive lag creation (`lag_1`, `lag_7`, `rolling_mean_7`).
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for exponential distribution transformations and 1-hour velocity features.
