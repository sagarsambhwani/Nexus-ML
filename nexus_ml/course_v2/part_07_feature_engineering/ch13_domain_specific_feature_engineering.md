# Chapter 13: Domain-Specific Feature Engineering Mastery

---

## 1. Big Picture

Feature engineering is where raw data is converted into signal. Applied Machine Learning is largely feature engineering: raw columns rarely contain strong direct predictive signal, but domain-informed transformations extract latent relationships that allow models to achieve high accuracy.

This chapter covers domain-specific feature engineering patterns across 7 major industries:
1. **Real Estate**: Spatial interaction terms, price per sqft ratios, age decay.
2. **Financial Fraud**: Rolling window aggregation counters, velocity features, velocity deltas.
3. **Medical & Healthcare**: Non-linear risk ratios (BMI, MAP - Mean Arterial Pressure), interaction terms.
4. **Finance & Trading**: Exponential Moving Averages, Volatility, RSI, Log Returns.
5. **Sports Analytics**: Elo ratings, rolling performance momentum.
6. **Recommendation Systems**: User-Item interaction counts, target encoding, matrix factor embeddings.
7. **Weather & Climate**: Cyclical sine/cosine temporal encodings, lag differences.

---

## 2. Intuition

- **Cyclical Time Encoding**: Hour `23` and Hour `00` are 1 hour apart in real life, but numerically `23 - 0 = 23`! Mapping hours onto a 2D circle using $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)$ and $\cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$ makes midnight and 11 PM adjacent vectors in space.
- **Velocity Features**: A user spending $50 is normal. A user spending $50 *5 times in 60 seconds* is credit card fraud. Velocity features measure counts over tight temporal windows.

---

## 3. Visualization

```text
Cyclical Hour Encoding on Unit Circle:

                      Hour 0 (Midnight)
                       (sin=0, cos=1)
                             ▲
              Hour 18        │        Hour 6
           (sin=-1,cos=0) ───┼───► (sin=1,cos=0)
                             │
                       Hour 12 (Noon)
                       (sin=0, cos=-1)
```

---

## 4. Mathematics

### 1. Cyclical Sine/Cosine Transform
For periodic temporal feature $t \in [0, T-1]$ (e.g. $T=24$ for hours, $T=7$ for day of week):

$$x_{\sin} = \sin\left( \frac{2\pi t}{T} \right), \quad x_{\cos} = \cos\left( \frac{2\pi t}{T} \right)$$

### 2. Out-of-Fold Target Encoding (With Smoothing)
To encode high-cardinality categorical variable $C$ with smoothing weight $m$:

$$\hat{S}_k = \frac{n_k \cdot \bar{y}_k + m \cdot y_{\text{global}}}{n_k + m}$$

Where $n_k$ is sample count for category $k$, $\bar{y}_k$ is target mean for category $k$, and $y_{\text{global}}$ is overall training target mean.

---

## 5. Python (Production Domain Feature Engine)

```python
import numpy as np
import pandas as pd

class DomainFeatureEngineer:
    def __init__(self, target_encode_smooth=10.0):
        self.smooth = target_encode_smooth
        self.target_map_ = {}
        self.global_mean_ = 0.0
        
    def fit(self, df: pd.DataFrame, target_col: str, cat_col: str):
        self.global_mean_ = df[target_col].mean()
        stats = df.groupby(cat_col)[target_col].agg(['count', 'mean'])
        
        # Smoothed Target Encoding formula
        smoothed = (stats['count'] * stats['mean'] + self.smooth * self.global_mean_) / (stats['count'] + self.smooth)
        self.target_map_[cat_col] = smoothed.to_dict()
        return self

    def transform_financial_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        # 1. Transaction Velocity Ratios
        res['amt_to_avg_ratio'] = res['amount'] / (res['user_avg_amount_30d'] + 1e-5)
        # 2. Transaction Frequency Spike
        res['tx_speed_spike'] = res['tx_count_1h'] / (res['tx_count_24h'] / 24.0 + 1e-5)
        return res
        
    def transform_cyclical_time(self, df: pd.DataFrame, hour_col: str) -> pd.DataFrame:
        res = df.copy()
        res[f'{hour_col}_sin'] = np.sin(2 * np.pi * res[hour_col] / 24.0)
        res[f'{hour_col}_cos'] = np.cos(2 * np.pi * res[hour_col] / 24.0)
        return res

# Test Transformations
df_tx = pd.DataFrame({
    'user_id': [1, 1, 2],
    'amount': [500.0, 50.0, 10.0],
    'user_avg_amount_30d': [50.0, 50.0, 100.0],
    'tx_count_1h': [10, 1, 0],
    'tx_count_24h': [12, 5, 2],
    'hour': [23, 0, 14]
})

fe = DomainFeatureEngineer()
df_prep = fe.transform_financial_fraud(df_tx)
df_prep = fe.transform_cyclical_time(df_prep, 'hour')
print("Engineered Domain Features:")
print(df_prep[['amt_to_avg_ratio', 'tx_speed_spike', 'hour_sin', 'hour_cos']])
```

---

## 6. Production Library (Category Encoders)

```python
import category_encoders as ce

# Out-of-Fold Target Encoder preventing target leakage
encoder = ce.TargetEncoder(cols=['zipcode'], smoothing=10.0)
# X_encoded = encoder.fit_transform(X_train, y_train)
```

---

## 7. Under the Hood

- Target encoding without Out-of-Fold (OOF) cross-validation creates severe target leakage: the model memorizes category target averages rather than learning generalizable patterns. Always compute target encodings inside CV folds!

---

## 8. Engineering Perspective

- **Dimensionality Control**: One-Hot Encoding a categorical column with 5,000 unique values (e.g. ZIP codes) creates 5,000 sparse columns, swelling memory footprint. Use Target Encoding or Hashing Trick (`FeatureHasher`) instead.

---

## 9. Common Mistakes

1. **One-Hot Encoding High Cardinality Columns**: Creating thousands of sparse columns leading to tree-depth exhaustion in XGBoost.
2. **Un-smoothed Target Encoders**: Setting target encoding equal to category mean when sample count $n=1$, creating severe overfitting.

---

## 10. Interview Questions

### Q1: How do you prevent target leakage when performing target encoding?
**Answer**: Prevent target leakage by performing target encoding strictly inside Out-of-Fold (OOF) cross-validation splits (`KFold`). Additionally, apply smoothing priors $m$ to pull low-count category estimates toward the global target mean, and add small Gaussian noise during training.

---

## 11. Exercises

1. **Coding**: Implement a custom Hashing Trick feature encoder in pure Python supporting `n_features=1024`.
2. **Domain**: Propose 3 novel engineered features for predicting hospital readmission risk from EHR medical records.

---

## 12. Mini Project: Domain Feature Engineering Pipeline

Write a script `domain_features.py` that processes raw financial transaction logs and generates 25 engineered velocity, ratio, and cyclical time features.

---

## 13. Capstone Integration

Utilized extensively in `src/fraud_detection/pipeline.py`, `src/house_prices/pipeline.py`, and `src/demand_forecasting/pipeline.py`.
