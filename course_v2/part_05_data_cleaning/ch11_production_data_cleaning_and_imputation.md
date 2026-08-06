# Chapter 11: Production Data Cleaning and Imputation

---

## 1. Big Picture

"Garbage in, garbage out." Real-world data is corrupted with missing values, corrupt strings, out-of-range sensor spikes, duplicate records, and inconsistent categorical encodings.

Data cleaning is not a static one-off notebook task. In ML engineering, data cleaning requires building **statistically sound, reproducible, and automated cleaning pipelines** capable of handling missing data mechanisms (MCAR, MAR, MNAR) without causing data leakage.

---

## 2. Intuition

Imagine a medical sensor recording patient heart rates:
- **Missing Completely at Random (MCAR)**: Sensor battery dies randomly. Imputation via mean/median is mathematically safe.
- **Missing at Random (MAR)**: Elderly patients forget to wear sensors more often, but missingness is fully explained by recorded `Age`. Imputation using group-by models (KNN/Iterative) works well.
- **Missing Not at Random (MNAR)**: Sensor fails specifically when patient heart rate exceeds 180 bpm (sensor clips/crashes). Imputing with mean will severely underestimate heart rate risks! You must add a binary `is_missing` indicator feature!

---

## 3. Visualization

```text
Missing Data Taxonomy & Imputation Strategy:

                      Missing Data Detected
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
    MCAR                     MAR                     MNAR
(Random Glitch)       (Explained by Age)      (High Value Crash)
       │                       │                       │
       ▼                       ▼                       ▼
Mean / Median Impute    KNN / MICE Impute      Missingness Indicator
                                               Feature + Domain Impute
```

---

## 4. Mathematics

### Multivariate Imputation by Chained Equations (MICE / IterativeImputer)
For dataset $X = (X_1, X_2, \dots, X_p)$ where columns have missing values:

1. Initialize missing values in all columns using mean imputation: $X_j^{(0)}$.
2. For each column $j = 1 \dots p$:
   - Fit regression model predicting observed $X_j^{obs}$ using all other columns $X_{-j}^{(t-1)}$:
$$\hat{\gamma}_j = \arg\min_\gamma \| X_j^{obs} - X_{-j}^{obs} \gamma \|^2$$
   - Update missing values $X_j^{miss}$ using predicted model values:
$$X_j^{(t)} = X_{-j}^{miss} \hat{\gamma}_j$$
3. Repeat step 2 for $T$ iterations until parameters stabilize.

---

## 5. Python (From Scratch Production Cleaner)

A production-grade Python data cleaning class adhering to Scikit-Learn Transformer API:

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ProductionDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, z_score_threshold=3.0):
        self.z_score_threshold = z_score_threshold
        self.medians_ = {}
        self.iqr_bounds_ = {}
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.select_dtypes(include=[np.number]).columns:
            # Store training median
            self.medians_[col] = X_df[col].median()
            
            # Store IQR Outlier Bounds
            q25, q75 = X_df[col].quantile(0.25), X_df[col].quantile(0.75)
            iqr = q75 - q25
            self.iqr_bounds_[col] = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
            
        return self
        
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        for col, median_val in self.medians_.items():
            # 1. Add missingness indicator (For MNAR handling)
            if X_df[col].isnull().any():
                X_df[f"{col}_was_missing"] = X_df[col].isnull().astype(int)
                
            # 2. Median Imputation
            X_df[col] = X_df[col].fillna(median_val)
            
            # 3. Winsorize Outliers using training bounds
            lower, upper = self.iqr_bounds_[col]
            X_df[col] = np.clip(X_df[col], lower, upper)
            
        return X_df.values

# Test Transformer
X_raw = np.array([[10.0], [12.0], [np.nan], [1000.0]]) # 1000 is an outlier
cleaner = ProductionDataCleaner()
X_clean = cleaner.fit_transform(X_raw)
print("Cleaned Data (Imputed + Winsorized Outlier Capped):")
print(X_clean)
```

---

## 6. Production Library (Scikit-Learn KNN & Iterative Imputer)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# MICE Imputer for correlated continuous numerical columns
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = mice_imputer.fit_transform(X_raw)
```

---

## 7. Under the Hood

- `IterativeImputer` uses Bayesian Ridge regression by default, sampling parameter vectors from posterior distributions to preserve true variance instead of shrinking variance like simple mean imputation.

---

## 8. Engineering Perspective

- **Pipeline Encapsulation**: Never calculate global mean/std across the full dataset *before* splitting into train/test! Outlier thresholds and mean values must be learned strictly on `X_train` during `.fit()` and applied to `X_test` via `.transform()`.

---

## 9. Common Mistakes

1. **Dropping Missing Value Rows (Listwise Deletion)**: Dropping every row containing any missing value often destroys 50%+ of training data and introduces severe selection bias.
2. **Hardcoding Outlier Magic Numbers**: Using static thresholds like `if value > 500` instead of statistical bounds (IQR, Z-Score).

---

## 10. Interview Questions

### Q1: Compare Mean Imputation, KNN Imputation, and MICE. What are their trade-offs?
**Answer**: Mean imputation is fast $O(1)$ but underestimates feature variance and distorts correlations. KNN imputation preserves local feature relationships but has high inference complexity $O(N \cdot d)$. MICE (Iterative Imputer) models complex multivariate relationships accurately with moderate training cost $O(T \cdot d \cdot N)$.

---

## 11. Exercises

1. **Coding**: Implement a custom Scikit-Learn transformer that cleans messy string columns (lowercasing, regex whitespace stripping, accent removal).
2. **Math**: Prove that simple mean imputation under-estimates sample variance $S^2 = \frac{1}{N-1} \sum (x_i - \bar{x})^2$.

---

## 12. Mini Project: Production Automated Data Hygiene Suite

Write a module `data_hygiene.py` that validates incoming DataFrames against Pydantic schemas, identifies missing mechanisms, applies MICE imputation, caps outliers, and exports clean Parquet artifacts.

---

## 13. Capstone Integration

Implemented as the first execution stage across all 12 pipelines in `src/`, particularly `src/house_prices/pipeline.py` and `src/medical_diagnosis/pipeline.py`.
