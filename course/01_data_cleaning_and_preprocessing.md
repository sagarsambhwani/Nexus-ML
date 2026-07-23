# 📘 Chapter 1: Data Cleaning, Imputation & Preprocessing Pipelines

## 1.1 Introduction to Real-World Data Imperfections
In academic machine learning benchmarks (e.g. MNIST, Iris), datasets are clean and complete. In enterprise production environments, raw data is noisy, incomplete, corrupt, and subject to sensor failures, user entry errors, and upstream database schema shifts.

Garbage in yields garbage out ($GIGO$). Data cleaning and preprocessing form 70-80% of an applied machine learning engineer's workflow.

---

## 1.2 Missing Data Taxonomy & Mechanics

Before imputing missing data, you must understand the underlying mechanism generating the missing values. Missing data is categorized into three distinct statistical classes:

```
                          Missing Data Taxonomy
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
Missing Completely           Missing at Random           Missing Not at Random
  at Random (MCAR)                (MAR)                       (MNAR)
```

### 1. Missing Completely at Random (MCAR)
- **Definition**: The probability of a data point being missing is entirely independent of both observed data and unobserved parameters.
- **Formulation**: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M)$
- **Example**: A laboratory test sample tube slips and breaks on the floor accidentally.
- **Handling**: Safe to perform listwise deletion if the missing percentage is small (< 5%), or apply simple univariate imputation (Mean/Median/Mode).

### 2. Missing at Random (MAR)
- **Definition**: The probability of missingness depends systematically on observed data, but *not* on the missing values themselves.
- **Formulation**: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M \mid Y_{\text{obs}})$
- **Example**: Older survey respondents are less likely to report income. Income missingness correlates with the observed variable *Age*.
- **Handling**: Advanced multivariate imputation (KNNImputer, MICE).

### 3. Missing Not at Random (MNAR)
- **Definition**: The probability of missingness depends directly on the unobserved missing values themselves.
- **Formulation**: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) \neq P(M \mid Y_{\text{obs}})$
- **Example**: High-income individuals intentionally skip disclosing their income on tax surveys due to privacy concerns.
- **Handling**: Explicit missingness indicators ($\mathbb{I}_{\text{missing}}$) or domain-specific pattern modeling.

---

## 1.3 Imputation Strategies

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

### 1. Univariate Imputation (Mean / Median / Mode)
- **Mean Imputation**: Replaces missing values with feature mean $\mu_j$. Use *only* for normally distributed numerical data.
- **Median Imputation**: Replaces missing values with feature median $\text{Med}_j$. Robust to heavy-tailed distributions and outliers.
- **Mode Imputation**: Replaces missing categorical values with the most frequent category.

```python
# Simple Median Imputer
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)
```

### 2. $K$-Nearest Neighbors Imputation (KNNImputer)
Computes Euclidean distance between samples using observed features to impute missing values from the $k$ nearest neighbors:
$$d(x_a, x_b) = \sqrt{\sum_{i \in \text{observed}} (x_{a, i} - x_{b, i})^2}$$

```python
# KNN Imputer using 5 nearest neighbors
knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
X_knn = knn_imputer.fit_transform(X_raw)
```

### 3. Multivariate Imputation by Chained Equations (MICE)
Fits a series of regression models sequentially across missing features. Each feature with missing values is modeled as a function of all other features:
$$Y_j = \beta_0 + \beta_1 Y_1 + \dots + \beta_{j-1} Y_{j-1} + \epsilon$$

```python
# MICE (IterativeImputer)
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
X_mice = mice_imputer.fit_transform(X_raw)
```

---

## 1.4 Outlier Detection & Remediation

An outlier is a data point that deviates significantly from the remaining sample. Outliers distort mean statistics, inflate variance, and degrade gradient descent convergence.

### 1. Interquartile Range (IQR) Rule
- **First Quartile ($Q_1$)**: 25th percentile
- **Third Quartile ($Q_3$)**: 75th percentile
- **$\text{IQR}$**: $Q_3 - Q_1$
- **Lower Bound**: $Q_1 - 1.5 \times \text{IQR}$
- **Upper Bound**: $Q_3 + 1.5 \times \text{IQR}$

```python
def winsorize_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df
```

### 2. $Z$-Score Filtering
Calculates how many standard deviations $\sigma$ a point $x_i$ lies from the mean $\mu$:
$$Z = \frac{x_i - \mu}{\sigma}$$
Points where $|Z| > 3.0$ are flagged as extreme anomalies.

### 3. Isolation Forest (Unsupervised Anomaly Detection)
Isolates anomalies by randomly selecting a feature and split value. Anomalous points require significantly fewer splits to isolate compared to normal dense clusters.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
anomalies = iso.fit_predict(X) # -1 for anomalies, 1 for normal
```

---

## 1.5 Preventing Data Leakage in Preprocessing

> [!CAUTION]
> **Data Leakage Hazard**: Calculating global preprocessing parameters (e.g. mean $\mu$, standard deviation $\sigma$, median) on the *entire dataset before splitting* leaks information from the test set into the training set, causing overly optimistic validation metrics that fail catastrophically in production.

### Correct Preprocessing Pipeline Architecture:

```
Raw Data
   │
   ├──► Train Set ──► fit_transform(Scaler/Imputer) ──► Model Fit
   │
   └──► Test Set  ──► transform(Scaler/Imputer)     ──► Model Predict
```

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Correct Scikit-Learn Pipeline preventing leakage
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# Fit ONLY on training data; transform applied safely during cross-validation
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## ⚓ Repository Code Reference
See [`src/medical_diagnosis/pipeline.py`](file:///e:/Downloads/ML_only/src/medical_diagnosis/pipeline.py) for an enterprise implementation of standardized data scaling and missing feature handling.
