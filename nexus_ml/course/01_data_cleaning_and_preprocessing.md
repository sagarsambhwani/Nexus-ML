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

# Notes from Chatgpt

That's exactly how I'd do it. I'd make it into **engineering notes**, not textbook notes. Each topic would have:

1. **Concept**
2. **Why it matters**
3. **When to use it**
4. **Example**
5. **Python/Pandas code**
6. **Common mistakes**
7. **Interview tips**
8. **Practice exercise**

Here's what the notes would look like.

---

# 📘 Machine Learning Data Preprocessing Notes

## Module 1: Introduction to Data Preprocessing

### What is Data Preprocessing?

Data preprocessing is the process of transforming raw, messy data into a clean, structured, and machine-learning-ready dataset.

Raw Data → Clean Data → Feature Engineering → Model Training

### Why is it Important?

Imagine building a house.

The machine learning algorithm is the house.

The dataset is the foundation.

A weak foundation means the house fails regardless of how good the design is.

Many real-world ML projects spend more time preparing data than training models.

---

# Data Preprocessing Pipeline

```
Raw Data
    │
    ▼
Data Inspection
    │
    ▼
Data Cleaning
    │
    ▼
Missing Value Handling
    │
    ▼
Outlier Detection
    │
    ▼
Data Transformation
    │
    ▼
Encoding
    │
    ▼
Scaling
    │
    ▼
Feature Engineering
    │
    ▼
Feature Selection
    │
    ▼
Train/Test Split
    │
    ▼
Machine Learning Model
```

---

# Module 2: Data Inspection

Before cleaning anything, understand the dataset.

## Load Dataset

```python
import pandas as pd

df = pd.read_csv("employees.csv")
```

---

## View First Rows

```python
df.head()
```

Output

```
Name   Age  Salary
John   25   50000
Alice  30   60000
...
```

---

## Dataset Information

```python
df.info()
```

Shows

* Number of rows
* Number of columns
* Data types
* Missing values

---

## Summary Statistics

```python
df.describe()
```

Useful for numerical columns.

---

## Column Names

```python
df.columns
```

---

## Shape

```python
df.shape
```

Example

```
(1000, 12)
```

1000 rows

12 columns

---

# Module 3: Data Cleaning

## What is Data Cleaning?

Removing incorrect, inconsistent, duplicate, or invalid data.

---

## Types of Data Problems

* Missing values
* Duplicate records
* Wrong data types
* Invalid values
* Inconsistent formatting
* Extra spaces
* Typographical errors

---

# Missing Values

Example

```
Age

25
30
NaN
40
```

Find missing values

```python
df.isnull()
```

Count missing values

```python
df.isnull().sum()
```

Percentage

```python
(df.isnull().sum()/len(df))*100
```

---

# Remove Missing Values

Remove rows

```python
df.dropna()
```

Remove columns

```python
df.dropna(axis=1)
```

---

# Fill Missing Values

Mean

```python
df["Age"].fillna(df["Age"].mean())
```

Median

```python
df["Age"].fillna(df["Age"].median())
```

Mode

```python
df["City"].fillna(df["City"].mode()[0])
```

Constant

```python
df["Gender"].fillna("Unknown")
```

---

## Which Method Should You Use?

| Data Type   | Preferred Method                               |
| ----------- | ---------------------------------------------- |
| Numerical   | Median (often robust), Mean (when appropriate) |
| Categorical | Mode                                           |
| Time Series | Forward/Backward Fill                          |

---

# Duplicate Data

Example

```
John
John
John
```

Check duplicates

```python
df.duplicated()
```

Count

```python
df.duplicated().sum()
```

Remove duplicates

```python
df.drop_duplicates()
```

---

# Wrong Data Types

Current

```
Age = "25"
```

Should be

```
Age = 25
```

Check types

```python
df.dtypes
```

Convert

```python
df["Age"] = df["Age"].astype(int)
```

Date conversion

```python
df["Date"] = pd.to_datetime(df["Date"])
```

---

# Remove Extra Spaces

Before

```
John

John
```

Code

```python
df["Name"] = df["Name"].str.strip()
```

---

# Rename Columns

Before

```
Employee Name
```

After

```
employee_name
```

```python
df.columns = (
    df.columns
      .str.lower()
      .str.replace(" ", "_")
)
```

---

# Replace Values

Example

```
M
Male

male
```

Standardize

```python
df["Gender"] = df["Gender"].replace({
    "M":"Male",
    "male":"Male"
})
```

---

# Remove Invalid Values

Example

Age

```
-5
```

```python
df = df[df["Age"] > 0]
```

---

# Detect Outliers

Using IQR

```python
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Salary"] < lower) |
    (df["Salary"] > upper)
]
```

---

# Remove Outliers

```python
df = df[
    (df["Salary"] >= lower) &
    (df["Salary"] <= upper)
]
```

---

# Check Unique Values

```python
df["Department"].unique()
```

Count

```python
df["Department"].value_counts()
```

---

# Data Consistency

Suppose

```
Yes

YES

yes

Y
```

Standardize

```python
df["Purchased"] = (
    df["Purchased"]
      .str.lower()
      .replace({
          "y":"yes"
      })
)
```

---

# Save Clean Dataset

```python
df.to_csv(
    "clean_data.csv",
    index=False
)
```

---

# Common Mistakes

❌ Removing all missing values without checking why they're missing.

❌ Dropping outliers that are actually valid observations (e.g., a CEO's salary).

❌ Converting data types without handling invalid values first.

❌ Forgetting to standardize inconsistent text values before encoding.

❌ Applying transformations before understanding the data.

---

# Interview Questions

1. When would you use the median instead of the mean for missing values?
2. What is the difference between cleaning data and preprocessing data?
3. Why shouldn't you remove every outlier?
4. How do you detect duplicate records?
5. How do you convert a string column into datetime?
6. What steps do you take before training a machine learning model?

---

# Practice Exercise

Given this dataset:

| Name  | Age |  Salary | City   |
| ----- | --: | ------: | ------ |
| Alice |  25 |   50000 | Delhi  |
| Bob   | NaN |   60000 | Mumbai |
| Bob   | NaN |   60000 | Mumbai |
| Carol |  -5 |   45000 | Delhi  |
| Dave  |  30 | 5000000 | Jaipur |
| Eve   |  28 |     NaN | Delhi  |

Tasks:

1. Load the dataset into a DataFrame.
2. Inspect the dataset (`head`, `info`, `describe`).
3. Count missing values.
4. Remove duplicate rows.
5. Handle missing values appropriately.
6. Remove invalid ages.
7. Detect salary outliers using the IQR method.
8. Decide whether to keep or remove the outlier, and explain why.
9. Save the cleaned dataset to `clean_data.csv`.

---

This covers the **Data Cleaning** portion of preprocessing. The next logical section would be **Data Transformation**, where you'll learn scaling, normalization, categorical encoding, and feature engineering—techniques that prepare clean data for machine learning algorithms.

