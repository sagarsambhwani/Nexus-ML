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

# 📘 Machine Learning Feature Engineering Notes by ChatGPT

Excellent choice. **Feature Engineering** is one of the highest-impact skills in machine learning, especially for structured (tabular) data. A well-engineered feature can improve a model more than switching to a more complex algorithm.

---

# 📘 Machine Learning Notes

# Module 4: Feature Engineering

## What is Feature Engineering?

Feature engineering is the process of **creating, transforming, or selecting features (columns)** that help a machine learning model learn patterns more effectively.

Think of it as giving the model **better information**, not just more information.

> **Raw Data → Better Features → Better Model**

---

# Why is Feature Engineering Important?

Imagine predicting house prices.

Original features:

| Size | Bedrooms |
| ---- | -------- |
| 1200 | 3        |

Now add:

* Price per square foot
* House age
* Distance to city center
* Number of nearby schools

These new features often capture relationships the original data does not.

---

# Feature Engineering Pipeline

```
Raw Features
      │
      ▼
Feature Creation
      │
      ▼
Feature Transformation
      │
      ▼
Feature Extraction
      │
      ▼
Feature Selection
      │
      ▼
Machine Learning Model
```

---

# Types of Feature Engineering

| Category               | Purpose                       |
| ---------------------- | ----------------------------- |
| Feature Creation       | Create new columns            |
| Feature Transformation | Change existing values        |
| Feature Extraction     | Extract useful information    |
| Feature Encoding       | Convert categories to numbers |
| Feature Scaling        | Normalize numerical values    |
| Feature Selection      | Keep only useful features     |

---

# 1. Feature Creation

Create new features from existing ones.

## Example 1 — BMI

Dataset

| Height (m) | Weight (kg) |
| ---------- | ----------- |
| 1.75       | 70          |

Create BMI

```python
df["BMI"] = df["Weight"] / (df["Height"] ** 2)
```

Result

| Height | Weight | BMI   |
| ------ | ------ | ----- |
| 1.75   | 70     | 22.86 |

---

## Example 2 — Age from Date of Birth

```python
import pandas as pd

today = pd.Timestamp.today()

df["Age"] = (
    (today - df["DOB"]).dt.days // 365
)
```

---

## Example 3 — Total Purchase

```python
df["Total"] = (
    df["Price"] *
    df["Quantity"]
)
```

---

# 2. Date Feature Engineering

Dates contain valuable information.

Original

```
2026-05-14
```

Extract

* Year
* Month
* Day
* Weekday
* Quarter
* Weekend

```python
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()
```

---

## Weekend Feature

```python
df["Weekend"] = (
    df["Date"]
      .dt.dayofweek >= 5
)
```

---

# 3. Text Feature Engineering

Original

```
"I love machine learning."
```

Useful features

* Number of words
* Number of characters
* Average word length
* Number of capital letters

---

## Word Count

```python
df["WordCount"] = (
    df["Review"]
      .str.split()
      .str.len()
)
```

---

## Character Count

```python
df["Characters"] = (
    df["Review"]
      .str.len()
)
```

---

# 4. Categorical Feature Engineering

Original

```
Red
Blue
Green
```

Machine learning models require numbers.

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["Color"] = encoder.fit_transform(df["Color"])
```

Result

```
Blue = 0
Green = 1
Red = 2
```

> Use Label Encoding mainly for **ordinal categories** or tree-based models. For nominal categories, prefer One-Hot Encoding.

---

### One-Hot Encoding

```python
pd.get_dummies(
    df,
    columns=["Color"],
    dtype=int
)
```

Result

| Red | Blue | Green |
| --- | ---- | ----- |
| 1   | 0    | 0     |

---

# 5. Numerical Transformations

Some variables are highly skewed.

Example

```
Income

25000
30000
35000
40000
10000000
```

Use logarithmic transformation.

```python
import numpy as np

df["Income"] = np.log1p(
    df["Income"]
)
```

`log1p()` safely computes `log(1 + x)` and works with zero values.

---

# 6. Binning

Convert continuous values into groups.

Age

```
18
24
31
52
67
```

↓

```
Young
Adult
Senior
```

```python
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0,18,35,60,100],
    labels=[
        "Child",
        "Adult",
        "Middle",
        "Senior"
    ]
)
```

---

# 7. Interaction Features

Sometimes combining features reveals useful patterns.

Example

```
Length
Width
```

↓

Area

```python
df["Area"] = (
    df["Length"] *
    df["Width"]
)
```

---

# 8. Polynomial Features

Useful for linear models when relationships are nonlinear.

Example

```
x
```

↓

```
x
x²
x³
```

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

new_features = poly.fit_transform(
    df[["x"]]
)
```

---

# 9. Aggregation Features

Customer Orders

| Customer | Amount |
| -------- | ------ |
| A        | 200    |
| A        | 300    |
| A        | 150    |

Create

Average Purchase

```python
avg = (
    df.groupby("Customer")["Amount"]
      .transform("mean")
)

df["AveragePurchase"] = avg
```

---

# 10. Frequency Encoding

Replace categories by how often they appear.

Example

```
City

Delhi
Delhi
Mumbai
Jaipur
Delhi
```

↓

```
Delhi = 3
Mumbai = 1
Jaipur = 1
```

```python
freq = df["City"].value_counts()

df["CityFreq"] = (
    df["City"]
      .map(freq)
)
```

---

# 11. Target Encoding (Use Carefully)

Replace categories with the target mean.

| City   | Purchased |
| ------ | --------- |
| Delhi  | 1         |
| Delhi  | 0         |
| Mumbai | 1         |

↓

```
Delhi = 0.5
Mumbai = 1.0
```

```python
target_mean = (
    df.groupby("City")["Purchased"]
      .mean()
)

df["CityEncoded"] = (
    df["City"]
      .map(target_mean)
)
```

⚠️ Compute this **only on the training set** to avoid data leakage.

---

# Common Feature Engineering Ideas

| Original             | New Feature          |
| -------------------- | -------------------- |
| Date                 | Year, Month, Weekday |
| Name                 | First name length    |
| Salary               | Salary category      |
| Height + Weight      | BMI                  |
| Price + Quantity     | Total                |
| Latitude + Longitude | Distance to store    |
| Review               | Word count           |
| Email                | Email domain         |
| Timestamp            | Hour of day          |
| Age                  | Age group            |

---

# Common Mistakes

❌ Creating features without understanding the business problem.

❌ One-hot encoding high-cardinality columns (thousands of unique values).

❌ Applying target encoding before splitting the data.

❌ Creating features that accidentally include future information.

❌ Generating hundreds of unnecessary features that add noise.

---

# Interview Questions

1. What is feature engineering?
2. Why can feature engineering improve model performance?
3. When should you use One-Hot Encoding instead of Label Encoding?
4. What is data leakage in feature engineering?
5. Give three examples of engineered features.
6. What are interaction features?

---

# Practice Exercise

Dataset:

| CustomerID | DOB        | Price | Quantity | City   | Review               |
| ---------- | ---------- | ----- | -------- | ------ | -------------------- |
| 101        | 1998-05-14 | 120   | 2        | Delhi  | "Great service!"     |
| 102        | 1992-09-01 | 250   | 1        | Mumbai | "Very fast delivery" |
| 103        | 1985-12-20 | 80    | 5        | Delhi  | "Excellent quality"  |

Create the following features:

1. Age from `DOB`
2. Total purchase (`Price × Quantity`)
3. Birth month
4. Word count of the review
5. One-hot encode `City`
6. Create an `AgeGroup` column using bins
7. Compute city frequency encoding
8. Build a final feature matrix ready for a machine learning model

---

## Key Takeaway

Feature engineering is not about applying every technique. It's about asking:

* **What information does the model need that isn't explicitly present?**
* **Can I derive that information from the existing data without leaking future knowledge?**

The best feature engineers combine domain understanding with careful validation to create features that genuinely improve predictive performance.


