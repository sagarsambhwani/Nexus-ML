# Chapter 12: Exploratory Data Analysis & Statistical Hypothesis Thinking

---

## 1. Big Picture

Exploratory Data Analysis (EDA) is not about drawing pretty charts for a slideshow presentation. It is **detective work for engineers**.

EDA answers critical technical questions: Why is this feature distribution skewed? Is this correlation spurious? Is there target leakage? What interaction effects exist between features?

---

## 2. Intuition

- **Anscombe's Quartet & DatasauRus Dozen**: 4 datasets can have identical means (9.0), identical variances (11.0), identical correlations (0.81), and identical linear regression lines, yet have completely different visual distributions (one is a line, one is a curve, one has an outlier, one is vertical!).
- Never trust summary statistics alone—always combine statistical hypothesis tests with visualization.

---

## 3. Visualization

```text
Anscombe's Quartet Concept:
  Dataset 1: Linear Scatter    Dataset 2: Parabolic Curve
      y ▲                          y ▲
        │   * *                      │     * * *
        │ * * *                      │   *       *
        │* *                         │ *           *
        └─────────► x                └──────────────► x
   (Same Mean, Same Variance, Same Linear Fit y = 3.0 + 0.5x!)
```

---

## 4. Mathematics

### 1. Pearson Correlation vs Spearman Rank Correlation
Pearson measures linear relationship:
$$r_{xy} = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}}$$

Spearman measures monotonic relationship using ranks $R(x_i), R(y_i)$:
$$\rho = 1 - \frac{6 \sum d_i^2}{N(N^2 - 1)} \quad \text{where } d_i = R(x_i) - R(y_i)$$

### 2. Kolmogorov-Smirnov (KS) Normality Test
$$D = \sup_x |F_N(x) - F_0(x)|$$

---

## 5. Python (Automated Statistical EDA Engine)

```python
import numpy as np
import pandas as pd
from scipy import stats

def automated_eda(df: pd.DataFrame):
    print("=== AUTOMATED STATISTICAL EDA REPORT ===")
    print(f"Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} Columns\n")
    
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        # Test Normality via Shapiro-Wilk test
        stat, p_val = stats.shapiro(series[:5000]) if len(series) > 5000 else stats.shapiro(series)
        is_normal = p_val > 0.05
        
        print(f"Feature: {col:<20}")
        print(f"  ├─ Skewness: {skewness:.4f} ({'Right-Skewed' if skewness > 1 else 'Left-Skewed' if skewness < -1 else 'Symmetric'})")
        print(f"  ├─ Kurtosis: {kurtosis:.4f}")
        print(f"  └─ Gaussian Normal? {is_normal} (Shapiro p-value: {p_val:.4e})")
        print("-" * 50)

# Run EDA
np.random.seed(42)
df_sample = pd.DataFrame({
    'income': np.random.lognormal(mean=10.5, sigma=0.75, size=1000), # Log-normal (Skewed)
    'age': np.random.normal(loc=40, scale=10, size=1000)            # Gaussian Normal
})
automated_eda(df_sample)
```

---

## 6. Production Library (YData Profiling & Great Expectations)

```python
# In production pipelines, automated data quality reports are generated programmatically:
# from ydata_profiling import ProfileReport
# profile = ProfileReport(df, title="Production EDA Report", minimode=True)
# profile.to_file("eda_report.html")
```

---

## 7. Under the Hood

- Log-normal distributions $Y = \exp(X)$ where $X \sim \mathcal{N}(\mu, \sigma^2)$ occur naturally in income, house prices, and web latency data due to multiplicative growth processes.
- Applying a log transformation $\log(1 + X)$ converts skewed distributions back to symmetric Gaussian distributions, stabilizing linear regression variance.

---

## 8. Engineering Perspective

- **Target Leakage Discovery**: If a feature exhibits a Pearson correlation $r > 0.98$ with the target variable, inspect it immediately! It is almost certainly a target leakage feature recorded *after* the label event occurred.

---

## 9. Common Mistakes

1. **Assuming Linear Correlations Only**: Using Pearson correlation exclusively and missing strong non-linear relationships (e.g., quadratic $y = x^2$ has $r \approx 0$).
2. **Ignoring Class Imbalance**: Computing overall accuracy on 99:1 imbalanced dataset instead of evaluating minority class distribution.

---

## 10. Interview Questions

### Q1: How do you detect and fix multi-collinearity during EDA?
**Answer**: Multi-collinearity occurs when two or more independent features are highly correlated ($r > 0.85$). Detect it by calculating Variance Inflation Factor (VIF): $\text{VIF}_j = \frac{1}{1 - R_j^2}$. Fix it by dropping high-VIF features ($\text{VIF} > 10$), combining them via PCA, or using $L_2$ Regularization.

---

## 11. Exercises

1. **Coding**: Write a Python function that calculates VIF for all numerical features in a DataFrame.
2. **Math**: Prove that $\log(X)$ transformation reduces right-skewness for exponential family distributions.

---

## 12. Mini Project: Interactive EDA Dashboard Script

Write a script `generate_eda_dashboard.py` using Polars and Plotly that outputs interactive HTML distribution plots, correlation heatmaps, and VIF scores.

---

## 13. Capstone Integration

Exploration stage used in `src/house_prices/pipeline.py` and `src/demand_forecasting/pipeline.py`.
