# 📏 Chapter 44: Distance-Based Anomaly Detection (LOF & Mahalanobis)

## 44.1 Spatial Anomaly Detection
Identifies outliers based on feature space distances.

---

## 44.2 Mahalanobis Distance
Accounts for feature correlations using covariance matrix $\Sigma$:

$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

---

## 44.3 Local Outlier Factor (LOF)
Compares local reachability density of sample $x$ against its $k$-nearest neighbors to flag isolated spatial anomalies.

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for distance anomaly metrics.
