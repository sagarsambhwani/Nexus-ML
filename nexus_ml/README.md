# ⚡ Nexus ML — Enterprise Machine Learning Suite

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

> **12 Production-Ready Machine Learning & Artificial Intelligence Use Cases in ONE Modular Package.**
>
> Spanning Classification, Regression, Recommendation Systems, Time-Series Demand Forecasting, Predictive Telemetry, Healthcare Diagnostics, Natural Language Processing (NLP), and Unsupervised Clustering.

---

## 📑 12 Production ML Pipelines Summary

| # | Pipeline | Machine Learning Task | Model Architecture & Feature Engineering | Primary Outputs |
|---|---|---|---|---|
| 1 | [**Fraud Detection**](src/fraud_detection/README.md) | Classification | Random Forest / XGBoost + Velocity & Anomaly features | Fraud Prob, Alert Status, Risk Factors |
| 2 | [**Credit Risk Prediction**](src/credit_risk/README.md) | Classification | Calibrated Logistic Scorecard + Standard Scaler | Default Prob, Risk Tier (AAA-CCC), Decision |
| 3 | [**Customer Churn**](src/customer_churn/README.md) | Classification | Gradient Boosting + Tenure, Support Tickets, Contract | Churn Prob, Retention Action |
| 4 | [**House Price Valuation**](src/house_prices/README.md) | Regression | Gradient Boosting + SqFt, Geo Score, Age, Amenities | Predicted Price ($), Price Bounds |
| 5 | [**Recommendation System**](src/recommendation/README.md) | Hybrid Filter | TruncatedSVD Matrix Factorization + Content Filters | Top-K Items, Match Score %, Reason |
| 6 | [**Demand Forecasting**](src/demand_forecasting/README.md) | Time Series | Lag Feature Regressor + Day of Week, Seasonality | Daily Forecast Units, Confidence Bounds |
| 7 | [**Predictive Maintenance**](src/predictive_maintenance/README.md) | Telemetry / Class. | Random Forest + Sensor Vibration, Temp, Pressure, RPM | Failure Prob, RUL Hours, Status |
| 8 | [**Medical Diagnosis Support**](src/medical_diagnosis/README.md) | Healthcare Class. | Calibrated Logistic Classifier + Biomarkers | Disease Risk, Elevated Biomarkers, Guidance |
| 9 | [**Sentiment Analysis**](src/sentiment_analysis/README.md) | NLP Classification | TF-IDF (Unigram/Bigram) + Logistic Classifier | Sentiment (Pos/Neu/Neg), Score |
| 10 | [**Document Classification**](src/document_classification/README.md) | NLP Multi-class | TF-IDF + Multinomial Naive Bayes / Softmax | Doc Category, Keywords Extracted |
| 11 | [**Defect Detection**](src/defect_detection/README.md) | Quality Inspection | Visual Patch Texture & Anomaly Feature Classifier | Defect Type, Severity Grade, QC Status |
| 12 | [**Customer Segmentation**](src/customer_segmentation/README.md) | Clustering | StandardScaler + K-Means (K=4) + PCA 2D Mapper | Cluster ID, Persona Name, Strategy |

---

## 🛠️ Package Organization

- **`src/`**: Self-contained implementations for all 12 pipelines with their individual comprehensive READMEs.
- **`artifacts/models/`**: Serialized `.joblib` model binaries and training metadata.
- **`course/`**: 100-Chapter Full-Stack Machine Learning Compendium (V1).
- **`course_v2/`**: 20-Part Production Machine Learning Engineering Guide (V2).
- **`scripts/`**: Automated training orchestrator (`train_all.py`).
- **`config.py`**: Package-level configuration, random seeds, and artifact paths.
