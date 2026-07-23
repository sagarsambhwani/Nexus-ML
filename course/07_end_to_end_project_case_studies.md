# 📙 Chapter 7: End-to-End Industry Case Studies (12 Pipelines)

## 7.1 Cross-Disciplinary Case Studies Overview

This chapter presents comprehensive architectural case studies cross-referencing the **12 Production ML Pipelines** implemented in this repository (`src/`).

---

## 7.2 Case Study Index

### 1. 🛡️ [Fraud Detection](file:///e:/Downloads/ML_only/src/fraud_detection/README.md)
- **Domain**: Financial Payments / Credit Card Security
- **Learning Objective**: Imbalanced classification, velocity feature engineering, and sub-50ms inference SLAs.
- **Source Code**: [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py)

### 2. 💳 [Credit Risk Prediction](file:///e:/Downloads/ML_only/src/credit_risk/README.md)
- **Domain**: Banking & Loan Underwriting
- **Learning Objective**: Regulatory compliance, adverse action points, and $L_2$-regularized scorecard logit calibration.
- **Source Code**: [`src/credit_risk/pipeline.py`](file:///e:/Downloads/ML_only/src/credit_risk/pipeline.py)

### 3. 📉 [Customer Churn Prediction](file:///e:/Downloads/ML_only/src/customer_churn/README.md)
- **Domain**: SaaS / Telecom Subscription Management
- **Learning Objective**: Retention propensity scoring, feature attributions, and automated campaign action engines.
- **Source Code**: [`src/customer_churn/pipeline.py`](file:///e:/Downloads/ML_only/src/customer_churn/pipeline.py)

### 4. 🏡 [House Price Valuation](file:///e:/Downloads/ML_only/src/house_prices/README.md)
- **Domain**: PropTech & Automated Property Valuation (AVM)
- **Learning Objective**: Non-linear continuous regression, price-per-sqft valuation, and 95% confidence intervals.
- **Source Code**: [`src/house_prices/pipeline.py`](file:///e:/Downloads/ML_only/src/house_prices/pipeline.py)

### 5. 🎯 [Recommendation System](file:///e:/Downloads/ML_only/src/recommendation/README.md)
- **Domain**: E-Commerce & Content Discovery
- **Learning Objective**: Sparse matrix decomposition via TruncatedSVD, latent embedding spaces, and cold-start fallback handling.
- **Source Code**: [`src/recommendation/pipeline.py`](file:///e:/Downloads/ML_only/src/recommendation/pipeline.py)

### 6. 📊 [Demand Forecasting](file:///e:/Downloads/ML_only/src/demand_forecasting/README.md)
- **Domain**: Supply Chain & Retail Inventory Management
- **Learning Objective**: Time-series autoregressive lag creation (`lag_1`, `lag_7`, `rolling_mean_7`), seasonality, and multi-step recursive horizons.
- **Source Code**: [`src/demand_forecasting/pipeline.py`](file:///e:/Downloads/ML_only/src/demand_forecasting/pipeline.py)

### 7. ⚙️ [Predictive Maintenance](file:///e:/Downloads/ML_only/src/predictive_maintenance/README.md)
- **Domain**: IIoT Industrial Manufacturing
- **Learning Objective**: Telemetry sensor anomaly scoring, Remaining Useful Life (RUL) estimation, and high-recall failure prediction.
- **Source Code**: [`src/predictive_maintenance/pipeline.py`](file:///e:/Downloads/ML_only/src/predictive_maintenance/pipeline.py)

### 8. 🩺 [Medical Diagnosis Support](file:///e:/Downloads/ML_only/src/medical_diagnosis/README.md)
- **Domain**: Healthcare & Clinical Decision Support (CDS)
- **Learning Objective**: Metabolic biomarker standardization, clinical probability calibration, and safety warning systems.
- **Source Code**: [`src/medical_diagnosis/pipeline.py`](file:///e:/Downloads/ML_only/src/medical_diagnosis/pipeline.py)

### 9. 💬 [Sentiment Analysis](file:///e:/Downloads/ML_only/src/sentiment_analysis/README.md)
- **Domain**: NLP / Brand Monitoring & Product Reviews
- **Learning Objective**: TF-IDF unigram/bigram feature extraction, text classification, and composite emotion scoring.
- **Source Code**: [`src/sentiment_analysis/pipeline.py`](file:///e:/Downloads/ML_only/src/sentiment_analysis/pipeline.py)

### 10. 📄 [Document Classification](file:///e:/Downloads/ML_only/src/document_classification/README.md)
- **Domain**: Enterprise Content Management & HR Operations
- **Learning Objective**: Multi-class text categorizer, sublinear TF-IDF scaling, and Naive Bayes keyword extraction.
- **Source Code**: [`src/document_classification/pipeline.py`](file:///e:/Downloads/ML_only/src/document_classification/pipeline.py)

### 11. 🔍 [Defect Detection](file:///e:/Downloads/ML_only/src/defect_detection/README.md)
- **Domain**: Computer Vision & Quality Assurance
- **Learning Objective**: Image patch texture feature extraction, fracture severity grading, and Quality Control pass/fail logic.
- **Source Code**: [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py)

### 12. 🧩 [Customer Segmentation](file:///e:/Downloads/ML_only/src/customer_segmentation/README.md)
- **Domain**: Strategic Marketing & Customer Analytics
- **Learning Objective**: Unsupervised $K$-Means clustering ($K=4$), 2D PCA visual projections, and persona profiling.
- **Source Code**: [`src/customer_segmentation/pipeline.py`](file:///e:/Downloads/ML_only/src/customer_segmentation/pipeline.py)
