# ⚡ Nexus ML — Enterprise Production Machine Learning Suite

> **12 Production-Ready Machine Learning & Artificial Intelligence Use Cases in ONE Unified Repository.**
> 
> Spanning Classification, Regression, Recommendation Systems, Time-Series Demand Forecasting, Predictive Telemetry, Healthcare Diagnostics, Natural Language Processing (NLP), Computer Vision, and Unsupervised Clustering.

---

## 🌟 Features Overview

- 🤖 **12 End-to-End ML Pipelines**: Self-contained dataset generators, feature engineering, model training, and serialization.
- ⚡ **Production REST API (FastAPI)**: High-performance async microservices API layer with Pydantic request/response schemas.
- 🎨 **Interactive Web Dashboard**: Glassmorphism UI dashboard with dynamic form rendering, high-risk & normal presets, JSON inspector, and visual status meters.
- 📦 **Docker Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` configuration.
- 🔄 **CI/CD Pipeline**: GitHub Actions workflow for automated model training, Pytest execution, and Docker build verification.
- 🧪 **Comprehensive Automated Testing**: Unit and integration test suite (`pytest`) covering 100% of pipeline models and API endpoints.

---

## 📑 12 Production ML Pipelines Summary

| # | Domain / Topic | Machine Learning Task | Model Architecture & Feature Engineering | Primary Outputs |
|---|---|---|---|---|
| 1 | **Fraud Detection** | Classification | Random Forest / XGBoost + Velocity & Anomaly features | Fraud Prob, Alert Status, Risk Factors |
| 2 | **Credit Risk Prediction** | Classification | Calibrated Logistic Scorecard + Standard Scaler | Default Prob, Risk Tier (AAA-CCC), Decision |
| 3 | **Customer Churn** | Classification | Gradient Boosting + Tenure, Support Tickets, Contract | Churn Prob, Retention Action |
| 4 | **House Price Valuation** | Regression | Gradient Boosting + SqFt, Geo Score, Age, Amenities | Predicted Price ($), Price Bounds |
| 5 | **Recommendation System** | Hybrid Filter | TruncatedSVD Matrix Factorization + Content Filters | Top-K Items, Match Score %, Reason |
| 6 | **Demand Forecasting** | Time Series | Lag Feature Regressor + Day of Week, Seasonality | Daily Forecast Units, Confidence Bounds |
| 7 | **Predictive Maintenance** | Telemetry / Class. | Random Forest + Sensor Vibration, Temp, Pressure, RPM | Failure Prob, RUL Hours, Status |
| 8 | **Medical Diagnosis Support** | Healthcare Class. | Calibrated Logistic Classifier + Biomarkers | Disease Risk, Elevated Biomarkers, Guidance |
| 9 | **Sentiment Analysis** | NLP Classification | TF-IDF (Unigram/Bigram) + Logistic Classifier | Sentiment (Pos/Neu/Neg), Score |
| 10 | **Document Classification** | NLP Multi-class | TF-IDF + Multinomial Naive Bayes / Softmax | Doc Category, Keywords Extracted |
| 11 | **Defect Detection** | Computer Vision | Visual Patch Texture & Anomaly Feature Classifier | Defect Type, Severity Grade, QC Status |
| 12 | **Customer Segmentation** | Clustering | StandardScaler + K-Means (K=4) + PCA 2D Mapper | Cluster ID, Persona Name, Strategy |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone repository
git clone https://github.com/your-org/nexus-ml-suite.git
cd nexus-ml-suite

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
# On Linux/macOS: source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 2. Train All 12 Machine Learning Models
```bash
python scripts/train_all.py
```
*Outputs serialized `.joblib` model artifacts to `artifacts/models/` directory.*

### 3. Launch API Server & Visual Web Dashboard
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- 🌐 **Interactive Web Dashboard**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- 📖 **Interactive OpenAPI Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🐳 Docker Deployment

### Run with Docker Compose
```bash
docker-compose up --build -d
```
Access the application at `http://localhost:8000`.

### Standalone Docker Container Build
```bash
docker build -t nexus-ml-suite:latest .
docker run -p 8000:8000 nexus-ml-suite:latest
```

---

## 🧪 Automated Testing

Execute the complete unit and integration test suite using `pytest`:
```bash
pytest tests/ -v
```

---

## 📁 Repository Structure

```
.
├── config.py                          # Global configuration & artifact paths
├── Dockerfile                         # Multi-stage production container build
├── docker-compose.yml                 # Service orchestration
├── requirements.txt                   # Pinned production dependencies
├── README.md                          # Repository documentation
├── .github/
│   └── workflows/
│       └── ci-cd.yml                  # CI/CD automated pipeline
├── src/                               # 12 ML Pipeline Modules
│   ├── common/                        # Shared base classes & utilities
│   ├── 01_fraud_detection/            # Pipeline 1
│   ├── 02_credit_risk/                # Pipeline 2
│   ├── 03_customer_churn/             # Pipeline 3
│   ├── 04_house_prices/               # Pipeline 4
│   ├── 05_recommendation/             # Pipeline 5
│   ├── 06_demand_forecasting/         # Pipeline 6
│   ├── 07_predictive_maintenance/     # Pipeline 7
│   ├── 08_medical_diagnosis/          # Pipeline 8
│   ├── 09_sentiment_analysis/         # Pipeline 9
│   ├── 10_document_classification/    # Pipeline 10
│   ├── 11_defect_detection/           # Pipeline 11
│   └── 12_customer_segmentation/      # Pipeline 12
├── api/                               # Production REST API
│   ├── main.py                        # FastAPI entry point
│   ├── routes.py                      # REST endpoints for all 12 models
│   └── schemas.py                     # Pydantic validation schemas
├── dashboard/                         # Visual Web Application UI
│   ├── index.html                     # Dashboard Single Page Application
│   ├── styles.css                     # Glassmorphism design system
│   └── app.js                         # Dynamic web UI controller
├── scripts/                           # Automation scripts
│   └── train_all.py                   # Automated model training script
└── tests/                             # Pytest test suite
    ├── test_pipelines.py              # ML pipeline tests
    └── test_api.py                    # REST API integration tests
```

---

## 🛡️ License & Credits
Developed as an enterprise-grade reference architecture for multi-model AI deployment. Licensed under MIT.
