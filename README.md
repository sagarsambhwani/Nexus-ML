# ⚡ Nexus ML — Enterprise Production Machine Learning Suite

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Build](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)

> **12 Production-Ready Machine Learning & Artificial Intelligence Use Cases in ONE Unified Monorepo.**
> 
> Spanning Classification, Regression, Recommendation Systems, Time-Series Demand Forecasting, Predictive Telemetry, Healthcare Diagnostics, Natural Language Processing (NLP), Computer Vision, and Unsupervised Clustering.

---

## 🌟 Features Overview

- 🤖 **12 End-to-End ML Pipelines**: Self-contained dataset generators, feature engineering, model training, and serialization in `nexus_ml/`.
- ⚡ **Production REST API (FastAPI)**: High-performance async microservices API layer with Pydantic request/response schemas.
- 🎨 **Interactive Web Dashboard**: Clean light glassmorphism UI dashboard with dynamic form rendering, high-risk & normal presets, JSON inspector, and visual status meters.
- 📦 **Docker Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` configuration.
- 🔄 **CI/CD Pipeline**: GitHub Actions workflow for automated model training, Pytest execution, and Docker build verification.
- 🧪 **Comprehensive Automated Testing**: Unit and integration test suite (`pytest`) covering 100% of pipeline models and API endpoints.

---

## 📑 12 Production ML Pipelines Summary

| # | Domain / Topic | Machine Learning Task | Model Architecture & Feature Engineering | Primary Outputs |
|---|---|---|---|---|
| 1 | [**Fraud Detection**](nexus_ml/src/fraud_detection/README.md) | Classification | Random Forest / XGBoost + Velocity & Anomaly features | Fraud Prob, Alert Status, Risk Factors |
| 2 | [**Credit Risk Prediction**](nexus_ml/src/credit_risk/README.md) | Classification | Calibrated Logistic Scorecard + Standard Scaler | Default Prob, Risk Tier (AAA-CCC), Decision |
| 3 | [**Customer Churn**](nexus_ml/src/customer_churn/README.md) | Classification | Gradient Boosting + Tenure, Support Tickets, Contract | Churn Prob, Retention Action |
| 4 | [**House Price Valuation**](nexus_ml/src/house_prices/README.md) | Regression | Gradient Boosting + SqFt, Geo Score, Age, Amenities | Predicted Price ($), Price Bounds |
| 5 | [**Recommendation System**](nexus_ml/src/recommendation/README.md) | Hybrid Filter | TruncatedSVD Matrix Factorization + Content Filters | Top-K Items, Match Score %, Reason |
| 6 | [**Demand Forecasting**](nexus_ml/src/demand_forecasting/README.md) | Time Series | Lag Feature Regressor + Day of Week, Seasonality | Daily Forecast Units, Confidence Bounds |
| 7 | [**Predictive Maintenance**](nexus_ml/src/predictive_maintenance/README.md) | Telemetry / Class. | Random Forest + Sensor Vibration, Temp, Pressure, RPM | Failure Prob, RUL Hours, Status |
| 8 | [**Medical Diagnosis Support**](nexus_ml/src/medical_diagnosis/README.md) | Healthcare Class. | Calibrated Logistic Classifier + Biomarkers | Disease Risk, Elevated Biomarkers, Guidance |
| 9 | [**Sentiment Analysis**](nexus_ml/src/sentiment_analysis/README.md) | NLP Classification | TF-IDF (Unigram/Bigram) + Logistic Classifier | Sentiment (Pos/Neu/Neg), Score |
| 10 | [**Document Classification**](nexus_ml/src/document_classification/README.md) | NLP Multi-class | TF-IDF + Multinomial Naive Bayes / Softmax | Doc Category, Keywords Extracted |
| 11 | [**Defect Detection**](nexus_ml/src/defect_detection/README.md) | Computer Vision | Visual Patch Texture & Anomaly Feature Classifier | Defect Type, Severity Grade, QC Status |
| 12 | [**Customer Segmentation**](nexus_ml/src/customer_segmentation/README.md) | Clustering | StandardScaler + K-Means (K=4) + PCA 2D Mapper | Cluster ID, Persona Name, Strategy |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone repository
git clone https://github.com/sagarsambhwani/Nexus-ML.git
cd Nexus-ML

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
python nexus_ml/scripts/train_all.py
```
*Outputs serialized `.joblib` model artifacts to `nexus_ml/artifacts/models/` directory.*

### 3. Launch Unified Server (Automatic Free Port Selection)

Run the unified application with a single command:
```bash
python run.py
```

Features of `python run.py`:
- 🔍 **Dynamic Port Finder**: Automatically finds an available free port (starts at 8000; if busy, automatically binds to the next available port).
- 🚀 **Unified Monolith**: Hosts ML Pipelines, Course Academy (V1 & V2), and Dashboard UI together on ONE port.
- 🌐 **Interactive Dashboard**: `http://127.0.0.1:<PORT>/static/index.html`
- 🎓 **Course Academy UI**: `http://127.0.0.1:<PORT>/static/course.html`
- 📖 **OpenAPI Docs**: `http://127.0.0.1:<PORT>/docs`

*(Optional)* You can also pass custom host/port arguments if desired:
```bash
python run.py --port 8080 --reload
```

---

## 🐳 Docker Deployment

### Run Both Services with Docker Compose (Recommended)
```bash
docker-compose up --build -d
```
This starts **two containers**:

| Container | Port | URL |
|---|---|---|
| `ml_service` | 8000 | [http://localhost:8000](http://localhost:8000) — ML Dashboard |
| `course_service` | 8001 | [http://localhost:8001](http://localhost:8001) — Course Academy |

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
├── nexus_ml/                          # Nexus Machine Learning Suite Package
│   ├── config.py                      # Package configuration & artifact paths
│   ├── README.md                      # ML Suite documentation
│   ├── src/                           # 12 ML Pipeline Modules
│   │   ├── common/                    # Shared base classes & utilities
│   │   ├── fraud_detection/           # Pipeline 1: Fraud Detection
│   │   ├── credit_risk/               # Pipeline 2: Credit Risk
│   │   ├── customer_churn/            # Pipeline 3: Customer Churn
│   │   ├── house_prices/              # Pipeline 4: House Price Valuation
│   │   ├── recommendation/            # Pipeline 5: Recommendation System
│   │   ├── demand_forecasting/        # Pipeline 6: Demand Forecasting
│   │   ├── predictive_maintenance/    # Pipeline 7: Predictive Maintenance
│   │   ├── medical_diagnosis/         # Pipeline 8: Medical Diagnosis
│   │   ├── sentiment_analysis/        # Pipeline 9: Sentiment Analysis
│   │   ├── document_classification/   # Pipeline 10: Document Classification
│   │   ├── defect_detection/          # Pipeline 11: Defect Detection
│   │   └── customer_segmentation/     # Pipeline 12: Customer Segmentation
│   ├── artifacts/                     # Serialized model joblibs & data
│   │   └── models/
│   ├── course/                        # 100 Markdown curriculum chapter files (V1)
│   ├── course_v2/                     # 20-Part Engineering Curriculum (V2)
│   └── scripts/                       # Model training automation
│       └── train_all.py
├── api/                               # Production REST API Gateway
│   ├── main.py                        # Unified monolith API entry point
│   ├── routes.py                      # Aggregated router
│   ├── schemas.py                     # Pydantic validation schemas (all 12 models)
│   ├── ml_service/                    # ML Dashboard Microservice
│   └── course_service/                # Course Academy Microservice
├── dashboard/                         # Visual Web Application UI
│   ├── index.html                     # ML Dashboard SPA
│   ├── course.html                    # Course Academy SPA
│   ├── styles.css                     # Shared light design system
│   ├── app.js                         # ML Dashboard controller
│   └── course_app.js                  # Course Academy controller
├── tests/                             # Pytest test suite
│   ├── test_pipelines.py              # ML pipeline unit tests
│   ├── test_api.py                    # Unified REST API integration tests
│   ├── test_microservices.py          # Microservice-specific integration tests
│   └── test_port_utils.py             # Port utility unit tests
├── Dockerfile                         # Production container build
├── docker-compose.yml                 # Multi-service orchestration
├── requirements.txt                   # Production dependencies
├── run.py                             # Server runner
└── README.md                          # Repository documentation
```

---

## 🛡️ License & Credits
Developed as an enterprise-grade reference architecture for multi-model AI deployment. Licensed under MIT.
