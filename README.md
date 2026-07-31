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
python scripts/train_all.py
```
*Outputs serialized `.joblib` model artifacts to `artifacts/models/` directory.*

### 3. Launch Microservices

This project runs as **two independent microservices**.

#### ⚡ ML Dashboard Service (Port 8000)
```bash
uvicorn api.ml_service.main:app --host 0.0.0.0 --port 8000 --reload
```
- 🌐 **ML Dashboard**: [http://localhost:8000](http://localhost:8000)
- 📖 **ML API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

#### 🎓 Course Academy Service (Port 8001)
```bash
uvicorn api.course_service.main:app --host 0.0.0.0 --port 8001 --reload
```
- 🌐 **Course Academy UI**: [http://localhost:8001](http://localhost:8001)
- 📖 **Course API Docs**: [http://localhost:8001/docs](http://localhost:8001/docs)

> **Tip**: For legacy single-host mode, you can still boot both routers from one process:
> `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`

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

### Standalone Docker Container (ML Service only)
```bash
docker build -t nexus-ml:latest .
docker run -p 8000:8000 nexus-ml:latest
```

### Standalone Docker Container (Course Service)
```bash
docker build -t nexus-ml:latest .
docker run -p 8001:8001 nexus-ml:latest uvicorn api.course_service.main:app --host 0.0.0.0 --port 8001
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
├── docker-compose.yml                 # Multi-service orchestration (ml-service + course-service)
├── requirements.txt                   # Pinned production dependencies
├── README.md                          # Repository documentation
├── .github/
│   └── workflows/
│       └── ci-cd.yml                  # CI/CD automated pipeline
├── src/                               # 12 ML Pipeline Modules
│   ├── common/                        # Shared base classes & utilities
│   ├── fraud_detection/               # Pipeline 1: Fraud Detection
│   ├── credit_risk/                   # Pipeline 2: Credit Risk
│   ├── customer_churn/                # Pipeline 3: Customer Churn
│   ├── house_prices/                  # Pipeline 4: House Price Valuation
│   ├── recommendation/                # Pipeline 5: Recommendation System
│   ├── demand_forecasting/            # Pipeline 6: Demand Forecasting
│   ├── predictive_maintenance/        # Pipeline 7: Predictive Maintenance
│   ├── medical_diagnosis/             # Pipeline 8: Medical Diagnosis
│   ├── sentiment_analysis/            # Pipeline 9: Sentiment Analysis
│   ├── document_classification/       # Pipeline 10: Document Classification
│   ├── defect_detection/              # Pipeline 11: CV Defect Detection
│   └── customer_segmentation/         # Pipeline 12: Customer Segmentation
├── api/                               # Production REST API
│   ├── main.py                        # Legacy unified entry point (monolith)
│   ├── routes.py                      # Unified router (aggregates both microservices)
│   ├── schemas.py                     # Pydantic validation schemas (all 12 models)
│   ├── ml_service/
│   │   ├── main.py                    # ML Dashboard Microservice entry point (Port 8000)
│   │   └── routes.py                  # ML inference + pipeline status endpoints
│   └── course_service/
│       ├── main.py                    # Course Microservice entry point (Port 8001)
│       └── routes.py                  # Course curriculum & chapter endpoints
├── dashboard/                         # Visual Web Application UI
│   ├── index.html                     # ML Dashboard SPA (served on Port 8000)
│   ├── course.html                    # Course Academy SPA (served on Port 8001)
│   ├── styles.css                     # Shared glassmorphism design system
│   ├── app.js                         # ML Dashboard web UI controller
│   └── course_app.js                  # Course Academy web UI controller
├── course/                            # 101 Markdown curriculum chapter files
│   ├── README.md                      # Course index & volume overview
│   └── 01_data_cleaning_...md → 101_*.md  # Individual chapter files
├── scripts/                           # Automation scripts
│   └── train_all.py                   # Automated model training script
└── tests/                             # Pytest test suite
    ├── test_pipelines.py              # ML pipeline unit tests
    ├── test_api.py                    # Unified REST API integration tests
    └── test_microservices.py          # Microservice-specific integration tests
```

---

## 🛡️ License & Credits
Developed as an enterprise-grade reference architecture for multi-model AI deployment. Licensed under MIT.
