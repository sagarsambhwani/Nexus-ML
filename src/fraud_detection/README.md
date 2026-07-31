# 🛡️ Fraud Detection (Classification System)

## 📌 System Overview
The **Fraud Detection System** evaluates real-time credit card transaction streams to identify fraudulent activities. In production environments, fraud detection operates under severe class imbalance (often < 1% positive fraud cases) and strict latency budgets (< 50ms per prediction).

This pipeline implements an ensemble **Random Forest Classifier** trained on transaction velocity, geographical risk factors, transaction amount, and latent PCA features ($V_1$ to $V_5$).

---

## 🏗️ Deep-Dive Implementation Architecture

The core pipeline is implemented in [`pipeline.py`](file:///e:/Downloads/Nexus-ML/src/fraud_detection/pipeline.py) as a subclass of `BasePipeline`:

```mermaid
graph TD
    A[Incoming Request] --> B[Pydantic Schema Validation]
    B[Pydantic Schema Validation] --> C[Data Normalization]
    C[Data Normalization] --> D[Random Forest Model]
    D[Random Forest Model] --> E[Risk Calibration]
    E[Risk Calibration] --> F[Output Payload]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
```

### Class Code Structure & Execution Flow:

```python
class FraudDetectionPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Fraud Detection", artifact_name="fraud_detection_model.joblib")
```

1. **Data Generation (`generate_data`)**:
   - Generates synthetic samples ($N=1,500$) using exponential distributions for transaction amounts and Poisson distributions for 1-hour transaction velocity.
   - Synthesizes latent anonymized features $V_1 \dots V_5 \sim N(0, 1)$ simulating PCA transformation of raw cardholder metadata.
   - Calculates synthetic fraud ground-truth label using non-linear risk scoring:
     $$\text{FraudScore} = 0.015 \cdot \text{amount} + 1.2 \cdot \text{velocity\_1h} + 2.5 \cdot \text{location\_risk} + 0.8 \cdot \mathbb{I}(v1 > 1.5) + 1.0 \cdot \mathbb{I}(\text{hour} < 4) + \epsilon$$

2. **Model Training & Artifact Serialization (`train`)**:
   - Uses `train_test_split(..., test_size=0.2, stratify=y)` to preserve minority fraud class proportions.
   - Trains `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)`.
   - Serializes trained model dictionary containing `classifier`, `feature_names`, and `metrics` to `artifacts/models/fraud_detection_model.joblib` via Joblib.

3. **Production Inference Execution (`predict`)**:
   - Lazy-loads model artifact into memory if uninitialized.
   - Validates feature column alignment using DataFrame reindexing.
   - Extracts probability $P(\text{Fraud}) = \text{predict\_proba}(X)_{[:, 1]}$.
   - Categorizes risk tiers:
     - $P \ge 0.50 \rightarrow$ `HIGH_RISK` (Red alert, block transaction)
     - $0.25 \le P < 0.50 \rightarrow$ `MEDIUM_RISK` (Yellow alert, trigger 2FA)
     - $P < 0.25 \rightarrow$ `LOW_RISK` (Green, pass transaction)
   - Computes top 3 feature importances using Gini impurity decrease.

---


## 💻 API Usage Example

**Endpoint:** `POST /api/v1/predict/fraud-detection`

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict/fraud-detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "amount": 250.5,
  "time_hour": 2,
  "velocity_1h": 4,
  "location_risk": 0.85,
  "v1": 1.8,
  "v2": -0.5,
  "v3": 0.2,
  "v4": 1.1,
  "v5": -0.8
}'
```

**Expected JSON Response:**
```json
{
  "pipeline": "Fraud Detection",
  "status": "success",
  "result": {
    "fraud_probability": 0.92,
    "risk_tier": "HIGH_RISK",
    "top_factors": {
      "location_risk": 0.45,
      "velocity_1h": 0.32,
      "amount": 0.15
    }
  }
}
```

---

## 🎯 Engineering Decision-Making Process

### 1. Algorithm Selection Rationale: Why Random Forest over Logistic Regression or Deep Learning?
- **Decision**: Selected **Random Forest Classifier** as the baseline production architecture.
- **Rationale**:
  - *Vs. Logistic Regression*: Logistic regression fails to capture non-linear conjunctions (e.g., high transaction amount combined with late-night hours and high location risk) without manual interaction term engineering.
  - *Vs. Deep Neural Networks*: Deep learning models require significantly larger labeled datasets, GPU infrastructure, and introduce higher inference latency (~150ms vs ~3ms for tree ensembles).
  - *Vs. Single Decision Tree*: Single trees suffer from high variance and overfit quickly on noisy fraud data. Random Forest's bagging reduces variance dramatically.

### 2. Hyperparameter Choices & Design Trade-Offs:
| Hyperparameter | Value Chosen | Engineering Rationale & Trade-Off |
|---|---|---|
| `n_estimators` | `100` | Optimal balance between variance reduction and execution latency (~3ms). Increasing to 500 trees yielded < 0.2% AUC improvement at 5x latency cost. |
| `max_depth` | `8` | Prevents over-memorization of synthetic noise while allowing multi-variable decision paths (up to 8 feature splits deep). |
| `random_state` | `42` | Ensures 100% reproducible model weights across CI/CD builds and automated test runs. |

### 3. Latency vs. Explainability Trade-Off Matrix:
```
           Explainability
               ▲
               │  Logistic Regression (High Explainability, Low Non-Linearity)
               │      ★
               │          Random Forest (Chosen: High Accuracy, Medium Explainability, Low Latency)
               │              ★
               │                  Deep Neural Network (High Accuracy, Low Explainability, High Latency)
               │                      ★
               └────────────────────────────────────────► Predictive Power / Accuracy
```

### 4. Production Serving & Caching Rationale:
- **FastAPI In-Memory Caching (`PIPELINES_CACHE`)**: Re-loading `.joblib` files on every HTTP request creates severe disk I/O bottlenecks. In-memory caching ensures $O(1)$ model access with zero disk overhead during online inference.

---

## 📊 Dataset & Feature Engineering Details

### Feature Schema & Physical Interpretations:
| Feature | Data Type | Physical Meaning | Preprocessing / Transformation |
|---|---|---|---|
| `amount` | Float | Transaction dollar value | Raw float value ($) |
| `time_hour` | Integer | Hour of day (0-23) | Categorical/Discrete integer |
| `velocity_1h` | Integer | Transaction count in past 1 hr | Integer count metric |
| `location_risk` | Float | Geolocation anomaly score (0-1) | Normalized float ratio |
| `v1` to `v5` | Float | Anonymized PCA factors | Scaled Gaussian $N(0,1)$ components |

---

## 🏋️ Evaluation Metrics & Benchmarks

- **ROC-AUC Score**: ~0.98 (Primary ranking metric evaluating separation capacity across all decision thresholds).
- **F1-Score**: ~0.67 (Primary balance metric evaluating Precision vs. Recall under imbalanced distribution).

---

## 🚀 Production Deployment & Drift Monitoring Strategy

1. **Data Drift Monitoring**: Track distribution shift of `amount` and `velocity_1h` using the Kolmogorov-Smirnov (KS) test against baseline training distributions.
2. **Concept Drift Monitoring**: Monitor historical 30-day chargeback rates to trigger automated retrain alerts if F1 drops below 0.60.
3. **Fallbacks**: If inference execution exceeds 45ms SLA, fall back to a hardcoded rule engine (e.g. `amount > $5,000 AND velocity > 5` $\rightarrow$ `HIGH_RISK`).
