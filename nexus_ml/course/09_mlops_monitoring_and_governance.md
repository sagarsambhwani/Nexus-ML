# ⚙️ Chapter 9: MLOps, Data Drift & Model Monitoring

## 9.1 Introduction to MLOps
Machine Learning Operations (MLOps) combines Machine Learning, Software Engineering, and DevOps to automate the continuous deployment, monitoring, and governance of production ML systems.

```
                      MLOps Continuous Lifecycle Loop
                               ┌──────────────┐
                               │  Data Prep   │
                               └──────┬───────┘
                                      │
                                      ▼
    ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
    │  Monitoring  │ ◄─────────│  Deployment  │ ◄─────────│  Model Fit   │
    │  & Alerts    │           │  & Inference │           │  & Evaluation│
    └──────┬───────┘           └──────────────┘           └──────────────┘
           │                                                     ▲
           └────────────────── Trigger Retraining ───────────────┘
```

---

## 9.2 Data Drift vs. Concept Drift

Model performance degrades over time in production due to environmental distribution shifts.

```
                            Distribution Shifts
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
       Data Drift                                        Concept Drift
P(X) changes while P(Y|X) holds                    P(Y|X) changes while P(X) holds
(e.g., User demographic shift)                    (e.g., Inflation alters pricing risk)
```

### 1. Data Drift (Covariate Shift)
- **Definition**: The input feature distribution $P(X)$ changes over time, while the conditional target probability $P(Y \mid X)$ remains unchanged.
- **Example**: A credit risk model trained on applicants aged 30-50 receives a marketing surge of applicants aged 18-25.

### 2. Concept Drift
- **Definition**: The conditional relationship $P(Y \mid X)$ changes, meaning the true relationship between features and target shifts.
- **Example**: Macroeconomic interest rate hikes alter default rates for borrowers with identical credit scores.

---

## 9.3 Statistical Drift Detection Math

### 1. Kolmogorov-Smirnov (KS) Test (Continuous Features)
Compares empirical cumulative distribution functions (eCDF) $F_1(x)$ (baseline training) and $F_2(x)$ (production inference):

$$D = \sup_x |F_1(x) - F_2(x)|$$

If $D > D_{\text{critical}}$ (or $p$-value $< 0.05$), statistically significant data drift is detected.

```python
from scipy.stats import ks_2samp

# Compare training baseline vs production feature distribution
stat, p_value = ks_2samp(X_train["amount"], X_production["amount"])
if p_value < 0.05:
    print(f"ALERT: Data drift detected in feature 'amount' (p-value={p_value:.4f})")
```

### 2. Population Stability Index (PSI)
Quantifies distribution shift across binned feature buckets $k \in [1, B]$:

$$\text{PSI} = \sum_{k=1}^B \left( \% \text{Actual}_k - \% \text{Expected}_k \right) \times \ln\left( \frac{\% \text{Actual}_k}{\% \text{Expected}_k} \right)$$

#### PSI Action Thresholds:
- $\text{PSI} < 0.10$: No significant distribution change.
- $0.10 \le \text{PSI} < 0.25$: Moderate drift; flag for review.
- $\text{PSI} \ge 0.25$: Severe drift; **trigger automated model retraining**.

---

## 9.4 Production Deployment Strategies

```
                            Deployment Strategies
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   Blue/Green Deploy                Canary Deploy                Shadow Deploy
(Instant Traffic Switch)       (Gradual Traffic Rollout)    (Parallel Silent Verification)
```

### 1. Blue/Green Deployment
Maintains two identical production environments: Blue (current live v1.0) and Green (new candidate v2.0). Traffic router switches 100% of user traffic to Green instantly upon verification.

### 2. Canary Deployment
Routes a small percentage of production requests (e.g. 5%) to new candidate model v2.0. If error rates and latencies remain stable, traffic is incrementally shifted (5% $\rightarrow$ 25% $\rightarrow$ 100%).

### 3. Shadow (Silent) Deployment
Routes incoming live requests to **both** model v1.0 (serves user response) and model v2.0 (calculates prediction silently in background). Logs and metrics are compared without exposing users to candidate model risk.

---

## 9.5 Automated Monitoring Architecture with Prometheus & FastAPI

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
import time

app = FastAPI()

# Prometheus Metrics Metrics Definition
INFERENCE_COUNT = Counter("model_inference_total", "Total inference requests", ["model_name", "status"])
INFERENCE_LATENCY = Histogram("model_latency_seconds", "Inference execution latency", ["model_name"])

@app.post("/predict")
def predict(payload: dict):
    start_time = time.time()
    try:
        # Perform prediction...
        result = {"status": "success"}
        INFERENCE_COUNT.labels(model_name="fraud_detection", status="success").inc()
        return result
    except Exception as e:
        INFERENCE_COUNT.labels(model_name="fraud_detection", status="error").inc()
        raise e
    finally:
        INFERENCE_LATENCY.labels(model_name="fraud_detection").observe(time.time() - start_time)

# Expose Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

## ⚓ Repository Code Reference
- See [`api/routes.py`](file:///e:/Downloads/ML_only/api/routes.py) for production health checks and metadata endpoint structures.
- See [`config.py`](file:///e:/Downloads/ML_only/config.py) for global model artifact directory paths.
