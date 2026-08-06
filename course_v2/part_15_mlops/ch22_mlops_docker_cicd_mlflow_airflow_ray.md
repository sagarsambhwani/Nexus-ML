# Chapter 22: Production MLOps: Docker, CI/CD, MLflow, Airflow, and Ray

---

## 1. Big Picture

Building a working ML model in a Jupyter Notebook is easy. Deploying that model into a resilient, scalable, self-healing production environment that automatically handles model versioning, experiment tracking, data drift monitoring, containerization, and distributed retraining is **MLOps (Machine Learning Operations)**.

This chapter covers Docker containerization, Git CI/CD pipelines, MLflow experiment tracking, Apache Airflow workflow orchestration, DVC data versioning, Ray distributed computing, and Evidently AI drift monitoring.

---

## 2. Intuition

- **Docker**: Packaging your Python code, PyTorch dependencies, CUDA drivers, and model binaries into an isolated lightweight container image. "Works on my machine" becomes "Works on any server everywhere."
- **MLflow**: A digital lab notebook that automatically records exact hyper-parameters, training loss metrics, code commit hashes, and model binary artifacts for every training run.
- **Airflow**: A master orchestra conductor scheduling complex multi-step pipelines (extract SQL $\to$ clean data $\to$ train model $\to$ evaluate $\to$ deploy) on recurring cron schedules.

---

## 3. Visualization

```text
Enterprise MLOps Automation Loop:

  Code Push (Git) ──► [ GitHub Actions CI/CD ] ──► Run Unit Tests & Linting
                                                          │
                                                          ▼
  Scheduled Cron  ──► [ Airflow Orchestrator ] ──► Execute Pipeline in Docker
                                                          │
                                                          ▼
  Model Training  ──► [ MLflow Tracking Server ] ◄── Log Metrics, Artifacts & Seeds
                                                          │
                                                          ▼
  Registry Model  ──► [ Kubernetes Deployment ] ──► FastAPI Microservice + Monitoring
```

---

## 4. Mathematics

### Population Stability Index (PSI) for Data Drift
To measure feature distribution shift between Baseline $B$ (Training) and Target $T$ (Serving) across $K$ bins:

$$\text{PSI} = \sum_{b=1}^K \left( T_b - B_b \right) \times \ln\left( \frac{T_b}{B_b} \right)$$

- $\text{PSI} < 0.10$: No significant data drift. Safe to serve.
- $0.10 \le \text{PSI} \le 0.25$: Moderate drift. Alert operations.
- $\text{PSI} > 0.25$: Severe Data Drift! Trigger automated model retraining pipeline!

---

## 5. Python (From Scratch Population Stability Index Calculator)

```python
import numpy as np

def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins=10) -> float:
    """Calculates Population Stability Index (PSI) to detect data drift."""
    # Determine quantile bin boundaries from baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(baseline, quantiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    
    # Calculate bin proportions
    b_counts, _ = np.histogram(baseline, bins=bins)
    t_counts, _ = np.histogram(target, bins=bins)
    
    b_props = b_counts / len(baseline)
    t_props = t_counts / len(target)
    
    # Handle zero division via Laplace smoothing
    b_props = np.where(b_props == 0, 1e-4, b_props)
    t_props = np.where(t_props == 0, 1e-4, t_props)
    
    # PSI Formula
    psi = np.sum((t_props - b_props) * np.log(t_props / b_props))
    return float(psi)

# Test Data Drift Detection
np.random.seed(42)
train_dist = np.random.normal(loc=100.0, scale=15.0, size=5000)
serving_no_drift = np.random.normal(loc=100.5, scale=15.0, size=1000)
serving_heavy_drift = np.random.normal(loc=120.0, scale=20.0, size=1000)

print(f"PSI (No Drift):    {calculate_psi(train_dist, serving_no_drift):.4f} (OK)")
print(f"PSI (Heavy Drift): {calculate_psi(train_dist, serving_heavy_drift):.4f} (ALERT: RETRAIN REQUIRED!)")
```

---

## 6. Production Dockerfile Configuration

```dockerfile
# Multi-stage production build for FastAPI + PyTorch model service
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 7. Under the Hood

- `docker build` caches image layers based on content hashes. Ordering instructions so `COPY requirements.txt .` and `pip install` occur *before* copying full source code prevents re-downloading heavy Python dependencies when only source code files change!

---

## 8. Engineering Perspective

- **Reproducibility Stack**:
  1. *Code*: Git commit hash.
  2. *Environment*: Docker image hash (`sha256:...`).
  3. *Data*: DVC content hash digest (`dvc.lock`).
  4. *Parameters*: MLflow run ID.

---

## 9. Common Mistakes

1. **Hardcoding Credentials in Docker Images**: Baking AWS keys or database passwords directly inside Dockerfiles. Use Environment Variables (`.env`) or Kubernetes Secrets.
2. **Missing Health Checks**: Deploying microservices without `/health` endpoints, causing load balancers to route live user traffic to un-initialized instances.

---

## 10. Interview Questions

### Q1: What is the difference between Data Drift and Concept Drift?
**Answer**: **Data Drift** (Covariate Shift) occurs when input feature distribution changes over time $P(X_{serving}) \neq P(X_{train})$, but the true underlying mapping $P(Y \mid X)$ remains unchanged. **Concept Drift** occurs when the relationship between features and target changes $P(Y \mid X_{serving}) \neq P(Y \mid X_{train})$ (e.g. consumer purchasing habits change after macro-economic events).

---

## 11. Exercises

1. **Coding**: Write a Python script using `mlflow` that logs hyper-parameters, metrics, and ROC curve artifacts.
2. **DevOps**: Write a GitHub Actions `.github/workflows/main.yml` file that runs `pytest` and builds a Docker container on push to main branch.

---

## 12. Mini Project: Automated Retraining Orchestrator

Write a Python script `mlops_monitor.py` that monitors serving API logs, computes PSI data drift daily, and automatically triggers MLflow model retraining when drift exceeds threshold $\text{PSI} > 0.20$.

---

## 13. Capstone Integration

Configured across root `Dockerfile`, `docker-compose.yml`, `api/main.py`, and `src/` pipeline artifacts.
