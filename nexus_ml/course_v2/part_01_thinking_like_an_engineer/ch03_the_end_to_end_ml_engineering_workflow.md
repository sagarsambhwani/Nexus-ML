# Chapter 03: The End-to-End ML Engineering Workflow

---

## 1. Big Picture

Machine Learning in production is 10% model training and 90% engineering infrastructure. Novices spend all their time tweaking hyper-parameters in Jupyter Notebooks. Senior ML Engineers build **reproducible, scalable, and automated pipelines** that translate raw business goals into production microservices.

Every chapter in this 20-Part handbook maps directly to this 8-stage production lifecycle:

```text
Business Problem ➜ Data Collection ➜ Data Preprocessing ➜ Feature Engineering ➜ Model Training ➜ Model Evaluation ➜ Deployment ➜ Monitoring
```

---

## 2. Intuition

Building a production ML system is like running a commercial restaurant kitchen:
- **Business Problem**: Setting the menu based on customer demand.
- **Data Collection**: Sourcing raw ingredients from reliable suppliers.
- **Data Cleaning**: Washing, peeling, and discarding spoiled produce.
- **Feature Engineering**: Chopping, marinating, and prepping portions (Mise en place).
- **Model Training**: Cooking dishes according to precise recipes.
- **Evaluation**: Quality inspection before sending food to tables.
- **Deployment**: Serving dishes hot to hundreds of diners simultaneously.
- **Monitoring**: Collecting diner feedback, checking food safety, and tracking inventory drift.

---

## 3. Visualization

```text
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ 1. BUSINESS GOAL  │ ───► │ 2. DATA INGESTION │ ───► │  3. DATA CLEANING │
   │ (Define KPIs, SLA)│      │(SQL, Streams, S3) │      │(Impute, Dedupe)   │
   └───────────────────┘      └───────────────────┘      └───────────────────┘
                                                                   │
                                                                   ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ 6. EVALUATION     │ ◄─── │ 5. MODEL TRAINING │ ◄─── │ 4. FEATURE ENGR.  │
   │ (ROC-AUC, Latency)│      │(XGBoost, PyTorch) │      │(Scalers, Encoding)│
   └───────────────────┘      └───────────────────┘      └───────────────────┘
             │
             ▼
   ┌───────────────────┐      ┌───────────────────┐
   │ 7. DEPLOYMENT     │ ───► │ 8. MONITORING     │
   │ (FastAPI, Docker) │      │(Drift, Performance│
   └───────────────────┘      └───────────────────┘
```

---

## 4. Mathematics

Workflow optimization requires defining a explicit loss trade-off between **Statistical Utility** $\mathcal{S}$ and **System Latency / Cost** $\mathcal{C}$:

$$\max_{\theta, \text{arch}} U(\theta) = \alpha \cdot \text{Metric}(\theta, \mathcal{D}_{test}) - \beta \cdot \text{Latency}(\theta, \mathbf{x}) - \gamma \cdot \text{InfrastructureCost}(\theta)$$

Where:
- $\text{Metric}(\theta)$ is business utility (e.g., Precision@K, F1-Score, Expected Revenue gain).
- $\text{Latency}(\theta)$ is p99 inference response time (e.g. $< 20\text{ms}$).
- $\alpha, \beta, \gamma$ are business constraint weighting coefficients.

---

## 5. Python (From Scratch)

A minimal end-to-end Python pipeline class illustrating clean stage separation:

```python
import numpy as np

class MinimalMLPipeline:
    def __init__(self):
        self.w = None
        self.b = None
        self.mean = None
        self.std = None
        
    def stage_1_clean_data(self, X):
        """Impute NaNs with median column values."""
        X_clean = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        return X_clean
        
    def stage_2_engineer_features(self, X_clean, fit=True):
        """Standardize features: Z = (X - mean) / std."""
        if fit:
            self.mean = np.mean(X_clean, axis=0)
            self.std = np.std(X_clean, axis=0) + 1e-8
        return (X_clean - self.mean) / self.std
        
    def stage_3_train(self, X_prep, y, lr=0.01, epochs=100):
        """Train via Ridge-regularized Gradient Descent."""
        N, d = X_prep.shape
        self.w = np.zeros((d, 1))
        self.b = 0.0
        
        for _ in range(epochs):
            y_pred = X_prep @ self.w + self.b
            dw = (2/N) * X_prep.T @ (y_pred - y) + 0.1 * self.w
            db = (2/N) * np.sum(y_pred - y)
            self.w -= lr * dw
            self.b -= lr * db
            
    def stage_4_predict(self, X_raw):
        """Production Serving Pipeline."""
        X_clean = self.stage_1_clean_data(X_raw)
        X_prep = self.stage_2_engineer_features(X_clean, fit=False)
        return X_prep @ self.w + self.b

# Test Execution
X_raw = np.array([[100.0, np.nan], [150.0, 2.0], [200.0, 3.0]])
y_raw = np.array([[300.0], [450.0], [600.0]])

pipeline = MinimalMLPipeline()
X_c = pipeline.stage_1_clean_data(X_raw)
X_p = pipeline.stage_2_engineer_features(X_c, fit=True)
pipeline.stage_3_train(X_p, y_raw)

preds = pipeline.stage_4_predict(np.array([[120.0, 2.5]]))
print(f"Production Prediction: ${preds[0,0]:.2f}k")
```

---

## 6. Production Library (Scikit-Learn Pipeline)

Using Scikit-Learn's `Pipeline` object ensures zero data leakage during cross-validation:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Encapsulate all stages into a single serializable artifact
production_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])

production_pipeline.fit(X_raw, y_raw.ravel())
print("Pipeline Score:", production_pipeline.score(X_raw, y_raw.ravel()))
```

---

## 7. Under the Hood

- Scikit-Learn `Pipeline` objects call `.fit_transform()` on intermediate steps during training, but strictly `.transform()` on inference.
- Storing transformers inside a single pipeline binary (`.joblib` or `.pkl`) guarantees that validation data is processed using identical transformation parameters (mean, variance, encodings) computed from training data.

---

## 8. Engineering Perspective

- **Data Leakage**: Computing scalers or mean values on the entire dataset *before* train-test splitting leaks test statistics into training, leading to overly optimistic metric estimations.
- **Reproducibility**: Production pipelines must log random seeds, dependency versions (`requirements.txt`), and data hash digests (`dvc.lock`).

---

## 9. Common Mistakes

1. **Notebook Debt**: Keeping code inside unordered Jupyter notebooks without refactoring into clean Python modules (`src/`).
2. **Missing Input Validation**: Deploying models without checking schema types, ranges, or missing values at the API boundary.

---

## 10. Interview Questions

### Q1: What is data leakage and how do you prevent it in an ML workflow?
**Answer**: Data leakage occurs when information from outside the training dataset (such as target values or test distribution metrics) is inadvertently used to fit preprocessing transformations or models. Prevent it by splitting data strictly *before* any feature extraction, scaling, or imputation, and using pipeline encapsulation objects.

---

## 11. Exercises

1. **Coding**: Refactor `MinimalMLPipeline` to add stage-level execution time tracking with a `@timer_decorator`.
2. **Architecture**: Design a directory structure for a production ML codebase separating pipelines, APIs, artifacts, and tests.

---

## 12. Mini Project: End-to-End Pipeline Script

Write a script `run_pipeline.py` that loads a raw dataset, performs automated splitting, fits an imputer + scaler + XGBoost pipeline, evaluates ROC-AUC, and saves the trained pipeline artifact to disk using `joblib`.

---

## 13. Capstone Integration

Maps directly to `src/customer_churn/pipeline.py` which demonstrates an enterprise-grade pipeline structure with modular components, validation steps, and inference endpoints.
