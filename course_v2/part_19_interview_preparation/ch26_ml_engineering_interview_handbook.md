# Chapter 26: Machine Learning Engineering Interview Handbook

---

## 1. Big Picture

Securing a job as an ML Engineer at top-tier tech companies requires mastering 6 distinct interview rounds:
1. **Coding & Algorithms** (LeetCode, Data Structures, Vectorization)
2. **ML Theory & Fundamentals** (Math, Derivations, Algorithm Mechanics)
3. **Data Science & SQL** (Complex Window Functions, Data Manipulation)
4. **ML System Design** (End-to-End Production Architectures)
5. **Applied Case Studies** (Debugging, Metric Lift, Behavioral Scenarios)
6. **Behavioral & Leadership** (Project Trade-offs, Cross-Functional Alignment)

This chapter provides a comprehensive interview preparation handbook containing 30+ canonical interview questions, step-by-step solutions, and cheat sheets across all 6 rounds.

---

## 2. Theoretical Fundamentals Cheat Sheet

| Topic | Key Equation / Concept | Primary Failure Mode |
| :--- | :--- | :--- |
| **Bias-Variance** | $\text{MSE} = \text{Bias}^2 + \text{Variance} + \sigma^2$ | High Variance = Overfitting, High Bias = Underfitting |
| **L1 vs L2** | $L_1 = \lambda \|w\|_1$ (Sparse), $L_2 = \frac{\lambda}{2} \|w\|_2^2$ (Smooth) | Unscaled features distort regularization penalties |
| **ROC-AUC** | Area under TPR ($\frac{TP}{P}$) vs FPR ($\frac{FP}{N}$) curve | AUC is threshold-invariant and class-imbalance robust |
| **Cross-Entropy** | $\mathcal{L} = -\sum y_i \log(\hat{y}_i)$ | Log(0) numerical underflow; use `eps` clipping |
| **Precision / Recall** | $\text{Prec} = \frac{TP}{TP+FP}, \quad \text{Rec} = \frac{TP}{TP+FN}$ | High precision misses positives; high recall raises false alarms |

---

## 3. Top 10 ML Engineering Interview Questions & Solutions

### Q1: Implement K-Means Clustering from Scratch in Python.
```python
import numpy as np

def kmeans_scratch(X: np.ndarray, k=3, max_iters=100):
    # Randomly initialize centroids from data points
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        # 1. Compute Euclidean Distances: (N, 1, d) - (1, K, d) -> (N, K)
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 2. Update Centroids to cluster means
        new_centroids = np.array([X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j] for j in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return centroids, labels

# Test K-Means
X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
cents, lbls = kmeans_scratch(X_data, k=2)
print("Learned Cluster Centroids:\n", cents)
```

---

### Q2: What is the difference between Micro F1 and Macro F1 in multi-class classification?
**Answer**: 
- **Macro F1**: Calculates F1-score independently for each class and averages them unweighted:
$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^C \text{F1}_c$$
Gives equal weight to all classes, highlighting poor performance on rare minority classes.
- **Micro F1**: Sums global True Positives, False Positives, and False Negatives across all classes before computing F1. Equivalent to overall Accuracy in multi-class classification.

---

### Q3: How do you handle severe class imbalance (e.g. 99.9% negative, 0.1% positive)?
**Answer**:
1. **Metrics**: Stop using Accuracy! Use PR-AUC (Precision-Recall Area Under Curve), F-beta score ($\beta=2$ to prioritize recall), or Normalized Confusion Matrices.
2. **Resampling**: SMOTE (Synthetic Minority Over-sampling Technique) or Tomek Links undersampling.
3. **Loss Modification**: Focal Loss $\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ or class-weighted cross entropy (`scale_pos_weight` in XGBoost).
4. **Threshold Tuning**: Optimize prediction probability cutoffs using PR curves instead of default 0.50.

---

### Q4: Write a SQL Query to find the 2nd Highest Salary per Department using Window Functions.
```sql
WITH RankedSalaries AS (
    SELECT 
        department_id,
        employee_id,
        salary,
        DENSE_RANK() OVER(PARTITION BY department_id ORDER BY salary DESC) as rnk
    FROM employees
)
SELECT department_id, employee_id, salary
FROM RankedSalaries
WHERE rnk = 2;
```

---

### Q5: Explain the Vanishing Gradient problem and 3 architectural solutions.
**Answer**: Vanishing gradients occur when deep neural network gradients $\frac{\partial \mathcal{L}}{\partial W_l}$ shrink exponentially toward zero during backpropagation as activations pass through saturating non-linearities (Sigmoid/Tanh whose derivatives max at 0.25).
**3 Solutions**:
1. **ReLU / GELU Activations**: Derivatives remain constant ($1.0$) for positive inputs.
2. **Residual Connections (ResNet)**: Skip connections $h(x) = f(x) + x$ create gradient identity highways where $\frac{\partial h}{\partial x} = \frac{\partial f}{\partial x} + 1$.
3. **Batch / Layer Normalization**: Keeps layer inputs zero-centered with unit variance, preventing activation saturation.

---

## 4. Mock Interview System Design Checklist

- [ ] Clarify business KPIs (CTR, Revenue lift) vs non-functional constraints (Latency < 20ms, Availability 99.99%).
- [ ] Diagram streaming Kafka data ingestion vs offline batch storage (Delta Lake).
- [ ] Define Redis Online Feature Store & Feast Offline Store architecture.
- [ ] Separate Candidate Generation (Recall) from Heavy Ranking (Precision).
- [ ] Address cold start, data leakage, and training-serving skew.
- [ ] Define fallbacks, circuit breakers, and shadow deployment cutover.
- [ ] Include PSI data drift monitoring and automated retraining triggers.

---

## 5. Capstone Integration

Serves as the master interview review guide summarizing concepts across all 20 Parts of Course V2.
