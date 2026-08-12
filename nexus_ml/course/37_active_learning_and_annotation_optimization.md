# 🎯 Chapter 37: Active Learning & Human-in-the-Loop Annotation Optimization

## 37.1 Active Learning Paradigm
In many real-world scenarios, unlabeled data is abundant, but expert annotation is expensive and time-consuming. **Active Learning** automatically identifies and queries human annotators (the Oracle) for labels on unannotated samples that maximize model performance improvement per labeled instance.

```
Unlabeled Data Pool ──► Active Learning Acquisition Function ──► Query Top-K Informative Samples ──► Human Oracle Annotates ──► Retrain Model
```

---

## 37.2 Uncertainty Sampling Strategies

Given model predicted probability vector $P(y \mid x)$ over $C$ classes:

### 1. Least Confidence Sampling
Queries samples where the model's top class prediction confidence is lowest:

$$U_{\text{LC}}(x) = 1 - \max_{c} P(y=c \mid x)$$

### 2. Margin Sampling
Queries samples where the difference between top two predicted class probabilities is smallest:

$$U_{\text{Margin}}(x) = 1 - \left( P(y = c_1 \mid x) - P(y = c_2 \mid x) \right)$$

### 3. Entropy Sampling
Queries samples with maximum Shannon Entropy (maximum uncertainty):

$$H(y \mid x) = -\sum_{c=1}^C P(y=c \mid x) \log_2 P(y=c \mid x)$$

---

## 37.3 Query-by-Committee (QBC) & Core-Set

```python
import numpy as np

def calculate_margin_uncertainty(probabilities):
    # Sort class probabilities descending per sample
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    # Margin = Top Prob - 2nd Top Prob
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    # Smallest margin indicates highest uncertainty
    return 1.0 - margin
```

---

## ⚓ Repository Code Reference
- See [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py) for quality inspection uncertainty scoring.
