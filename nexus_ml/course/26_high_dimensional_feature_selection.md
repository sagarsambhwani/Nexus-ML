# 🔬 Chapter 26: High-Dimensional Feature Selection & Sparse Representations

## 26.1 The Feature Selection Imperative
In high-dimensional datasets ($p \gg N$), carrying uninformative or redundant features increases training time, induces overfitting, and degrades model interpretability. Feature selection isolates the optimal feature subset $\mathcal{S}^* \subset \mathcal{F}$ maximizing predictive signal.

```
                            Feature Selection Methods
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
Filter Methods                   Wrapper Methods                 Embedded Methods
(Statistical Scoring)           (Iterative Model Evaluation)    (Built-in Regularization)
• Chi-Square χ²                  • Recursive Feature Elim (RFE) • Lasso L1 Sparsity
• Mutual Information             • Sequential Selection          • Boruta Shadow Features
```

---

## 26.2 Filter Methods

Filter methods score features independently of model training using statistical hypothesis testing and information theory.

### 1. Chi-Square ($\chi^2$) Test (Categorical Features)
Measures independence between categorical feature $X$ and class target $Y$:

$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{i,j} - E_{i,j})^2}{E_{i,j}}$$

where $O_{i,j}$ is the observed frequency and $E_{i,j} = \frac{n_i \cdot n_j}{N}$ is the expected frequency under $H_0$ (independence).

### 2. Mutual Information (MI) Score
Quantifies non-linear information shared between feature $X$ and target $Y$:

$$I(X; Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log \left( \frac{P(x, y)}{P(x) P(y)} \right) = H(Y) - H(Y \mid X)$$

If $I(X; Y) = 0$, $X$ and $Y$ are completely independent.

---

## 26.3 Wrapper Methods: RFE & Boruta

### 1. Recursive Feature Elimination (RFE)
Iteratively trains an estimator, ranks features by feature importance weights $|w_j|$, and prunes the lowest-ranked $k$ features per iteration until target subset size is reached.

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Cross-validated Recursive Feature Elimination
selector = RFECV(estimator=RandomForestClassifier(), step=1, cv=5, scoring="roc_auc")
selector.fit(X_train, y_train)

# Optimal features mask
selected_features = X_train.columns[selector.support_]
```

### 2. Boruta Algorithm
Boruta creates **shadow features** (permuted copies of original features) to construct a noise benchmark. It trains a Random Forest and retains only features whose importance score significantly exceeds the maximum shadow feature importance using a $Z$-test.

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for feature importance ranking and velocity feature selection.
