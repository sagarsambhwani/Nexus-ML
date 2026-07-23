# 📓 Chapter 5: Advanced Tree Ensembles — XGBoost, LightGBM & CatBoost

## 5.1 Decision Tree Foundations
A Decision Tree recursively partitions feature space into axis-aligned rectangular regions $R_m$ to predict target outcomes.

```
                           Root Node: Income > $70k?
                                ╱             ╲
                              Yes              No
                              ╱                 ╲
                   Age > 45?               Tenure > 12?
                    ╱     ╲                 ╱        ╲
              High Risk  Low Risk     Medium Risk  Low Risk
```

### Splitting Criteria:
1. **Gini Impurity** (Classification):
   $$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
2. **Entropy & Information Gain** (Classification):
   $$H(p) = -\sum_{k=1}^K p_k \log_2(p_k)$$
   $$\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
3. **Variance Reduction / MSE** (Regression):
   $$\text{MSE}_m = \frac{1}{N_m} \sum_{i \in R_m} (y_i - \hat{y}_m)^2$$

---

## 5.2 Ensemble Philosophy: Bagging vs. Boosting

```
                          Ensemble Methods
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
     Bagging                                          Boosting
(Bootstrap Aggregating)                       (Gradient Descent in Function Space)
• Independent parallel trees                   • Sequential dependent trees
• Reduces Variance                              • Reduces Bias & Variance
• Example: Random Forest                        • Examples: XGBoost, LightGBM, CatBoost
```

---

## 5.3 Random Forest Mechanics
Random Forest builds $M$ independent decision trees trained on bootstrap samples of the dataset.

### Variance Reduction via Feature Decorrelation:
For $M$ trees with individual variance $\sigma^2$ and pairwise correlation $\rho$:

$$\text{Var}(\text{Ensemble}) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$

Random Forest randomly selects a subset $m = \sqrt{p}$ features at each split, reducing tree-to-tree correlation $\rho$ and driving total ensemble variance down.

---

## 5.4 XGBoost (Extreme Gradient Boosting)

XGBoost expands traditional Gradient Boosting by utilizing a **second-order Taylor expansion** of the loss function and incorporating explicit structural regularization.

### Mathematical Formulation:
At iteration $t$, objective function to minimize:

$$\mathcal{L}^{(t)} = \sum_{i=1}^N l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

where tree complexity regularization is:

$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

($T$ is leaf count, $w_j$ is leaf weight vector).

### 2nd-Order Taylor Approximation:
$$l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2$$

where 1st and 2nd order gradients are:
$$g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$$

### Optimal Leaf Weight & Split Gain:
Optimal score for split node with left gradient set $I_L$ and right gradient set $I_R$:

$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

---

## 5.5 LightGBM (Light Gradient Boosting Machine)

LightGBM achieves 10x-15x faster training speeds on massive datasets via two novel algorithms:

```
                                  LightGBM Innovations
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     Gradient-based One-Side Sampling (GOSS)     Exclusive Feature Bundling (EFB)
  • Keeps samples with large gradients          • Bundles sparse mutually exclusive
  • Randomly samples small gradient instances     features to reduce dimensionality
```

### Leaf-Wise Tree Growth:
Unlike traditional level-wise (depth-wise) tree growth, LightGBM uses **leaf-wise (best-first)** growth, splitting the leaf node that minimizes loss the most.

```
       Level-Wise Growth (XGBoost/RF)                 Leaf-Wise Growth (LightGBM)
                  ◯                                                ◯
               ╱     ╲                                          ╱     ╲
              ◯       ◯                                        ◯       ◯
             ╱ ╲     ╱ ╲                                              ╱ ╲
            ◯   ◯   ◯   ◯                                            ◯   ◯
    (Splits all nodes at current depth)                     (Splits single maximum-loss leaf)
```

---

## 5.6 CatBoost (Categorical Boosting)

CatBoost solves target leakage and categorical feature preprocessing via two innovations:

1. **Ordered Boosting**: Traditional boosting calculates gradients on samples used to train previous trees, causing target leakage. CatBoost calculates unbiased gradient estimates on historical permutations of the training set.
2. **Oblivious (Symmetric) Trees**: CatBoost builds symmetric decision trees where the same feature split is applied across all nodes at the same tree depth, enabling lightning-fast CPU inference via SIMD instructions.

---

## ⚖️ Comparative Decision Matrix across Tree Algorithms

| Metric / Feature | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| **Training Speed** | Medium | Fast | **Ultra Fast** | Fast |
| **Inference Speed**| Fast | Fast | Fast | **Ultra Fast (SIMD)** |
| **Categorical Handling** | Requires OHE/Ordinal | Requires Encoding | Integer Encoding | **Native Auto-Encoding** |
| **Small Datasets** | Excellent | **Excellent** | Risk of Overfitting | Excellent |
| **Massive Datasets** | Slow | Fast | **Best-in-Class** | Fast |
| **Overfitting Risk**| Very Low | Low (with $\gamma, \lambda$) | Medium (use `max_depth`) | **Very Low** |

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for Random Forest ensemble implementations.
- See [`src/customer_churn/pipeline.py`](file:///e:/Downloads/ML_only/src/customer_churn/pipeline.py) and [`src/house_prices/pipeline.py`](file:///e:/Downloads/ML_only/src/house_prices/pipeline.py) for Gradient Boosting implementations.
