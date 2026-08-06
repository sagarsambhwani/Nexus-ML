# Chapter 15: Tree Ensembles: Random Forests, XGBoost, LightGBM, and CatBoost

---

## 1. Big Picture

For tabular data, **Gradient Boosted Decision Trees (GBDT)** consistently dominate machine learning competitions and enterprise applications. While neural networks excel on raw perceptual grid data (images/text), GBDT models handle non-linear numerical/categorical boundaries, missing values, and varying feature scales with minimal preprocessing.

This chapter covers Decision Trees (CART, Gini Impurity, Variance Reduction), Bagging (Random Forest), and Gradient Boosting mathematics across **XGBoost**, **LightGBM**, and **CatBoost**.

---

## 2. Intuition

- **Decision Tree**: Asking a sequence of binary 20-questions ("Is Income > $50k?", "Is Credit Score > 700?") to partition feature space into hyper-rectangular axis-aligned boxes.
- **Random Forest (Bagging)**: Asking a committee of 100 independent expert trees trained on random bootstrap data subsets and random feature subsets. Their averaged vote dramatically reduces variance ($\text{Var} \to \frac{\text{Var}}{K}$).
- **Gradient Boosting (XGBoost)**: A sequential team of weak trees where tree #2 is trained specifically to fix the residual errors made by tree #1, tree #3 fixes residuals of tree #2, and so on.

---

## 3. Visualization

```text
Bagging (Random Forest) Parallel Architecture:
  Dataset ──► [ Bootstrap 1 ] ──► Tree 1 ──┐
          ──► [ Bootstrap 2 ] ──► Tree 2 ──┼──► [ Majority Vote / Average ] ──► Output
          ──► [ Bootstrap K ] ──► Tree K ──┘

Boosting (XGBoost) Sequential Architecture:
  Dataset ──► Tree 1 ──► Residual 1 ──► Tree 2 ──► Residual 2 ──► Tree K ──► Final Sum
```

---

## 4. Mathematics

### 1. Decision Tree Split Criterion (Gini Impurity & Information Gain)
For a node $m$ with class proportions $p_{mk}$:

$$H(Q_m) = \text{Gini}(m) = 1 - \sum_{k=1}^K p_{mk}^2$$

Splitting parent node $m$ into left child $L$ and right child $R$ maximizes Impurity Reduction:

$$\Delta H = H(Q_m) - \left( \frac{N_L}{N_m} H(Q_L) + \frac{N_R}{N_m} H(Q_R) \right)$$

### 2. XGBoost Second-Order Taylor Objective Function
At iteration $t$, XGBoost minimizes second-order approximation of loss:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

Where 1st derivative $g_i = \frac{\partial \mathcal{L}(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$ and 2nd derivative $h_i = \frac{\partial^2 \mathcal{L}(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$.

Optimal leaf weight $w_j^*$ for leaf $j$:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

---

## 5. Python (From Scratch Decision Tree Split Finder)

Finding optimal split thresholds in pure Python & NumPy:

```python
import numpy as np

def find_best_gini_split(X, y):
    """Calculates best split feature index and threshold."""
    N, d = X.shape
    best_gini = 1.0
    best_feature = None
    best_threshold = None
    
    for feat_idx in range(d):
        thresholds = np.unique(X[:, feat_idx])
        for thresh in thresholds:
            left_mask = X[:, feat_idx] <= thresh
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
                
            # Compute Gini Impurity for left and right
            p_left = np.mean(y[left_mask])
            gini_left = 1.0 - (p_left**2 + (1-p_left)**2)
            
            p_right = np.mean(y[right_mask])
            gini_right = 1.0 - (p_right**2 + (1-p_right)**2)
            
            weighted_gini = (np.sum(left_mask)/N) * gini_left + (np.sum(right_mask)/N) * gini_right
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feat_idx
                best_threshold = thresh
                
    return best_feature, best_threshold, best_gini

# Test Split Finder
X_test = np.array([[10, 100], [20, 200], [30, 100], [40, 300]])
y_test = np.array([0, 0, 1, 1])

feat, thresh, gini = find_best_gini_split(X_test, y_test)
print(f"Optimal Split: Feature Index {feat} at Threshold {thresh} (Gini: {gini:.4f})")
```

---

## 6. Production Library Benchmark (XGBoost, LightGBM, CatBoost)

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# 1. XGBoost Production Classifier
model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist')

# 2. LightGBM (GOSS - Gradient-based One-Side Sampling for ultra speed)
model_lgb = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1)

# 3. CatBoost (Target Encoding for Categoricals without Leakage)
model_cat = CatBoostClassifier(iterations=100, depth=6, verbose=0)
```

---

## 7. Under the Hood

- **XGBoost**: Uses Histogram-based binning (`tree_method='hist'`), grouping continuous features into 256 discrete bins to accelerate split finding by 10x-50x.
- **LightGBM**: Uses Leaf-Wise tree growth (best-first) instead of Depth-Wise, achieving lower loss at equal node counts.
- **CatBoost**: Uses Symmetric (Oblivious) trees where the same split criterion is evaluated across an entire tree level, preventing target leakage via Ordered Target Statistics.

---

## 8. Engineering Perspective

- **Hyper-Parameter Tuning Strategy**:
  - `n_estimators`: Set high (e.g. 1000) and use Early Stopping (`early_stopping_rounds=50`).
  - `learning_rate` ($\eta$): Set small (0.01 – 0.05).
  - `max_depth` / `num_leaves`: Control model complexity (depth 3–8 for tabular data).

---

## 9. Common Mistakes

1. **Setting Max Depth Too High in XGBoost**: Setting `max_depth=15` in XGBoost causes extreme overfitting and memory explosion. Keep XGBoost trees shallow (depth 3–8).
2. **Ignoring Imbalanced Class Weights**: Fitting tree ensembles on 99:1 imbalanced datasets without setting `scale_pos_weight = count(negative) / count(positive)`.

---

## 10. Interview Questions

### Q1: Compare XGBoost vs LightGBM vs CatBoost.
**Answer**: XGBoost uses depth-wise growth with second-order Taylor exact/hist splits. LightGBM uses leaf-wise growth with GOSS and EFB for maximum throughput on large datasets. CatBoost uses oblivious trees and ordered target statistics, outperforming on datasets with heavy categorical features.

---

## 11. Exercises

1. **Math**: Derive the optimal leaf weight formula $w_j^*$ for XGBoost from its second-order Taylor objective.
2. **Coding**: Implement a mini Random Forest Classifier in Python combining 10 Decision Trees with Bootstrap sampling.

---

## 12. Mini Project: Ensembling Benchmark Competition

Write a script `benchmark_trees.py` comparing Accuracy, ROC-AUC, Training Time, and Latency across Random Forest, XGBoost, LightGBM, and CatBoost on the Fraud Detection dataset.

---

## 13. Capstone Integration

Serves as the primary production engine in `src/house_prices/pipeline.py`, `src/fraud_detection/pipeline.py`, and `src/demand_forecasting/pipeline.py`.
