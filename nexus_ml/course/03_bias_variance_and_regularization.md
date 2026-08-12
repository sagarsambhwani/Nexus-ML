# 📗 Chapter 3: Bias-Variance Decomposition & Regularization Theory

## 3.1 Overfitting vs. Underfitting
The primary objective of machine learning is **generalization**—achieving high predictive performance on unseen data generated from the same underlying distribution.

```
       High Bias (Underfitting)             Optimal Balance             High Variance (Overfitting)
       (Model too simple)                (Good Generalization)            (Model memorizes noise)
            ◯    ◯                            ◯    ◯                            ◯ ─── ◯
          ◯        ◯                        ◯  ╱     ◯                        ◯  ╲ ╱   ◯
        ◯   ──────   ◯                    ◯   ╱───────  ◯                   ◯   ──╳──   ◯
```

- **Underfitting (High Bias)**: The model is overly restrictive (e.g. fitting a straight line to quadratic data) and fails to capture underlying structure in both training and test sets.
- **Overfitting (High Variance)**: The model is overly complex (e.g. a 20th-degree polynomial) and fits training sample noise, exhibiting low training error but high test error.

---

## 3.2 Mathematical Bias-Variance Decomposition

Consider a true data-generating process $y = f(x) + \epsilon$, where $\epsilon \sim N(0, \sigma^2)$ is irreducible noise. Given a training set $D$, our model estimates $\hat{f}(x; D)$.

The expected mean squared error (MSE) at a test point $x$ decomposes into three fundamental components:

$$\mathbb{E}_D \left[ (y - \hat{f}(x))^2 \right] = \underbrace{\Big( f(x) - \mathbb{E}_D[\hat{f}(x)] \Big)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_D \left[ \Big( \hat{f}(x) - \mathbb{E}_D[\hat{f}(x)] \Big)^2 \right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$

### Mathematical Proof Breakdown:
1. **Bias**: Difference between the expected model prediction and the true function $f(x)$.
2. **Variance**: Sensitivity of the model's predictions to small variations in the training set $D$.
3. **Irreducible Noise ($\sigma^2$)**: Inherent variance in the data that no model can overcome.

---

## 3.3 Regularization Fundamentals

Regularization introduces an explicit penalty term $R(w)$ to the loss function to constrain model parameters $w$, penalizing model complexity and reducing variance.

$$\mathcal{L}_{\text{regularized}}(w) = \mathcal{L}_{\text{data}}(w) + \lambda R(w)$$

where $\lambda \ge 0$ is the regularization hyperparameter.

```
Loss Function = Empirical Loss + Regularization Penalty
                L(w)            + λ * R(w)
```

---

## 3.4 L1 Regularization (Lasso Regression)

Lasso (Least Absolute Shrinkage and Selection Operator) penalizes the sum of absolute parameter weights ($L_1$ norm):

$$\mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2N} \sum_{i=1}^N (y_i - w^T x_i)^2 + \lambda \sum_{j=1}^p |w_j| = \frac{1}{2N} \| y - Xw \|_2^2 + \lambda \|w\|_1$$

### Automatic Feature Selection & Sparsity:
Lasso drives non-essential feature weights **exactly to zero** ($w_j = 0$), performing automatic feature selection.

```
        Lasso Constraint (L1 Diamond)              Ridge Constraint (L2 Circle)
                 w2                                         w2
                 ▲                                          ▲
                 │                                          │
               / │ \                                      ╭─┼─╮
             /   │   \                                  │  │  │
            /    │    \                                 │  │  │
    ───────◆─────┼─────◆──────► w1             ─────────┼──┼──┼─────────► w1
            \    │    /                                 │  │  │
             \   │   /                                  ╰─┼─╯
               \ │ /                                        │
                 ▼                                          ▼
   Corners intersect axes at w_j = 0            Smooth boundary retains non-zero weights
```

---

## 3.5 L2 Regularization (Ridge Regression)

Ridge regression penalizes the sum of squared parameter weights ($L_2$ norm):

$$\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2N} \sum_{i=1}^N (y_i - w^T x_i)^2 + \frac{\lambda}{2} \sum_{j=1}^p w_j^2 = \frac{1}{2N} \| y - Xw \|_2^2 + \frac{\lambda}{2} \|w\|_2^2$$

### Closed-Form Solution:
Solving $\nabla_w \mathcal{L}_{\text{Ridge}} = 0$ yields:

$$w^* = (X^T X + \lambda I)^{-1} X^T y$$

Adding $\lambda I$ to $X^T X$ ensures the matrix is strictly invertible, solving the **multicollinearity problem** (when features are highly correlated and $X^T X$ is singular or ill-conditioned).

---

## 3.6 ElasticNet Regularization

ElasticNet combines both $L_1$ (Lasso) and $L_2$ (Ridge) penalties:

$$\mathcal{L}_{\text{ElasticNet}}(w) = \frac{1}{2N} \| y - Xw \|_2^2 + \lambda \left( \alpha \|w\|_1 + \frac{1 - \alpha}{2} \|w\|_2^2 \right)$$

- $\alpha = 1 \rightarrow$ Pure Lasso ($L_1$)
- $\alpha = 0 \rightarrow$ Pure Ridge ($L_2$)
- **When to use**: High-dimensional datasets where features are strongly correlated in groups (ElasticNet selects or drops correlated features together).

---

## ⚖️ Summary Comparison of Regularization Techniques

| Technique | Penalty Term $R(w)$ | Mathematical Effect | Primary Advantage |
|---|---|---|---|
| **Lasso ($L_1$)** | $\sum \|w_j\|$ | Drives weights to exact $0$ | Sparse feature selection |
| **Ridge ($L_2$)** | $\sum w_j^2$ | Shrinks weights smoothly toward $0$ | Resolves multicollinearity |
| **ElasticNet** | $\alpha \|w\|_1 + \frac{1-\alpha}{2} \|w\|_2^2$ | Hybrid shrinkage & selection | Optimal for correlated feature groups |

---

## ⚓ Repository Code Reference
- See [`src/credit_risk/pipeline.py`](file:///e:/Downloads/ML_only/src/credit_risk/pipeline.py) for $L_2$-regularized Logistic Regression scorecard calibration.
- See [`src/house_prices/pipeline.py`](file:///e:/Downloads/ML_only/src/house_prices/pipeline.py) for shrinkage regularization controlling tree leaf weights.
