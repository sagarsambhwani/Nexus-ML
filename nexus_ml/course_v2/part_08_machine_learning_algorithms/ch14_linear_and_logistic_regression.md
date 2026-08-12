# Chapter 14: Linear and Logistic Regression from Scratch

---

## 1. Big Picture

Linear and Logistic Regression are the bedrock of supervised machine learning. Before deploying complex deep neural networks or ensemble gradient boosted trees, an ML engineer establishes baseline benchmarks using linear models.

This chapter details the math, derivations, loss functions, Newton-Raphson optimization, regularizations ($L_1$ Lasso, $L_2$ Ridge, ElasticNet), failure cases, and interview questions for Linear and Logistic Regression.

---

## 2. Intuition

- **Linear Regression**: Fitting a hyper-plane $y = \mathbf{w}^T \mathbf{x} + b$ through continuous data points to minimize squared vertical distances (residuals).
- **Logistic Regression**: Passing the linear hyper-plane through a non-linear Sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$, squeezing outputs into probability range $[0, 1]$ to form a linear decision boundary.

---

## 3. Visualization

```text
Logistic Regression Sigmoid Activation & Decision Boundary:

           Probability P(y=1) ▲
                          1.0 ┼               * * * * *
                              │            *
                          0.5 ┼─────────*── Decision Boundary (z = 0)
                              │       *
                          0.0 ┼ * * *
                              └───────────────────────────► z = w^T x + b
                                 Class 0       Class 1
```

---

## 4. Mathematics

### 1. Ordinary Least Squares (OLS) & Normal Equation
$$\mathcal{L}_{\text{OLS}}(\mathbf{w}) = \frac{1}{2N} \|X \mathbf{w} - \mathbf{y}\|_2^2$$

Setting gradient $\nabla_\mathbf{w} \mathcal{L} = \mathbf{0}$ yields the analytical **Normal Equation**:
$$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

### 2. Binary Cross-Entropy Loss (Log Loss)
$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}$$

$$\mathcal{L}_{\text{BCE}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda_1}{\|w\|_1} + \frac{\lambda_2}{2} \|w\|_2^2$$

### 3. Newton-Raphson Optimization (Iteratively Reweighted Least Squares - IRLS)
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - H^{-1} \nabla \mathcal{L}$$

Where Hessian $H = X^T W X$ and $W = \text{diag}(\hat{y}_i (1 - \hat{y}_i))$.

---

## 5. Python (From Scratch Logistic Regression)

```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=500, l2_penalty=0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2_penalty
        self.w = None
        self.b = None
        
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        
    def fit(self, X, y):
        N, d = X.shape
        self.w = np.zeros((d, 1))
        self.b = 0.0
        y = y.reshape(-1, 1)
        
        for epoch in range(self.epochs):
            # Forward Pass
            z = X @ self.w + self.b
            y_hat = self._sigmoid(z)
            
            # Loss Calculation with L2 Regularization
            loss = -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15)) + (self.l2 / (2*N)) * np.sum(self.w**2)
            
            # Analytical Gradients
            dw = (1 / N) * X.T @ (y_hat - y) + (self.l2 / N) * self.w
            db = (1 / N) * np.sum(y_hat - y)
            
            # Parameter Updates
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

# Test Implementation
np.random.seed(42)
X_sim = np.random.randn(200, 2)
y_sim = (X_sim[:, 0] + X_sim[:, 1] > 0).astype(int)

model = LogisticRegressionScratch(lr=0.5, epochs=300)
model.fit(X_sim, y_sim)
probs = model.predict_proba(X_sim[:5])
print("Predicted Class Probabilities (First 5):")
print(probs.ravel())
```

---

## 6. Production Library (Scikit-Learn)

```python
from sklearn.linear_model import LogisticRegression

# Production Logistic Regression with L1/L2 ElasticNet penalty
clf = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', max_iter=1000)
clf.fit(X_sim, y_sim)
print("Scikit-Learn Accuracy:", clf.score(X_sim, y_sim))
```

---

## 7. Under the Hood

- Scikit-Learn `LogisticRegression` uses C++ backends **LIBLINEAR** (for coordinate descent) and **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno quasi-Newton method) for fast convex optimization.

---

## 8. Engineering Perspective

- **Failure Case: Perfect Linear Separability**: When classes are completely linearly separable, raw un-regularized Logistic Regression weights $\mathbf{w} \to \infty$ attempting to make Sigmoid probabilities exactly 0 and 1. Always include $L_2$ regularization ($\lambda > 0$) to prevent weight explosion!

---

## 9. Common Mistakes

1. **Not Scaling Features**: Running regularized logistic regression without standardization ($Z = \frac{X - \mu}{\sigma}$) penalizes large-magnitude unscaled features unfairly.
2. **Interpreting Raw Weights as Feature Importance without Scaling**: Unscaled regression weights reflect feature measurement units, not feature importance.

---

## 10. Interview Questions

### Q1: What is the main difference between L1 (Lasso) and L2 (Ridge) regularization?
**Answer**: $L_1$ regularization ($\lambda \|w\|_1$) adds absolute weight values to the loss function, producing sparse models by driving irrelevant feature weights strictly to zero (performing feature selection). $L_2$ regularization ($\lambda \|w\|_2^2$) shrinks weights continuously toward zero without forcing them exactly to zero, handling multicollinearity effectively.

---

## 11. Exercises

1. **Math**: Prove that the loss function of Logistic Regression (Binary Cross-Entropy) is strictly convex.
2. **Coding**: Implement Newton-Raphson IRLS algorithm for Logistic Regression in NumPy and compare convergence steps vs Gradient Descent.

---

## 12. Mini Project: Logistic Regression Credit Risk Classifier

Write a script `credit_classifier.py` that fits an ElasticNet Logistic Regression model on loan applicant data, plots ROC curves, and extracts feature odds-ratios.

---

## 13. Capstone Integration

Serves as baseline classifier in `src/customer_churn/pipeline.py` and `src/fraud_detection/pipeline.py`.
