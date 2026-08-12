# 🎛️ Chapter 11: Hyperparameter Optimization & Optuna

## 11.1 Parameters vs. Hyperparameters
- **Model Parameters**: Learned automatically during training by optimizing the objective function (e.g. linear weights $w_i$, decision tree split thresholds).
- **Hyperparameters**: Set prior to training to govern model structure, capacity, and learning dynamics (e.g. learning rate $\eta$, max depth $d$, regularization $\lambda$, tree count $M$).

---

## 11.2 Hyperparameter Search Space Strategies

```
                           Search Space Strategies
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
    Grid Search                 Random Search             Bayesian Optimization
 (Exhaustive Matrix)       (Random Sampling Matrix)       (Surrogate Sequential Search)
```

### 1. Grid Search (`GridSearchCV`)
- **Mechanism**: Evaluates all Cartesian product combinations of specified hyperparameter grids.
- **Drawback**: Exponential scaling $O(K^d)$ (Curse of Dimensionality). Wastes compute searching non-sensitive parameter dimensions.

### 2. Random Search (`RandomizedSearchCV`)
- **Mechanism**: Randomly samples hyperparameter combinations from probability distributions.
- **Advantage**: Bergstra & Bengio (2012) proved Random Search explores unique values across sensitive parameter dimensions significantly more efficiently than Grid Search.

### 3. Bayesian Optimization
- **Mechanism**: Models hyperparameter search history sequentially using a probabilistic surrogate model (e.g. Gaussian Processes or Tree-structured Parzen Estimators) to sample combinations most likely to improve validation metrics.

---

## 11.3 Mathematical Foundation of Bayesian Optimization

Bayesian Optimization aims to find the global optimum:

$$x^* = \arg\max_{x \in \mathcal{X}} f(x)$$

where evaluating objective function $f(x)$ requires an expensive full model training and validation cycle.

```
       Surrogate Gaussian Process Model        Acquisition Function (Expected Improvement)
             Objective f(x)                                 Acquisition Score
            ▲      ╭───────╮                               ▲            ╭───
            │    ╭─╯       ╰─╮                             │          ╭─╯
            │  ╭─╯  Uncertain  ╰─╮                         │        ╭─╯ (Sample Next)
            └──┴──────────────────┴───►                    └────────┴─────────────►
```

### Components of Bayesian Optimization:
1. **Surrogate Model (Gaussian Process / TPE)**: Fits a prior distribution $P(f \mid D_{1:t})$ over function evaluations up to step $t$.
2. **Acquisition Function (Expected Improvement - EI)**: Determines the next point $x_{t+1}$ to sample by balancing **exploration** (high uncertainty regions) and **exploitation** (high predicted reward regions):
   $$\text{EI}(x) = \mathbb{E} \left[ \max(0, f(x) - f(x^+)) \right]$$
   where $f(x^+)$ is the best validation score observed so far.

---

## 11.4 Optuna Framework & Tree-structured Parzen Estimators (TPE)

Optuna is an industrial-grade hyperparameter optimization framework utilizing **Tree-structured Parzen Estimators (TPE)** and automatic trial pruning.

### TPE Optimization Algorithm:
Instead of modeling $P(f \mid x)$, TPE models feature densities conditional on target performance threshold $y^*$:

$$P(x \mid y) = \begin{cases} l(x) & \text{if } y < y^* \quad (\text{Good Trials}) \\ g(x) & \text{if } y \ge y^* \quad (\text{Bad Trials}) \end{cases}$$

The acquisition function maximizes the likelihood ratio:

$$\arg\max_x \frac{l(x)}{g(x)}$$

---

## 11.5 Python Optuna Code Example with XGBoost

```python
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    # 1. Define Search Space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42
    }
    
    # 2. Evaluate CV Score
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    return scores.mean()

# 3. Create Study & Optimize via TPE
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, timeout=300)

print(f"Best Trial ROC-AUC: {study.best_value:.4f}")
print("Best Hyperparameters:", study.best_params)
```

---

## 11.6 Automated Trial Pruning (Median & Hyperband Pruners)

Optuna accelerates optimization by prematurely terminating unpromising trials (e.g. if validation loss at epoch 5 is significantly worse than median historical trials).

```python
# Enable Median Pruner for Early Stopping
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    direction="maximize"
)
```

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for tree depth hyperparameter definitions.
- See [`src/house_prices/pipeline.py`](file:///e:/Downloads/ML_only/src/house_prices/pipeline.py) for learning rate shrinkage parameters.
