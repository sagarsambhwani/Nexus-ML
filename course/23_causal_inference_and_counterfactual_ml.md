# 🔮 Chapter 23: Causal Inference & Counterfactual Machine Learning

## 23.1 Correlation vs. Causation
Standard machine learning models measure **statistical association** ($P(Y \mid X)$). However, correlation does not imply causation. 

```
                                  Confounder Bias
                               Ice Cream Sales (X)
                                    ▲     ▲
                                   ╱       ╲
                                  ╱         ╲
                      Confounder: Hot Temperature (Z)
                                  ╲         ╱
                                   ╲       ╱
                                    ▼     ▼
                             Drowning Incidents (Y)
```

Predictive models correlate Ice Cream Sales ($X$) with Drowning Incidents ($Y$). But banning ice cream will not reduce drownings because both variables are driven by an unobserved **confounder** ($Z = \text{Temperature}$).

---

## 23.2 Judea Pearl's Causal Hierarchy

Judea Pearl structured causal reasoning into three distinct levels:

```
                           Pearl's Causal Hierarchy
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
L1: Association                   L2: Intervention                 L3: Counterfactuals
P(Y | X)                          P(Y | do(X))                      P(Y_x | x', y')
"What does observing X           "What happens if we               "What would have happened
 tell us about Y?"               intentionally force X?"           had X been different?"
```

---

## 23.3 Structural Causal Models (SCMs) & Do-Calculus

A Structural Causal Model (SCM) represents causal relationships using a **Directed Acyclic Graph (DAG)**:

$$X = f_X(Z, U_X), \quad Y = f_Y(X, Z, U_Y)$$

### The `do`-Operator:
$P(Y \mid \text{do}(X = x))$ models an explicit intervention where variable $X$ is forcibly set to value $x$, severing all incoming edges to $X$ in the causal DAG.

```
       Observational DAG                       Interventional DAG do(X = x)
             Z                                            Z
           ╱   ╲                                           ╲
          ▼     ▼                                           ▼
          X ───► Y                                    (X = x) ───► Y
```

---

## 23.4 Double Machine Learning (DML) for Causal Estimation

Estimating the True Causal Treatment Effect ($\theta_0$) in observational data with high-dimensional confounders $W$:

$$Y = \theta_0 X + g_0(W) + U, \quad \mathbb{E}[U \mid X, W] = 0$$
$$X = m_0(W) + V, \quad \mathbb{E}[V \mid W] = 0$$

### Chernozhukov et al. (2018) DML Algorithm:
1. **Model Outcome Noise**: Train ML model 1 to predict $Y$ from $W$: $\hat{Y} = \hat{g}(W)$. Compute residual $\tilde{Y} = Y - \hat{Y}$.
2. **Model Treatment Noise**: Train ML model 2 to predict $X$ from $W$: $\hat{X} = \hat{m}(W)$. Compute residual $\tilde{X} = X - \hat{X}$.
3. **Isolate Unbiased Treatment Effect**: Perform linear regression of $\tilde{Y}$ on $\tilde{X}$ to obtain $\hat{\theta}_0$:
   $$\hat{\theta}_0 = \frac{\sum \tilde{X}_i \tilde{Y}_i}{\sum \tilde{X}_i^2}$$

---

## ⚓ Repository Code Reference
- See [`src/customer_churn/pipeline.py`](file:///e:/Downloads/ML_only/src/customer_churn/pipeline.py) for retention incentive treatment rules.
