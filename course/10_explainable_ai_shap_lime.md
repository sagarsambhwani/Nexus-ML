# 🔍 Chapter 10: Explainable AI (XAI) — SHAP & LIME

## 10.1 The Imperative for Explainability
As machine learning models automate critical real-world decisions (credit underwriting, medical risk scoring, fraud blocking, hiring screening), black-box AI outputs are no longer acceptable. Legal regulations (e.g. EU General Data Protection Regulation GDPR "Right to Explanation", US Fair Credit Reporting Act FCRA) require institutions to explain **why** an automated decision was rendered for a specific individual.

```
                            Explainability Taxonomy
                                       │
           ┌───────────────────────────┴───────────────────────────┐
           ▼                                                       ▼
  Global Explainability                                   Local Explainability
(How does model behave overall?)                     (Why did model output Y for sample X?)
• Global SHAP Summary Plot                           • SHAP Waterfall Plot
• Permutation Importance                             • LIME Local Surrogate Line
```

---

## 10.2 SHAP (SHapley Additive exPlanations)

SHAP is grounded in **cooperative game theory**. It treats features as players in a coalition, computing the fair marginal contribution of each feature to the final prediction outcome.

### Mathematical Formula for Shapley Values:
The Shapley value $\phi_i$ for feature $i$ is calculated across all feature subsets $S \subseteq F \setminus \{i\}$:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \Big( f_x(S \cup \{i\}) - f_x(S) \Big)$$

- $|F|$: Total number of features.
- $S$: Subset of features excluding feature $i$.
- $f_x(S)$: Expected model prediction conditioned on feature subset $S$.

### Axiomatic Guarantees of SHAP:
1. **Efficiency / Additivity**: The sum of Shapley values equals the difference between model output and expected baseline prediction:
   $$f(x) = \mathbb{E}[f(X)] + \sum_{i=1}^p \phi_i(x)$$
2. **Symmetry**: Identical features contributing equally receive equal Shapley values.
3. **Dummy / Null Player**: A feature that contributes zero incremental value receives $\phi_i = 0$.

---

## 10.3 SHAP Visualizations in Python

```python
import shap
import xgboost as xgb

# Train XGBoost model
model = xgb.XGBClassifier().fit(X_train, y_train)

# Compute TreeSHAP values (O(TL D^2) optimized for tree ensembles)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# 1. Global Summary Plot (Feature Importance + Value Effect Direction)
shap.summary_plot(shap_values, X_test)

# 2. Local Waterfall Plot (Single Applicant Credit Rejection Analysis)
shap.plots.waterfall(shap_values[0])
```

### Interpreting a SHAP Waterfall Plot for Credit Denials:

```
Base Value E[f(X)] = 0.15 (15% Default Rate)
   │
   ├── +0.32  (DTI Ratio = 55% ──► High debt ratio increases risk)
   ├── +0.24  (Delinquencies = 3 ──► Delinquency history increases risk)
   ├── -0.12  (Credit Score = 720 ──► Good score reduces risk)
   │
   ▼
Final Prediction f(x) = 0.59 (59% Default Probability ──► REJECTED)
```

---

## 10.4 LIME (Local Interpretable Model-Agnostic Explanations)

LIME explains black-box predictions by training a simple, interpretable **local surrogate linear model** around the immediate neighborhood of a specific test sample $x$.

```
               Global Complex Non-Linear Decision Boundary
                     \                   /
                      \   ●   ●   ●     /
                       \    ★ (Sample) /
                        \  ●   ●   ●  /
                         ─────────────
                     Local Linear Decision Boundary (LIME Fit)
```

### Mathematical Optimization Objective:
$$\text{Explanation}(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

- $f(x)$: Complex original black-box model.
- $g$: Interpretable surrogate linear model (e.g. Ridge Regression).
- $\pi_x(z)$: Exponential distance kernel measuring proximity between perturbed sample $z$ and target sample $x$:
  $$\pi_x(z) = \exp\left( -\frac{D(x, z)^2}{\sigma^2} \right)$$
- $\Omega(g)$: Complexity penalty constraining the number of active surrogate features.

---

## ⚖️ SHAP vs. LIME Comparison

| Property | SHAP (Shapley Additive Explanations) | LIME (Local Interpretable Surrogate) |
|---|---|---|
| **Theoretical Foundation** | Cooperative Game Theory (Solid math proofs) | Heuristic Local Sampling |
| **Consistency / Additivity** | **Guaranteed** | Not guaranteed (Sampling variance) |
| **Speed** | Slow for exact; Fast for trees (`TreeSHAP`) | Fast local perturbation sampling |
| **Global Explainability** | **Yes** (Aggregates local SHAP values) | No (Purely local explanations) |
| **Best Use Case** | Regulatory credit/medical auditability | Quick model sanity checks |

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for Gini impurity feature attribution extraction (`top_risk_factors`).
- See [`src/credit_risk/pipeline.py`](file:///e:/Downloads/ML_only/src/credit_risk/pipeline.py) for scorecard linear feature weight attributions.
