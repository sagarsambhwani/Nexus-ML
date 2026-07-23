# ⚖️ Chapter 20: AI Ethics, Fairness & Safety Alignment

## 20.1 Algorithmic Bias & Societal Risk
Machine learning models trained on historical data risk inheriting, amplifying, and perpetuating historical human biases against protected demographic groups (race, gender, age, disability status).

---

## 20.2 Mathematical Metrics for Algorithmic Fairness

Let $A \in \{0, 1\}$ represent a protected attribute (e.g. $A=0$ unprivileged group, $A=1$ privileged group), $X$ be un-protected features, $Y \in \{0, 1\}$ be the true ground-truth target, and $\hat{Y} \in \{0, 1\}$ be the binary model prediction.

```
                           Fairness Definitions
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
 Demographic Parity           Equalized Odds              Equal Opportunity
(Equal Positive Rate)      (Equal TPR & FPR)           (Equal True Positive Rate)
```

### 1. Demographic Parity (Statistical Parity)
Requires that the positive outcome selection rate is equal across both demographic groups:

$$P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$$

- **Disparate Impact Ratio (80% Rule)**:
  $$\text{Disparate Impact} = \frac{P(\hat{Y} = 1 \mid A = 0)}{P(\hat{Y} = 1 \mid A = 1)}$$
  If $\text{Disparate Impact} < 0.80$, the model exhibits adverse legal discrimination under US employment law (EEOC guidelines).

### 2. Equalized Odds
Requires that predictions $\hat{Y}$ and protected attribute $A$ are conditionally independent given true target $Y$:

$$P(\hat{Y} = 1 \mid A = 0, Y = y) = P(\hat{Y} = 1 \mid A = 1, Y = y) \quad \text{for } y \in \{0, 1\}$$

- True Positive Rates ($\text{TPR}$) and False Positive Rates ($\text{FPR}$) must match across groups.

### 3. Equal Opportunity
Requires equal True Positive Rates ($\text{TPR}$) for qualified candidates ($Y=1$):

$$P(\hat{Y} = 1 \mid A = 0, Y = 1) = P(\hat{Y} = 1 \mid A = 1, Y = 1)$$

---

## 20.3 Bias Mitigation Strategies

```
                           Bias Mitigation Lifecycle
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
Pre-Processing                    In-Processing                     Post-Processing
(Modify Training Data)            (Modify Optimization Loss)        (Adjust Prediction Thresholds)
• Re-weighing sample weights      • Adversarial Debiasing           • Group-specific thresholding
• Synthetic oversampling          • Fairness loss penalty λ        • Equalizing TPR decision cutoffs
```

```python
# Pre-processing Bias Mitigation: Re-weighing Sample Weights
from sklearn.utils.class_weight import compute_sample_weight

# Calculate sample weights to balance demographic group outcomes
sample_weights = compute_sample_weight("balanced", y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

---

## 20.4 AI Safety Alignment: RLHF & Guardrails

As AI systems become more capable, aligning model behavior with human intent, safety guidelines, and truthfulness is essential to prevent harmful, toxic, or deceptive outputs.

```
                              LLM Alignment Pipeline
Pre-Trained Base Model ──► Supervised Fine-Tuning (SFT) ──► Preference Alignment (RLHF / DPO) ──► Guardrail Layer
```

### 1. Reinforcement Learning from Human Feedback (RLHF)
- **Step 1 (SFT)**: Fine-tune base LLM on high-quality demonstration datasets.
- **Step 2 (Reward Model)**: Train a Reward Model $R_\psi(x, y)$ on human pairwise preferences ($y_w \succ y_l$).
- **Step 3 (PPO Optimization)**: Optimize LLM policy $\pi_\phi$ using PPO to maximize reward scores while maintaining a KL-divergence penalty to prevent policy drift.

### 2. Guardrails (Llama Guard & NeMo Guardrails)
Integrates secondary safety classification models directly at input/output boundaries to filter:
- Malicious prompt injections & jailbreaks
- Hate speech & toxic outputs
- PII (Personally Identifiable Information) leaks
- Medical & legal unauthorized advice

---

## ⚓ Repository Code Reference
- See [`src/credit_risk/README.md`](file:///e:/Downloads/ML_only/src/credit_risk/README.md) for regulatory compliance and Fair Credit Reporting Act (FCRA) audit requirements.
