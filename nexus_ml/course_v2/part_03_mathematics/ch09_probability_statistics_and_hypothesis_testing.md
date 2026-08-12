# Chapter 09: Probability, Statistics, and Hypothesis Testing in Code

---

## 1. Big Picture

Machine Learning is applied probability under uncertainty. Models do not predict deterministic truth; they output conditional probability distributions $P(Y \mid X)$.

Understanding probability density functions (PDFs), joint/marginal distributions, Bayes' Theorem, Maximum Likelihood Estimation (MLE), Maximum A Posteriori (MAP), Central Limit Theorem (CLT), confidence intervals, and p-value hypothesis testing (A/B testing) is required to evaluate models, establish statistical significance, and detect data drift.

---

## 2. Intuition

- **Probability**: Given a known coin with $P(\text{Heads}) = 0.7$, what is the chance of getting 8 heads in 10 flips? (Deductive reasoning).
- **Statistics**: Given 8 heads in 10 flips from an unknown coin, what is our best estimate of $P(\text{Heads})$ and how confident are we that it's not a fair 0.5 coin? (Inductive reasoning / MLE).
- **A/B Testing**: Proving that an increase in model click-through-rate (CTR) from 4.2% to 4.8% is a genuine statistical improvement rather than random noise.

---

## 3. Visualization

```text
Normal Distribution PDF & Null Hypothesis Testing:

                   Null Hypothesis Distribution H0
                                ▲
                               │       Reject H0 Region (α = 0.05)
                               │              │
                           * * │ * *          │
                       *       │       *      │
                     *         │         *    │
                   *           │           *  ▼
            ──────*────────────┼────────────*───► Z-score
                             μ=0           Z_crit = 1.96
                                            │
                                            ▼ Observed Z_stat = 2.45 (Statistically Significant!)
```

---

## 4. Mathematics

### 1. Bayes' Theorem & MAP
$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{P(\mathcal{D})}$$

$$\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log P(x_i \mid \theta)$$

$$\theta_{\text{MAP}} = \arg\max_\theta \left[ \sum_{i=1}^N \log P(x_i \mid \theta) + \log P(\theta) \right]$$

*Note: Ridge Regression ($L_2$) is mathematically identical to MAP estimation assuming a Gaussian Prior $P(\theta) = \mathcal{N}(0, \sigma^2 I)$!*

### 2. Two-Sample Welch's t-Test Statistic
To test if treatment group mean $\bar{X}_1$ differs significantly from control group mean $\bar{X}_2$:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}$$

Degrees of freedom $\nu$ calculated via Welch–Satterthwaite equation:
$$\nu \approx \frac{\left( \frac{s_1^2}{N_1} + \frac{s_2^2}{N_2} \right)^2}{\frac{(s_1^2/N_1)^2}{N_1 - 1} + \frac{(s_2^2/N_2)^2}{N_2 - 1}}$$

---

## 5. Python (From Scratch A/B Test & Bootstrap Confidence Intervals)

Evaluating model variant performance via Bootstrap Sampling from scratch:

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Synthetic CTR Data: Model A (Control) vs Model B (Treatment)
n_control, n_treatment = 1000, 1000
ctr_a = np.random.binomial(1, 0.05, size=n_control)    # 5.0% true conversion
ctr_b = np.random.binomial(1, 0.068, size=n_treatment)  # 6.8% true conversion

# 1. Analytical Welch's t-Test
t_stat, p_val = stats.ttest_ind(ctr_b, ctr_a, equal_var=False)
print(f"Analytical Welch t-Test: t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")
print(f"Significant at α=0.05? {p_val < 0.05}")

# 2. Non-Parametric Bootstrap 95% Confidence Interval
n_bootstraps = 5000
diffs = []
for _ in range(n_bootstraps):
    sample_a = np.random.choice(ctr_a, size=n_control, replace=True)
    sample_b = np.random.choice(ctr_b, size=n_treatment, replace=True)
    diffs.append(np.mean(sample_b) - np.mean(sample_a))

ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
print(f"Bootstrap 95% CI of CTR Lift: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

---

## 6. Production Library (SciPy & Statsmodels)

```python
from statsmodels.stats.proportion import proportions_ztest

# Production Z-test for A/B testing conversion ratios
count = np.array([ctr_b.sum(), ctr_a.sum()])
nobs = np.array([n_treatment, n_control])

z_stat, p_val_z = proportions_ztest(count, nobs, alternative='two-sided')
print(f"Production Z-Test p-value: {p_val_z:.4f}")
```

---

## 7. Under the Hood

- Scipy `stats.ttest_ind` calculates studentized $t$-distributions by evaluating incomplete beta functions $I_x(a, b)$ via optimized C routines in Cephes library.
- Bootstrap sampling operates with replacement, creating empirical empirical risk estimates without assuming Gaussian normality.

---

## 8. Engineering Perspective

- **Minimum Detectable Effect (MDE)**: Calculate required sample size *before* launching A/B tests to prevent underpowered experiments:
$$N \approx \frac{2 (Z_{\alpha/2} + Z_\beta)^2 \sigma^2}{\delta^2}$$
- **Data Drift**: Use Kolmogorov-Smirnov (KS) test or Population Stability Index (PSI) to detect distribution shifts between serving data and training data.

---

## 9. Common Mistakes

1. **p-hacking & Early Stopping**: Peeking at p-values daily during an active A/B test and stopping as soon as $p < 0.05$ inflates False Positive rates from 5% to over 30%!
2. **Confusing Correlation with Causation**: Assuming feature correlation $\rho_{X,Y} > 0.8$ implies that altering $X$ will cause changes in $Y$.

---

## 10. Interview Questions

### Q1: Prove why Minimizing Mean Squared Error (MSE) is equivalent to Maximum Likelihood Estimation (MLE) under Gaussian Noise.
**Answer**: Let $y_i = f_\theta(x_i) + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. The log-likelihood is $\log P(\mathcal{D} \mid \theta) = -\frac{N}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f_\theta(x_i))^2$. Maximizing this log-likelihood with respect to $\theta$ is equivalent to minimizing $\sum_{i=1}^N (y_i - f_\theta(x_i))^2$, which is Mean Squared Error!

---

## 11. Exercises

1. **Math**: Prove the Central Limit Theorem via Moment Generating Functions (MGFs).
2. **Coding**: Write a Python script `drift_detector.py` that computes Population Stability Index (PSI) between two dataset distributions and flags drift if $\text{PSI} > 0.25$.

---

## 12. Mini Project: Automated A/B Testing Suite

Build a production script `ab_testing_suite.py` that ingests user event logs, runs power calculations, performs Welch's t-test and Bootstrap validation, and exports an HTML report with p-values and confidence intervals.

---

## 13. Capstone Integration

Powers model drift evaluation in `src/customer_churn/pipeline.py` and confidence interval scoring across `src/medical_diagnosis/pipeline.py`.
