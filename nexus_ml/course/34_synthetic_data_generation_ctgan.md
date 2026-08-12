# 🧪 Chapter 34: Synthetic Data Generation & Tabular GANs (CTGAN)

## 34.1 The Synthetic Data Imperative
Enterprise data science teams face severe privacy compliance barriers (GDPR, HIPAA, PCI-DSS) and class imbalance problems (rare medical conditions, 0.1% transaction fraud). **Synthetic Data Generation** constructs privacy-preserving artificial datasets that replicate the mathematical distribution, correlations, and predictive utility of real production data without exposing sensitive PII.

---

## 34.2 CTGAN (Conditional Tabular GAN) Architecture

Introduced by Xu et al. (MIT, 2019), **CTGAN** solves the two main challenges of tabular data generation:
1. **Multimodal Non-Gaussian Continuous Features**: Continuous variables with complex skewed distributions.
2. **Severe Categorical Imbalance**: High-cardinality discrete features with rare category modes.

```
                      CTGAN Mode-Specific Normalization & Conditional Generator
Continuous Input x ──► Fit Gaussian Mixture Model (GMM) ──► Representation (Val v, One-Hot Mode u) ──┐
                                                                                                    ├──► Conditional Generator ──► Synthetic Table
Categorical Input c ──► Conditional Vector Representation ──────────────────────────────────────────┘
```

### 1. Mode-Specific Normalization (Continuous Columns)
CTGAN represents each continuous value $x_{i,j}$ using a Gaussian Mixture Model (GMM) with $K$ components:

$$x_{i,j} = \sum_{k=1}^K \mu_k \cdot u_{i,j,k} + \sigma_k \cdot v_{i,j,k}$$

- $u_{i,j}$: One-hot vector indicating which GMM Gaussian component mode $x_{i,j}$ belongs to.
- $v_{i,j}$: Normalized scalar value indicating how many standard deviations $x_{i,j}$ lies from mode mean $\mu_k$.

### 2. Training-by-Sampling & Conditional Vector
To prevent generator collapse on minority categories, CTGAN feeds a **Conditional Vector** $r$ specifying category constraints into both the Generator and Discriminator during training.

---

## 34.3 Utility & Privacy Evaluation Metrics

```
                            Synthetic Data Metrics
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
   Fidelity & Utility                                         Privacy Protection
• Kolmogorov-Smirnov (KS) Test                               • Distance to Closest Record (DCR)
• Machine Learning Efficacy (Train on Synth, Test on Real)   • Nearest Neighbor Distance Ratio (NNDR)
```

---

## ⚓ Repository Code Reference
- See [`src/medical_diagnosis/pipeline.py`](file:///e:/Downloads/ML_only/src/medical_diagnosis/pipeline.py) for medical feature distribution sampling.
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for imbalanced transaction sampling.
