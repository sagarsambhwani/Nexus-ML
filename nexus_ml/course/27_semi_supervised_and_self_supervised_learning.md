# 🔄 Chapter 27: Semi-Supervised & Self-Supervised Learning (SSL)

## 27.1 Unlabeled Data Exploitation
In modern enterprise applications, unlabeled data is abundant, whereas obtaining high-quality expert annotations is expensive. Semi-Supervised and Self-Supervised Learning leverage unannotated data to build powerful feature representations.

```
                           Unlabeled Data Paradigms
                                      │
         ┌────────────────────────────┴────────────────────────────┐
         ▼                                                         ▼
Semi-Supervised Learning                               Self-Supervised Learning (SSL)
(Small Labeled + Large Unlabeled Data)                 (Zero Labeled Data Pre-training)
• Pseudo-Labeling                                      • Contrastive (SimCLR)
• Consistency Regularization (FixMatch)                • Masked Autoencoders (MAE, BERT)
```

---

## 27.2 Semi-Supervised Learning: FixMatch

FixMatch (Sohn et al. 2020) combines **pseudo-labeling** and **consistency regularization**:

```
Unlabeled Sample x ──┬──► Weak Augmentation (Crop/Flip) ──► Model P(y|x_weak) ──► High Conf? (p > τ) ──► Pseudo-Label y_hat
                     │                                                                                      │
                     └──► Strong Augmentation (AutoAugment) ──► Model P(y|x_strong) ─────────────────────────┴──► Cross-Entropy Loss
```

### FixMatch Loss Objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \lambda_u \mathcal{L}_{\text{unsupervised}}$$

$$\mathcal{L}_{\text{unsupervised}} = \frac{1}{B_u} \sum_{i=1}^{B_u} \mathbb{I}\left( \max(q_i) \ge \tau \right) \mathcal{H}\left( \hat{q}_i, \; p_m(y \mid \text{StrongAug}(u_i)) \right)$$

where $q_i = p_m(y \mid \text{WeakAug}(u_i))$ is the model prediction on weakly augmented sample, and $\tau \approx 0.95$ is the confidence threshold.

---

## 27.3 Self-Supervised Vision Learning: SimCLR & MAE

### 1. SimCLR (Contrastive Learning)
Generates two augmented views $\tilde{x}_i, \tilde{x}_j$ of the same image $x$. Maximizes agreement between positive pairs using **NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss**:

$$\ell_{i, j} = -\log \frac{\exp\left( \text{sim}(z_i, z_j) / \tau \right)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp\left( \text{sim}(z_i, z_k) / \tau \right)}$$

### 2. Masked Autoencoders (MAE - He et al. 2021)
Randomly masks a large portion (e.g. 75%) of image patches. The ViT encoder processes *only unmasked patches*, while a lightweight decoder reconstructs missing pixel patches from latent representations.

```
Original Image ──► Mask 75% Patches ──► ViT Encoder (25% Patches) ──► Decoder ──► Reconstructed 100% Image
```

---

## ⚓ Repository Code Reference
- See [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py) for patch feature representation.
