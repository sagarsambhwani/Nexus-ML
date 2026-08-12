# 🔮 Phase 07 — Representation Learning & Self-Supervised Vision

## 🎯 Phase Overview & Goal
- **Duration**: ~3 weeks
- **Objective**: Answer the fundamental question: *"How can a neural network learn universal, rich visual features from millions of uncurated images without a single human label?"* Master Autoencoders, Masked Autoencoding (MAE), and Contrastive Learning (**SimCLR $\to$ MoCo $\to$ BYOL $\to$ DINO**).

---

## 1. The Paradigm Shift: Beyond Supervised Pre-training

```text
 Supervised Learning (ImageNet)                    Self-Supervised Representation Learning
 ┌──────────────┐                                  ┌──────────────┐
 │  Image Data  ├──────┐                           │  Image Data  ├──────┐
 └──────────────┘      ▼                           └──────────────┘      ▼
 ┌──────────────┐  [Supervised Loss]                                 [Pretext Task Loss]
 │ Human Labels ├─────► (Cross-Entropy)                              (Contrastive / MAE)
 └──────────────┘      │                                                 │
                       ▼                                                 ▼
               Class-Specific Weights                           Universal Visual Encoder
               (Expensive, Biased)                              (Self-Supervised Features)
```

Human labeling is expensive, noisy, and limits representations to fixed discrete categories. Self-Supervised Learning (SSL) forces the network to discover the underlying physical and semantic structure of visual reality.

---

## 2. Autoencoders & Masked Autoencoders (MAE)

### 2.1 Classical Autoencoder
$$\mathbf{z} = f_\theta(\mathbf{x}), \quad \mathbf{\hat{x}} = g_\phi(\mathbf{z}), \quad \mathcal{L} = \|\mathbf{x} - \mathbf{\hat{x}}\|_2^2$$
- **Limitation**: The network can easily learn trivial identity mappings or high-frequency pixel artifacts rather than semantic object abstractions.

### 2.2 Masked Autoencoders (MAE - He et al., 2021)
1. Divide image into non-overlapping patches (e.g. $16 \times 16$).
2. **Mask a high proportion (75% to 80%)** of patches at random.
3. Feed only the remaining 25% unmasked patches into a Vision Transformer (ViT) encoder.
4. Insert learnable [MASK] tokens and run a lightweight decoder to reconstruct original pixel values in the masked regions.

```text
 Original Image (100%) ──► Random Masking (75%) ──► Unmasked Patches (25%) ──► ViT Encoder ──► [MASK] Tokens ──► Lightweight Decoder ──► Reconstructed Image
```

---

## 3. Contrastive Learning & SimCLR (Chen et al., 2020)

Contrastive learning operates on the principle of **Invariance under Data Augmentation**: two different augmented views of the same image should yield nearly identical vector embeddings in latent space.

```text
                                    Input Image x
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         Augmented View x_i                              Augmented View x_j  (Positive Pair)
                  │                                               │
                  ▼                                               ▼
          Base Encoder f(·)                               Base Encoder f(·)  (e.g., ResNet-50)
                  │                                               │
                  ▼ Representation h_i                            ▼ Representation h_j
          Projection Head g(·)                            Projection Head g(·) (MLP 2048->128)
                  │                                               │
                  ▼ Embedding z_i                                 ▼ Embedding z_j
                  └───────────────────────┬───────────────────────┘
                                          ▼
                            NT-Xent Contrastive Loss (Pull z_i and z_j together,
                                                      Push away negative samples in batch)
```

### The NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss:
For positive pair $(i, j)$ in a batch of $N$ images ($2N$ augmented views):
$$\mathcal{L}_{i, j} = -\log \frac{\exp\left(\frac{\text{sim}(\mathbf{z}_i, \mathbf{z}_j)}{\tau}\right)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp\left(\frac{\text{sim}(\mathbf{z}_i, \mathbf{z}_k)}{\tau}\right)}$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ is the Cosine Similarity, and $\tau$ is the Temperature parameter ($\tau \approx 0.07 - 0.1$).

> [!NOTE]
> **Why Discard the Projection Head $g(h)$ for Downstream Tasks?**
> The projection head $g(\cdot)$ discards specific transformation information (e.g., color distribution, orientation) to maximize invariance in $\mathbf{z}$. The intermediate feature vector $\mathbf{h}$ retains much richer linear information, making $\mathbf{h}$ significantly better for downstream classification.

---

## 4. Self-Supervised Evolutionary Milestones

| Framework | Authors | Negative Pairs Needed? | Core Mechanism |
|---|---|---|---|
| **SimCLR** | Chen et al. (Google, 2020) | Yes (Requires huge batch sizes: 4096) | Dual-view data augmentation + Projection head + NT-Xent loss. |
| **MoCo (v1/v2/v3)** | He et al. (Meta, 2020) | Yes (Decoupled from batch size) | Dynamic FIFO Memory Queue dictionary + Slow Momentum Target Encoder. |
| **BYOL** | Grill et al. (DeepMind, 2020) | **No** (Zero negative pairs) | Online network predicts Target network representation; asymmetry + EMA prevents collapse. |
| **DINO** | Caron et al. (Meta, 2021) | **No** (Self-distillation) | ViT Student-Teacher network with centering & sharpening; produces emergent semantic segmentation maps! |
