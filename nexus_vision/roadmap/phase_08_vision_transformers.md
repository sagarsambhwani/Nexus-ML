# ⚡ Phase 08 — Vision Transformers (ViT) & Self-Attention Architectures

## 🎯 Phase Overview & Goal
- **Duration**: ~4 weeks
- **Objective**: Understand how the **Transformer** architecture (originally designed for NLP) revolutionized Computer Vision by completely replacing convolutions with **Multi-Head Self-Attention**. Master patch embeddings, the `[CLS]` token, positional encoding, Swin Transformers, and the fundamental trade-off between **CNN Inductive Biases vs. Transformer Generalization**.

---

## 1. Vision Transformer (ViT) Architecture (Dosovitskiy et al., 2020)

```text
 ┌──────────────────────┐
 │ Input Image (224x224)│
 └──────────┬───────────┘
            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Divide into N = 196 non-overlapping Patches (each 16x16x3)   │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Linear Patch Projection: Flatten (16*16*3=768) ──► Vector D  │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Prepend Learnable [CLS] Token & Add 1D Positional Embeddings │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │  L x Transformer Blocks      │
                 │  ┌────────────────────────┐  │
                 │  │ LayerNorm + MHSA + Skip│  │
                 │  ├────────────────────────┤  │
                 │  │ LayerNorm + MLP + Skip │  │
                 │  └────────────────────────┘  │
                 └──────────────┬───────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Extract Output [CLS] Token ──► MLP Head ──► Class Output     │
 └──────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Walkthrough of ViT Components

### 2.1 Patch Extraction & Flattening
Given an input image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ and patch resolution $P = 16$:
- Number of visual patches:
  $$N = \frac{H \cdot W}{P^2} = \frac{224 \cdot 224}{16^2} = \mathbf{196 \text{ tokens}}$$
- Each patch is flattened into a 1D vector $\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C} = \mathbb{R}^{768}$.

### 2.2 Linear Projection & Positional Embedding
$$\mathbf{z}_0 = \left[ \mathbf{x}_{\text{class}}; \, \mathbf{x}_p^1 \mathbf{E}; \, \mathbf{x}_p^2 \mathbf{E}; \, \dots; \, \mathbf{x}_p^N \mathbf{E} \right] + \mathbf{E}_{pos}$$
- $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$: Learnable linear projection matrix (mapping raw pixels to hidden dimension $D = 768$).
- $\mathbf{x}_{\text{class}} \in \mathbb{R}^{1 \times D}$: Learnable classification token.
- $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$: Learnable 1D spatial position embeddings (retains 2D coordinate order).

### 2.3 Multi-Head Scaled Dot-Product Self-Attention (MHSA)
For queries $\mathbf{Q}$, keys $\mathbf{K}$, and values $\mathbf{V}$ projected via learnable weights $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V}$$

- **Global Receptive Field in Layer 1**: Unlike a CNN where receptive field grows layer by layer, **every patch can attend to every other patch in the very first layer of a ViT**!

---

## 3. Swin Transformer: Hierarchical Shifted Windows (Liu et al., 2021)

### The Quadratic Bottleneck of Standard ViT:
Standard ViT computes global self-attention across all $N$ tokens with **$O(N^2)$ quadratic computational complexity**. For high-resolution images ($1024 \times 1024$), this is computationally prohibitive.

### The Swin Innovation:
1. Computes local self-attention within non-overlapping $M \times M$ windows ($O(N)$ linear complexity).
2. **Shifted Window Partitioning (SW-MSA)**: Shifts windows across consecutive layers to introduce cross-window connections without increasing computational overhead.
3. **Hierarchical Merging**: Merges $2 \times 2$ neighboring patch tokens down the network, creating feature pyramids ($4\times, 8\times, 16\times, 32\times$) ideal for dense downstream tasks like **Object Detection (Mask R-CNN) and Segmentation (UPerNet)**.

```text
 Layer l (Regular Window Partition)             Layer l+1 (Shifted Window Partition)
 ┌──────────┬──────────┐                        ┌───┬──────────┬───┐
 │ Window 1 │ Window 2 │                        │   │          │   │
 ├──────────┼──────────┤                        ├───┼──────────┼───┤
 │ Window 3 │ Window 4 │                        │   │  Shifted │   │
 └──────────┴──────────┘                        └───┴──────────┴───┘
```

---

## 4. CNN Inductive Bias vs. Transformer Flexibility

```text
 Performance ▲                                                  Vision Transformer (ViT)
             │                                                 / (Uncapped Asymptotic Capacity)
             │                                                /
             │                            CNN (ResNet)       /
             │                           /──────────────────/
             │                          /  (Plateaus early due to
             │                         /    fixed spatial inductive bias)
             │                        /
             └───────────────────────/──────────────────────► Dataset Size (Images)
                           Small Data             Huge Data (JFT-300M, LAION-5B)
```

| Dimension | Convolutional Neural Networks (CNNs) | Vision Transformers (ViTs) |
|---|---|---|
| **Inductive Bias** | Strong (Locality, Translation Equivariance) | Weak (Must learn spatial topology from scratch) |
| **Small Dataset Performance** | **High** (Performs well on small datasets like CIFAR-10) | Low (Overfits without massive pretraining/heavy augmentation) |
| **Large-Scale Data Capacity** | Plateaus on billion-scale datasets | **Exceeds CNNs**; exhibits power-law scaling |
| **Receptive Field** | Local in early layers, grows linearly with depth | **Global in layer 1** across all image patches |
| **Robustness to Occlusion & Corruption** | Moderate | **High** (Dynamic attention can reroute around missing patches) |
