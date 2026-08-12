# 🧠 Phase 03 — Convolutional Neural Networks (CNNs) from First Principles

## 🎯 Phase Overview & Goal
- **Duration**: ~4 weeks
- **Objective**: Build deep visual understanding from first principles. Master receptive fields, weight sharing, spatial downsampling, batch normalization, and trace the foundational evolution of architectures: **LeNet $\to$ AlexNet $\to$ VGG $\to$ Inception $\to$ ResNet**. For every architecture, answer: *"What fundamental bottleneck did the previous network have, and what exact mechanism solved it?"*

---

## 1. The Two Inductive Biases of CNNs

Why do CNNs dominate over standard Multi-Layer Perceptrons (MLPs) for vision?

1. **Spatial Locality**: Nearby pixels are strongly correlated; pixels on opposite corners are not. Standard MLPs connect all pixels to all hidden units, destroying 2D spatial structure.
2. **Translation Equivariance (Weight Sharing)**: A cat in the top-left corner has the same visual features (whiskers, ears) as a cat in the bottom-right corner. Convolving the same filter across the entire image guarantees that if the input translates, the feature map translates equivalently:
   $$f(\text{Shift}(X)) = \text{Shift}(f(X))$$

```text
  Fully-Connected (MLP)                         Convolutional Layer (CNN)
 ┌───┐         ┌───┐                            ┌───┬───┬───┐
 │ x1├────────►│ h1│                            │ x1│ x2│ x3│   <-- Local Receptive Field
 ├───┤  \   /  ├───┤                            ├───┼───┼───┤
 │ x2├────X───►│ h2│   (N*M Weights)            │ x4│ x5│ x6│ * [W1 W2] (Shared Kernel)
 ├───┤  /   \  ├───┤                            └───┴───┴───┘   [W3 W4]
 │ x3├────────►│ h3│
 └───┴─────────┴───┘
```

---

## 2. Receptive Field Mathematics

The **Effective Receptive Field (ERF)** is the area in the original input image that influences a particular unit in a deep feature map.

### Formula for Layer $l$:
$$RF_l = RF_{l-1} + (K_l - 1) \cdot J_{l-1}$$
where $J_{l-1} = \prod_{i=1}^{l-1} S_i$ is the cumulative stride up to layer $l-1$, and $K_l$ is the kernel size.

### The Stacked $3\times3$ Principle (VGG Insight):
Two cascaded $3 \times 3$ convolutional layers have an effective receptive field of:
$$RF = 3 + (3 - 1) \cdot 1 = \mathbf{5 \times 5}$$

Three cascaded $3 \times 3$ layers have an effective receptive field of $\mathbf{7 \times 7}$.

**Why use stacks of $3\times3$ instead of one $7\times7$?**
- **Parameter Efficiency**:
  - One $7 \times 7$ conv with $C$ channels: $7 \times 7 \times C^2 = 49 C^2$ weights.
  - Three $3 \times 3$ convs with $C$ channels: $3 \times (3 \times 3 \times C^2) = 27 C^2$ weights (**45% parameter reduction!**).
- **Non-Linearity**: Three non-linear activation functions (ReLUs) instead of one, creating more expressive feature representations.

---

## 3. Core CNN Building Blocks

### 3.1 Batch Normalization (Ioffe & Szegedy, 2015)
Standardizes activations across mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ during training:
$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta \quad (\text{Learnable scale } \gamma \text{ and shift } \beta)$$

- **Why it matters**: Enables training with higher learning rates, acts as a regularizer, and prevents vanishing/exploding gradients in deep networks.

### 3.2 Global Average Pooling (GAP)
Instead of flattening high-dimensional feature maps into massive fully-connected layers (which contain 80-90% of model parameters and overfit), GAP computes the spatial average of each channel:
$$GAP(F_c) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W F_c(i, j)$$

---

## 4. The Architectural Evolution Matrix

```text
 LeNet-5 (1998) ──► AlexNet (2012) ──► VGG-16 (2014) ──► GoogLeNet (2014) ──► ResNet (2015)
```

| Architecture | Year | Core Innovation | Problem it Solved in Previous Network |
|---|---|---|---|
| **LeNet-5** | 1998 | Conv + Average Pool + Tanh on 32x32 digits | Handwritten digit OCR baseline. |
| **AlexNet** | 2012 | GPU parallelism, ReLU activation, Dropout, Data Augmentation | Overcame Vanishing Gradients and Overfitting on large-scale ImageNet (1.2M images). |
| **VGG-16/19** | 2014 | Homogeneous architecture using strictly $3\times3$ convolutions | Eliminated arbitrary kernel designs ($11\times11, 7\times7, 5\times5$); proved depth improves representations. |
| **GoogLeNet / Inception** | 2014 | Multi-scale Inception modules + $1\times1$ bottleneck convolutions | Processed features at multiple scales simultaneously while drastically cutting computational FLOPs. |
| **ResNet** | 2015 | Residual learning with Identity Skip Connections: $\mathcal{H}(x) = \mathcal{F}(x) + x$ | Solved the **Degradation Problem** (vanishing gradient / optimization barrier) allowing networks to scale to 152+ layers. |

---

## 5. ResNet Deep-Dive: Why Residual Connections Work

In a plain deep network, adding more layers causes the training error to *increase* (the degradation problem):

```text
 Plain Network:                          Residual Block (ResNet):
 ┌───────────┐                           ┌───────────┐
 │   Input x ├──┐                        │   Input x ├───────┐ Identity Shortcut
 └─────┬─────┘  │                        └─────┬─────┘       │ (x)
       ▼        │                              ▼             │
 ┌───────────┐  │                        ┌───────────┐       │
 │ Conv Layer│  │                        │ Conv Layer│       │
 └─────┬─────┘  │                        └─────┬─────┘       │
       ▼        │                              ▼             │
 ┌───────────┐  │                        ┌───────────┐       │
 │ Conv Layer│  │                        │ Conv Layer│       │
 └─────┬─────┘  │                        └─────┬─────┘       │
       ▼        ▼                              ▼             ▼
  Output H(x) = F(x)                      Output H(x) = F(x) + x
```

### The Gradient Propagation Advantage:
During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to input $x$ is:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \cdot \frac{\partial \mathcal{H}}{\partial x} = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \left( \frac{\partial \mathcal{F}(x)}{\partial x} + \mathbf{I} \right) = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \frac{\partial \mathcal{F}(x)}{\partial x} + \mathbf{\frac{\partial \mathcal{L}}{\partial \mathcal{H}}}$$

The crucial term is $\mathbf{+ \frac{\partial \mathcal{L}}{\partial \mathcal{H}}}$! Even if the learned weight gradient $\frac{\partial \mathcal{F}(x)}{\partial x}$ approaches zero, the gradient flows directly backwards unimpeded through the identity path $\mathbf{I}$.
