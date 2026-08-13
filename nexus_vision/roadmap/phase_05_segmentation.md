# 🎭 Phase 05 — Image Segmentation (Semantic, Instance & Panoptic)

## 🎯 Phase Overview & Goal
- **Duration**: ~3 weeks
- **Objective**: Extend visual recognition to the **pixel level**. Master the three segmentation paradigms (**Semantic vs. Instance vs. Panoptic**), classical topological algorithms (Watershed, Graph Cut), and deep encoder-decoder architectures (**FCN $\to$ U-Net $\to$ DeepLab $\to$ Mask R-CNN**).

---

## 1. The Three Segmentation Paradigms

```text
 Semantic Segmentation             Instance Segmentation             Panoptic Segmentation
 ┌──────────────────────┐          ┌──────────────────────┐          ┌──────────────────────┐
 │  Sky (Blue)          │          │                      │          │  Sky (Stuff: Blue)   │
 │  Road (Gray)         │          │  Car 1 (Red)         │          │  Road (Stuff: Gray)  │
 │  Cars (All Green)    │          │  Car 2 (Yellow)      │          │  Car 1 (Thing: Red)  │
 │                      │          │                      │          │  Car 2 (Thing: Yellow│
 └──────────────────────┘          └──────────────────────┘          └──────────────────────┘
 (Class per pixel)                 (Mask per object instance)        (Unified Stuff + Things)
```

1. **Semantic Segmentation**: Assigns a semantic class label $c \in \{1, \dots, C\}$ to every pixel $(x, y)$. Does not distinguish between individual objects of the same class (e.g., all pedestrians share the same label).
2. **Instance Segmentation**: Detects individual object instances ("things") and outputs a precise binary mask for each object. Ignores background ("stuff").
3. **Panoptic Segmentation**: Unifies Semantic and Instance segmentation into a single dense segmentation map where every pixel is assigned both a semantic category and an instance ID.

---

## 2. Classical Segmentation: Watershed & Graph Cut

### 2.1 The Watershed Transform
Treats the gradient magnitude image $|\nabla I|$ as a topological surface:
- High gradient values (edges) act as **mountain ridges / watershed lines**.
- Low gradient values act as **catchment basins**.
- **Immersion Simulation**: Water rises from local minima markers until basins meet at boundary lines.

### 2.2 GrabCut (Graph Cuts Energy Minimization)
Formulates foreground/background segmentation as a **Min-Cut / Max-Flow** graph optimization problem on a Markov Random Field (MRF):
$$E(\alpha, k, \theta, z) = U(\alpha, k, \theta, z) + V(\alpha, z)$$
where $U$ is the regional data term (Gaussian Mixture Models of foreground/background color) and $V$ is the smoothness boundary penalty term.

---

## 3. Deep Segmentation Architectures

```text
 Fully Convolutional Networks (FCN) ──► U-Net (2015) ──► DeepLabv3+ (2018) ──► Mask R-CNN (2017)
 (Replaced FC layers with 1x1 Conv)     (Skip Concatenations) (Atrous Convolutions + ASPP) (RoIAlign + FCN Branch)
```

---

## 4. U-Net Deep-Dive (Ronneberger et al., 2015)

Originally designed for biomedical microscopy segmentation, U-Net is the gold standard encoder-decoder architecture:

```text
 Contracting Encoder Path                                           Expansive Decoder Path
 ┌──────────────────────┐                                           ┌──────────────────────┐
 │  Input Image (572x572)                                           │  Output Mask (388x388│
 └──────────┬───────────┘                                           └──────────▲───────────┘
            ▼                                                                  │
      ┌───────────┐         High-Resolution Skip Connections             ┌───────────┐
      │ Conv 3x3  ├─────────────────────────────────────────────────────►│ Conv 3x3  │
      └─────┬─────┘                                                      └─────▲─────┘
            ▼ MaxPool 2x2                                                      │ UpConv 2x2
      ┌───────────┐                                                      ┌─────┴─────┐
      │ Conv 3x3  ├─────────────────────────────────────────────────────►│ Conv 3x3  │
      └─────┬─────┘                                                      └─────▲─────┘
            ▼ MaxPool 2x2                                                      │ UpConv 2x2
      ┌───────────┐                                                      ┌─────┴─────┐
      │ Conv 3x3  ├─────────────────────────────────────────────────────►│ Conv 3x3  │
      └─────┬─────┘                                                      └─────▲─────┘
            ▼ MaxPool 2x2                                                      │ UpConv 2x2
                  ┌───────────────────────────────────────────────┐            │
                  │                   Bottleneck                  ├────────────┘
                  └───────────────────────────────────────────────┘
```

### Why Skip Connections are Essential in U-Net:
1. **The Downsampling Dilemma**: Max-pooling operations in the encoder compress spatial dimensions, destroying high-frequency edge and boundary location details.
2. **The Skip Solution**: Skip connections copy high-resolution spatial feature maps directly from the encoder stages and **concatenate them channel-wise** to the corresponding decoder stages.
3. **Result**: The decoder receives both **deep semantic context** (from the bottleneck) and **precise boundary coordinates** (from the encoder skips).

---

## 5. DeepLab & Atrous (Dilated) Convolutions

DeepLab solves the resolution loss problem without pooling using **Atrous (Dilated) Convolutions**:

$$y[i] = \sum_k x[i + r \cdot k] \cdot w[k]$$
where $r$ is the **Dilation Rate**.

```text
  Standard Conv (Rate r=1)               Dilated Conv (Rate r=2)                Dilated Conv (Rate r=3)
       ┌───┬───┬───┐                          ┌───┬───┬───┬───┬───┐                  ┌───┬───┬───┬───┬───┬───┬───┐
       │ x │ x │ x │                          │ x │   │ x │   │ x │                  │ x │   │   │ x │   │   │ x │
       ├───┼───┼───┤                          ├───┼───┼───┼───┼───┤                  ├───┼───┼───┼───┼───┼───┼───┤
       │ x │ x │ x │                          │   │   │   │   │   │                  │   │   │   │   │   │   │   │
       ├───┼───┼───┤                          ├───┼───┼───┼───┼───┤                  ├───┼───┼───┼───┼───┼───┼───┤
       │ x │ x │ x │                          │ x │   │ x │   │ x │                  │ x │   │   │ x │   │   │ x │
       └───┴───┴───┘                          └───┴───┴───┴───┴───┘                  └───┴───┴───┴───┴───┴───┴───┘
       Receptive Field: 3x3                   Receptive Field: 5x5                   Receptive Field: 7x7
       (9 Parameters)                         (9 Parameters - Zero Extra FLOPs!)     (9 Parameters)
```

- **Atrous Spatial Pyramid Pooling (ASPP)**: Applies multiple parallel atrous convolutions with varying dilation rates (e.g. $r \in \{6, 12, 18, 24\}$) to capture multi-scale context without downsampling.

---

## 6. Loss Functions for Segmentation

### 6.1 Dice Loss (Sørensen–Dice Coefficient)
Critical for imbalanced datasets (e.g., small tumor or scratch defect vs. 99% background pixels):
$$\text{Dice} = \frac{2 |\mathbf{X} \cap \mathbf{Y}|}{|\mathbf{X}| + |\mathbf{Y}|} = \frac{2 \sum_{i=1}^N p_i g_i}{\sum_{i=1}^N p_i^2 + \sum_{i=1}^N g_i^2 + \epsilon}$$
$$\mathcal{L}_{\text{Dice}} = 1 - \text{Dice}$$
where $p_i \in [0, 1]$ is the predicted probability and $g_i \in \{0, 1\}$ is the ground truth.
