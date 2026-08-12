# 📦 Phase 04 — Modern Object Detection Systems

## 🎯 Phase Overview & Goal
- **Duration**: ~4 weeks
- **Objective**: Move beyond whole-image classification to spatial localization and multi-object recognition. Understand the historical transition from **Sliding Window Search** to **Two-Stage Detectors (R-CNN $\to$ Faster R-CNN)** and **One-Stage Real-Time Detectors (YOLO $\to$ RetinaNet $\to$ DETR)**.

---

## 1. Problem Formulation: Classification vs. Localization

Object Detection requires answering two questions simultaneously for every object in an image:
1. **What is it?** (Classification label $c \in \{1, \dots, C\}$).
2. **Where is it?** (Spatial bounding box $\mathbf{b} = [x_{min}, y_{min}, x_{max}, y_{max}]$ or $[x_c, y_c, w, h]$).

```text
 Image Classification             Object Localization               Object Detection
 ┌───────────────────┐            ┌───────────────────┐            ┌───────────────────┐
 │                   │            │      ┌─────┐      │            │ ┌──┐              │
 │                   │            │      │ Dog │      │            │ │Dog ┌─────┐      │
 │       "Dog"       │            │      └─────┘      │            │ └──┘ │ Cat │      │
 │                   │            │  [x, y, w, h]     │            │      └─────┘      │
 └───────────────────┘            └───────────────────┘            └───────────────────┘
```

---

## 2. Classical Object Detection: Sliding Window & HOG

Before 2014, object detection evaluated hundreds of thousands of candidate windows:

```text
Image ──► Multi-scale Pyramid ──► Sliding Window (Step=8px) ──► Extract HOG ──► Linear SVM ──► NMS
```

### Why Sliding Window Failed to Scale:
- For a $1000 \times 1000$ image evaluated across 5 scales and 3 aspect ratios:
  $$\text{Total Evaluated Windows} \approx \mathbf{200,000 \text{ window evaluations per image!}}$$
- Evaluating deep neural networks 200,000 times per frame was computationally impossible.

---

## 3. The Two-Stage Detector Evolution

```text
 R-CNN (2014) ────────► Fast R-CNN (2015) ────────► Faster R-CNN (2015)
 (Selective Search)      (RoI Pooling Layer)       (Region Proposal Network - RPN)
 ~47s / image            ~2.3s / image             ~0.2s / image (Real-time)
```

| Architecture | Proposal Generation | Feature Computation | Inference Speed | Key Innovation |
|---|---|---|---|---|
| **R-CNN** | Selective Search (2000 CPU proposals) | Forward passes CNN 2000 times per image | ~47 seconds | Proved CNN transfer learning works for object detection. |
| **Fast R-CNN** | Selective Search (CPU) | Convolves full image **once**; uses **RoI Pooling** on feature map | ~2.3 seconds | Shared convolutional features + multi-task loss (classification + bbox regression). |
| **Faster R-CNN** | **Region Proposal Network (RPN)** on GPU | Fully convolutional end-to-end network sharing backbone features | ~0.2 seconds (5-10 FPS) | Eliminated external CPU proposal bottlenecks via Anchor Boxes + RPN. |

### The Region Proposal Network (RPN) & Anchor Boxes:
Faster R-CNN places a set of $k$ predefined **Anchor Boxes** (e.g. 3 scales $\times$ 3 aspect ratios $= 9$ anchors) at every spatial location of the convolutional feature map.

```text
 Feature Map Pixel
        ● ──────┬──────► 1x1 Conv ──► 2k Objectness Scores (Foreground vs Background)
        │       └──────► 1x1 Conv ──► 4k Bounding Box Offsets (dx, dy, dw, dh)
   ┌────┴────┐
   │ Anchor  │
   │  Boxes  │
   └─────────┘
```

---

## 4. One-Stage Detectors: YOLO (You Only Look Once)

Joseph Redmon (2016) reformulated object detection as a **single regression problem**:

```text
Input Image ────────────────► Unified CNN ────────────────► S x S x (B * 5 + C) Tensor
(448 x 448 x 3)                                              (7 x 7 x 30 Tensor)
```

### The YOLO Grid Mechanism:
1. Divide input into an $S \times S$ grid (e.g., $7 \times 7$).
2. If the center of an object falls into a grid cell, that cell is responsible for predicting it.
3. Each grid cell predicts $B$ bounding boxes ($x, y, w, h, \text{confidence}$) and $C$ conditional class probabilities.
4. **Single forward pass**: Achieves **45 to 155 FPS** on GPU!

---

## 5. Mathematical Evaluation Metrics

### 5.1 Intersection over Union (IoU)
$$\text{IoU}(\mathbf{B}_1, \mathbf{B}_2) = \frac{\text{Area}(\mathbf{B}_1 \cap \mathbf{B}_2)}{\text{Area}(\mathbf{B}_1 \cup \mathbf{B}_2)} = \frac{\text{Intersection Area}}{\text{Area}(\mathbf{B}_1) + \text{Area}(\mathbf{B}_2) - \text{Intersection Area}}$$

```text
       ┌──────────────┐
       │ Box A        │
       │     ┌────────┼─────┐
       │     │Overlap │     │   IoU = Overlap Area / (Area A + Area B - Overlap)
       └─────┼────────┘     │
             │        Box B │
             └──────────────┘
```

### 5.2 Non-Maximum Suppression (NMS) Algorithm
To eliminate multiple overlapping detections for the same physical object:
1. Sort all predicted boxes by their confidence scores descending: $\mathcal{B} = [b_1, b_2, \dots, b_n]$.
2. Select box $M$ with highest confidence score, append to final output list $\mathcal{D}$, and remove from $\mathcal{B}$.
3. Discard any remaining box $b_i \in \mathcal{B}$ if $\text{IoU}(M, b_i) > \text{threshold}$ (typically $0.45 - 0.5$).
4. Repeat until $\mathcal{B}$ is empty.

### 5.3 Mean Average Precision (mAP)
For each object class, compute Precision and Recall across decreasing confidence thresholds:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$\text{AP} = \int_0^1 p(r) \, dr \approx \sum_{k=1}^N (r_k - r_{k-1}) \cdot p_{interp}(r_k)$$
$$\text{mAP} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c$$
- **COCO Metric**: $\text{mAP}@[.50:.05:.95]$ averages mAP across IoU thresholds from $0.50$ to $0.95$ in steps of $0.05$.
