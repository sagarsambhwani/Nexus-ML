# 🎥 Phase 12 — Video Vision & Temporal Understanding

## 🎯 Phase Overview & Goal
- **Duration**: ~3 weeks
- **Objective**: Extend spatial computer vision into the **time dimension**. Master **Optical Flow**, **Multi-Object Tracking (SORT $\to$ DeepSORT $\to$ ByteTrack)**, **3D Convolutions (I3D)**, and **Divided Space-Time Video Transformers (TimeSformer)**.

---

## 1. The 4th Dimension of Computer Vision

A video is a 4D spatio-temporal tensor:
$$\mathbf{V} \in \mathbb{R}^{T \times C \times H \times W}$$
where $T$ is the number of sequential temporal video frames.

```text
 Frame t-1                       Frame t                         Frame t+1
 ┌──────────────────────┐        ┌──────────────────────┐        ┌──────────────────────┐
 │       ▲              │        │            ▲         │        │                 ▲    │
 │      / \ (Ball)      │ ─────► │           / \ (Ball) │ ─────► │                / \   │
 │     └───┘            │        │          └───┘       │        │               └───┘  │
 └──────────────────────┘        └──────────────────────┘        └──────────────────────┘
 ◄────────────────────────────── Temporal Motion Vector ────────────────────────────────►
```

---

## 2. Optical Flow Mathematics

Optical flow computes the 2D apparent motion field $\mathbf{u} = [u, v]^\top$ of pixels between consecutive video frames $I(t)$ and $I(t + \Delta t)$.

### The Brightness Constancy Assumption:
$$I(x + \Delta x, y + \Delta y, t + \Delta t) \approx I(x, y, t)$$

By 1st-order Taylor series expansion:
$$I(x + \Delta x, y + \Delta y, t + \Delta t) \approx I(x, y, t) + \frac{\partial I}{\partial x}\Delta x + \frac{\partial I}{\partial y}\Delta y + \frac{\partial I}{\partial t}\Delta t$$

Dividing by $\Delta t$ yields the **Optical Flow Constraint Equation**:
$$I_x u + I_y v + I_t = 0 \iff \begin{bmatrix} I_x & I_y \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -I_t$$

### Lucas-Kanade Algorithm (Sparse Flow):
Assuming a constant motion $[u, v]^\top$ within an $n \times n$ local window (e.g. $3 \times 3 = 9$ equations for 2 unknowns):
$$\begin{bmatrix} I_{x_1} & I_{y_1} \\ I_{x_2} & I_{y_2} \\ \vdots & \vdots \\ I_{x_n} & I_{y_n} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_{t_1} \\ I_{t_2} \\ \vdots \\ I_{t_n} \end{bmatrix} \implies \mathbf{A}\mathbf{u} = -\mathbf{b}$$

Solved via Least Squares using the pseudo-inverse:
$$\mathbf{u} = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top (-\mathbf{b})$$
where $\mathbf{A}^\top \mathbf{A} = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$ is identical to the **Harris Structure Tensor**!

---

## 3. Multi-Object Tracking (MOT) Paradigm

Modern video tracking predominantly uses the **Tracking-by-Detection** paradigm:

```text
 Video Frame t ──► Detector (YOLO) ──► Candidate Bounding Boxes ──┐
                                                                 ├──► Hungarian Data Association ──► Updated Tracks
 Active Tracks (t-1) ──► Kalman Filter ──► Predicted State ──────┘    (IoU + Cosine Embeddings)
```

| Tracker | Trajectory Estimation | Similarity Metric | Key Advantage |
|---|---|---|---|
| **SORT** (Bewley, 2016) | Kalman Filter (Constant Velocity) | Spatial Bounding Box IoU | Super fast (>250 FPS on CPU); lacks re-identification through occlusion. |
| **DeepSORT** (Wojke, 2017) | Kalman Filter | IoU + Deep CNN Re-ID Appearance Vectors | Preserves track IDs even after long-term occlusion. |
| **ByteTrack** (Zhang, 2022) | Kalman Filter | Two-stage association on **both high and low score detections** | SOTA on MOT17/20; avoids dropping partially occluded or blurred objects. |

---

## 4. Video Action Recognition Architectures

### 4.1 3D Convolutional Networks (I3D - Carreira & Zisserman, 2017)
Inflates standard 2D $K \times K$ convolutional kernels along the temporal dimension into $K \times K \times K$ 3D kernels.
- **Weight Inflation**: Copies pre-trained 2D ImageNet filter weights $T$ times and divides by $T$:
  $$W_{3D}(t, i, j) = \frac{1}{T} W_{2D}(i, j)$$

### 4.2 TimeSformer: Divided Space-Time Self-Attention (Bertasius et al., 2021)
Rather than computing full spatio-temporal attention ($O(T^2 H^2 W^2)$), TimeSformer applies **Divided Self-Attention**:
1. **Temporal Attention**: Compares patches at the same spatial coordinates across time steps $T$.
2. **Spatial Attention**: Compares patches within the same frame across spatial coordinates $H \times W$.

```text
 Patch Token (t, h, w) ──► [ Temporal Attention across T frames ] ──► [ Spatial Attention across HxW space ]
```
