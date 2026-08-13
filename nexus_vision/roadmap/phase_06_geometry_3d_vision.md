# 📐 Phase 06 — Geometry & 3D Computer Vision

## 🎯 Phase Overview & Goal
- **Duration**: ~4 weeks
- **Objective**: Bridge the gap between **2D digital pixel grids and the 3D physical world**. Master the pinhole camera model, intrinsic/extrinsic calibration, homography perspective transformations, epipolar geometry, stereo disparity depth estimation, and Structure from Motion (SfM).

---

## 1. The Pinhole Camera Model & Coordinate Transformations

A 3D physical point $\mathbf{P}_w = [X_w, Y_w, Z_w]^\top$ undergoes three successive coordinate transformations to become a 2D image pixel $\mathbf{p} = [u, v]^\top$:

```text
 ┌─────────────────────────┐          ┌─────────────────────────┐          ┌─────────────────────────┐
 │ World Coordinate Frame  │ ───────► │ Camera Coordinate Frame │ ───────► │ 2D Pixel Coordinate Grid│
 │       (Xw, Yw, Zw)      │ [R | T]  │       (Xc, Yc, Zc)      │    K     │          (u, v)         │
 └─────────────────────────┘          └─────────────────────────┘          └─────────────────────────┘
```

### The Full Perspective Projection Equation:
$$z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \cdot [\mathbf{R} \mid \mathbf{T}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

$$\mathbf{P} = \mathbf{K} [\mathbf{R} \mid \mathbf{T}] \in \mathbb{R}^{3 \times 4}$$

---

## 2. Camera Intrinsic Matrix ($\mathbf{K}$) vs. Extrinsic Matrix ($[\mathbf{R} \mid \mathbf{T}]$)

### 2.1 Intrinsic Parameters ($\mathbf{K} \in \mathbb{R}^{3 \times 3}$)
Encodes internal optical and sensor characteristics:
$$\mathbf{K} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
- $f_x, f_y$: Focal lengths expressed in pixel units ($f_x = \frac{f}{p_x}$, where $p_x$ is physical pixel width).
- $c_x, c_y$: **Principal Point** (intersection of optical axis with sensor, usually near image center $\approx \frac{W}{2}, \frac{H}{2}$).
- $s$: Axis skew factor (typically $0$ in modern CMOS/CCD sensors).

### 2.2 Extrinsic Parameters ($[\mathbf{R} \mid \mathbf{T}] \in \mathbb{R}^{3 \times 4}$)
Encodes the **Camera Pose** (position and orientation) in world space:
- $\mathbf{R} \in \text{SO}(3)$: $3 \times 3$ Rotation matrix (3 degrees of freedom: Roll, Pitch, Yaw).
- $\mathbf{T} \in \mathbb{R}^3$: $3 \times 1$ Translation vector (Camera position relative to world origin).

---

## 3. Planar Homography ($\mathbf{H}$)

A **Homography** is an invertible mapping between two planar surfaces in projective space $\mathbb{P}^2$:
$$\mathbf{\tilde{x}}_2 = \mathbf{H} \mathbf{\tilde{x}}_1 = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \\ 1 \end{bmatrix}$$

- **Degrees of Freedom**: $8$ (since scale is arbitrary: $\mathbf{H} \sim \lambda \mathbf{H}$).
- **Minimum Correspondences**: Requires at least **4 point pairs** to solve via Direct Linear Transform (DLT) with SVD.
- **Applications**: Document scanning perspective rectification, bird's-eye view camera calibration, panorama image stitching.

```text
 Distorted Trapezoid View                               Rectified Orthogonal Top-Down View
 ┌──────────────────────┐                               ┌──────────────────────┐
 │      ┌────────┐      │                               │ ┌──────────────────┐ │
 │     /          \     │   ───► Homography H ───►      │ │                  │ │
 │    /            \    │                               │ │                  │ │
 │   └──────────────┘   │                               │ └──────────────────┘ │
 └──────────────────────┘                               └──────────────────────┘
```

---

## 4. Epipolar Geometry & The Fundamental Matrix

When two cameras observe the same 3D point $\mathbf{X}$, their centers $\mathbf{C}_1, \mathbf{C}_2$ and $\mathbf{X}$ form the **Epipolar Plane**:

```text
                             3D Point X
                                 ●
                                / \
                               /   \
                              /     \
                             /       \
                     x1 ●   /         \   ● x2
                       /   /           \   \
                      /   /             \   \
                     /   /               \   \
                 e1 ●───/─────────────────\───● e2 (Epipoles)
            Camera C1                         Camera C2
```

### The Epipolar Constraint:
For corresponding points $\mathbf{x}_1$ in Image 1 and $\mathbf{x}_2$ in Image 2:
$$\mathbf{x}_2^\top \mathbf{F} \mathbf{x}_1 = 0$$

- $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ is the **Fundamental Matrix** (rank 2, 7 degrees of freedom).
- The product $\mathbf{l}_2 = \mathbf{F} \mathbf{x}_1$ represents the **Epipolar Line** in Image 2. Point $\mathbf{x}_2$ MUST lie on line $\mathbf{l}_2$!
- **Search Space Reduction**: We do not need to search the entire 2D image for matching keypoints; we only search along a **1D line**!

---

## 5. Stereo Vision & Depth Calculation

In a calibrated stereo camera pair with baseline distance $B$ and focal length $f$:

```text
 Camera Left (CL)                                    Camera Right (CR)
        ● ◄───────────────── Baseline B ──────────────────► ●
        │                                                   │
        │ \                                               / │
        │   \                                           /   │
        │     \                                       /     │
        │       \                 3D Point P        /       │
        │         \                   ●           /         │
        │           \               /   \       /           │
        └─────────────\───────────/───────\───/─────────────┘
                    x_L         /           x_R
                                ◄──── Z ────►
                                 (Depth)
```

### The Fundamental Triangulation Equation:
$$\text{Disparity } d = x_L - x_R$$
$$\text{Depth } Z = \frac{f \cdot B}{d}$$

- **Inverse Relationship**: As distance $Z \to \infty$, disparity $d \to 0$. Nearby objects produce large disparities.
