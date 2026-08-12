# 📐 Phase 00 — Mathematical & Computational Foundations of Computer Vision

## 🎯 Phase Overview & Goal
- **Duration**: ~2 weeks
- **Objective**: Master the exact subset of mathematics that Computer Vision actually uses in practice—no superfluous pure math, only the computational linear algebra, vector calculus, probability, and optimization that power spatial filtering, CNNs, 3D geometry, and latent spaces.

---

## 1. Linear Algebra for Visual Computing

### 1.1 Digital Images as Multidimensional Tensors
In Computer Vision, an image is not a picture; it is a matrix or a 3-rank tensor:
- **Grayscale Image**: $\mathbf{I} \in \mathbb{R}^{H \times W}$, where $I(y, x) \in [0, 255]$ or $[0.0, 1.0]$.
- **RGB Color Image**: $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ (or batch tensor $\mathbf{X} \in \mathbb{R}^{N \times C \times H \times W}$ in PyTorch), where $C = 3$ (Red, Green, Blue channels).

```text
       Width (W: X-axis)
      ┌───────────────────────┐
  H   │ (0,0)         (0, W)  │  --> Channel 0 (Red)
  e   │                       │  --> Channel 1 (Green)
  i   │                       │  --> Channel 2 (Blue)
  g   │                       │
  h   │ (H, 0)        (H, W)  │
  t   └───────────────────────┘
 (Y)
```

> [!NOTE]
> **Spatial vs. Matrix Indexing Convention**:
> In image coordinate space $(x, y)$, $x$ is horizontal (columns) and $y$ is vertical (rows). But in matrix array indexing $\mathbf{I}[i, j]$, $i$ is row index ($y$) and $j$ is column index ($x$). Hence, $I(x, y) \equiv \mathbf{I}[y, x]$.

---

### 1.2 Matrix Transformations & Geometric Warping
A 2D point $\mathbf{p} = \begin{bmatrix} x \\ y \end{bmatrix}$ is expressed in **Homogeneous Coordinates** as $\mathbf{\tilde{p}} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$.

Linear and affine geometric transformations in $\mathbb{P}^2$ are computed via matrix multiplication:
$$\mathbf{\tilde{p}}' = \mathbf{M} \mathbf{\tilde{p}} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

- **Rotation by angle $\theta$**:
  $$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
- **Translation by $(t_x, t_y)$**:
  $$\mathbf{T}(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$$
- **Scaling by $(s_x, s_y)$**:
  $$\mathbf{S}(s_x, s_y) = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

---

### 1.3 Eigenvalues, Eigenvectors & Principal Directions
For a square matrix $\mathbf{A}$, an eigenvector $\mathbf{v}$ and eigenvalue $\lambda$ satisfy:
$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v} \implies (\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = \mathbf{0}$$

**Applications in Computer Vision**:
1. **Harris Corner Detection**: The Structure Tensor $\mathbf{M} = \sum \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$ has eigenvalues $\lambda_1, \lambda_2$. If both $\lambda_1, \lambda_2$ are large, the local patch is a **corner**.
2. **Principal Component Analysis (PCA)**: Finding dominant axes of variation across image datasets (Eigenfaces, visual descriptors).
3. **Epipolar Geometry**: Solving for the Fundamental Matrix $\mathbf{F}$ and Essential Matrix $\mathbf{E}$ via Singular Value Decomposition (SVD).

---

## 2. 2D Discrete Convolution Mathematics

The 2D continuous convolution between an image $I(x, y)$ and a continuous filter kernel $K(u, v)$ is:
$$(I * K)(x, y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} I(\tau, \eta) K(x - \tau, y - \eta) \, d\tau \, d\eta$$

In digital image processing and neural networks, we compute the **Discrete 2D Cross-Correlation** (often referred to as convolution in Deep Learning):
$$S(i, j) = (I * K)(i, j) = \sum_{m=-k}^{k} \sum_{n=-k}^{k} I(i + m, j + n) K(m, n)$$

```text
 Input Image (5x5)          Kernel (3x3)           Output Feature Map (3x3)
┌───┬───┬───┬───┬───┐       ┌───┬───┬───┐         ┌────┬────┬────┐
│ 1 │ 2 │ 0 │ 1 │ 3 │       │ 1 │ 0 │-1 │         │ 3  │ 1  │ -2 │
├───┼───┼───┼───┼───┤   *   ├───┼───┼───┤   =     ├────┼────┼────┤
│ 0 │ 1 │ 1 │ 2 │ 0 │       │ 2 │ 0 │-2 │         │ 4  │ 0  │ -1 │
├───┼───┼───┼───┼───┤       ├───┼───┼───┤         ├────┼────┼────┤
│ 2 │ 3 │ 1 │ 0 │ 1 │       │ 1 │ 0 │-1 │         │ 2  │ 5  │  0 │
└───┴───┴───┴───┴───┘       └───┴───┴───┘         └────┴────┴────┘
```

### Output Dimensionality Formula:
Given an input of spatial size $W_{in} \times H_{in}$, kernel size $K$, padding $P$, stride $S$, and dilation $D$:
$$W_{out} = \left\lfloor \frac{W_{in} - D(K - 1) - 1 + 2P}{S} \right\rfloor + 1$$

---

## 3. Vector Calculus & Image Gradients

### 3.1 Spatial Image Derivatives
Since digital images are discrete, spatial derivatives are approximated using finite differences:
- Horizontal gradient $\frac{\partial I}{\partial x} \approx I(x + 1, y) - I(x - 1, y)$
- Vertical gradient $\frac{\partial I}{\partial y} \approx I(x, y + 1) - I(x, y - 1)$

The **Image Gradient Vector** $\nabla I$ points in the direction of greatest intensity change:
$$\nabla I = \begin{bmatrix} I_x \\ I_y \end{bmatrix} = \begin{bmatrix} \frac{\partial I}{\partial x} \\ \frac{\partial I}{\partial y} \end{bmatrix}$$

- **Gradient Magnitude** (Edge Strength):
  $$|\nabla I| = \sqrt{I_x^2 + I_y^2} \approx |I_x| + |I_y|$$
- **Gradient Orientation** (Edge Direction):
  $$\theta = \arctan2(I_y, I_x) \in [-\pi, \pi]$$

---

## 4. Optimization in Computer Vision

### 4.1 Gradient Descent & Backpropagation
To train a visual network with parameters $\mathbf{W}$ and loss function $\mathcal{L}(\mathbf{W})$:
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla_{\mathbf{W}} \mathcal{L}(\mathbf{W}_t)$$

In deep convolutional layers, the gradient of the loss with respect to input activations $\mathbf{X}$ is computed via cross-correlation with the spatially flipped kernel $\mathbf{K}^\top$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} * \text{rot}_{180}(\mathbf{K})$$

---

## 💻 Milestone Deliverable: Linear Algebra & Convolution in Pure NumPy

Below is a self-contained computational verification script:

```python
import numpy as np

def conv2d_scratch(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Computes 2D discrete convolution from first principles using pure NumPy."""
    H, W = image.shape
    kH, kW = kernel.shape
    
    # Apply Zero-Padding
    if padding > 0:
        padded_img = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        padded_img = image
        
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    output = np.zeros((out_H, out_W), dtype=np.float32)
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + kH
            w_start = j * stride
            w_end = w_start + kW
            
            patch = padded_img[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(patch * kernel)
            
    return output

if __name__ == "__main__":
    img = np.array([
        [10, 10, 10, 0, 0],
        [10, 10, 10, 0, 0],
        [10, 10, 10, 0, 0],
        [10, 10, 10, 0, 0],
        [10, 10, 10, 0, 0]
    ], dtype=np.float32)
    
    sobel_v = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    edges = conv2d_scratch(img, sobel_v, padding=1)
    print("Vertical Edge Detection Output:\n", edges)
```
