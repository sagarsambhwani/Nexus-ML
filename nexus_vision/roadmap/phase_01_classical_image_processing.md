# 🖼️ Phase 01 — Classical Image Processing

## 🎯 Phase Overview & Goal
- **Duration**: ~3 weeks
- **Objective**: Understand low-level image operations from first principles. Before deep learning, computer vision relied on analytical spatial filters, morphology, and frequency analysis. We master these fundamentals to understand the exact operations that modern CNNs learn automatically.

---

## 1. Digital Image Representation & Color Spaces

### 1.1 Color Spaces
Human color perception is trichromatic, but RGB is not the only useful representation for computer vision:

| Color Space | Channels | Best Use Case | Mathematical Advantage |
|---|---|---|---|
| **RGB** | Red, Green, Blue | Display Hardware & Deep CNN inputs | Additive color model; default camera sensor output. |
| **HSV / HSL** | Hue, Saturation, Value/Lightness | Color-based Object Segmentation | Separates chromaticity (Hue) from illumination (Value). Lighting changes only alter $V$, keeping $H$ stable. |
| **Grayscale** | Single Intensity ($Y$) | Edge detection, Feature extraction, OCR | $Y = 0.299R + 0.587G + 0.114B$. Eliminates color noise, reduces memory by $3\times$. |
| **YCrCb / LAB** | Luminance + Color Opponents | Compression, Skin detection, CLAHE | Perceptually uniform color space where Euclidean distance approximates human color difference. |

```text
       RGB Color Cube                        HSV Cylinder
    ┌─────────────────┐                     Value (Lightness)
   /                 /│                            ▲
  /                 / │                            │   / Hue (Angle 0-360)
 ┌─────────────────┐  │                            │  /
 │      White      │  │                            │ /
 │                 │  │                       ┌────┴────┐
 │                 │  │                       │         │
 │                 │ ┌┘                       │ ◄───────┼── Saturation (Radius)
 │      Black      │/                         └─────────┘
 └─────────────────┘
```

---

## 2. Histograms & Contrast Enhancement

### 2.1 Histogram Equalization & CLAHE
An image histogram $h(r_k) = n_k$ counts the frequency of pixel intensity $r_k \in [0, L-1]$.

**Global Histogram Equalization** flattens the cumulative distribution function (CDF):
$$s_k = T(r_k) = (L - 1) \sum_{j=0}^{k} p_r(r_j) = \frac{L - 1}{MN} \sum_{j=0}^{k} n_j$$

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
Dividing the image into non-overlapping grid tiles (e.g. $8 \times 8$), equalizing histograms locally, and clipping local contrast peaks to prevent over-amplifying background noise.

---

## 3. Spatial Filtering Kernels & Denoising

### 3.1 Gaussian Smoothing Filter
Used to attenuate high-frequency Gaussian noise before gradient computation:
$$G(x, y; \sigma) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$

A $5 \times 5$ normalized discrete Gaussian kernel with $\sigma = 1.0$:
$$K_G = \frac{1}{273} \begin{bmatrix} 1 & 4 & 7 & 4 & 1 \\ 4 & 16 & 26 & 16 & 4 \\ 7 & 26 & 41 & 26 & 7 \\ 4 & 16 & 26 & 16 & 4 \\ 1 & 4 & 7 & 4 & 1 \end{bmatrix}$$

### 3.2 Median Filter (Non-Linear)
Replaces every pixel with the statistical median of its neighborhood.
- **Superpower**: Completely removes "Salt & Pepper" (impulse) noise without blurring sharp edge boundaries.

### 3.3 Bilateral Filter (Edge-Preserving Smoothing)
Combines spatial proximity with radiometric (intensity) similarity:
$$I^{filtered}(p) = \frac{1}{W_p} \sum_{q \in S} I(q) \cdot G_{\sigma_s}(\|p - q\|) \cdot G_{\sigma_r}(|I(p) - I(q)|)$$

---

## 4. Edge Detection & The Canny Algorithm

Edges correspond to local maxima of the image gradient magnitude $|\nabla I|$.

```text
Raw Image ──► Gaussian Blur ──► Sobel Gradients (Ix, Iy) ──► Non-Max Suppression ──► Hysteresis Thresholding ──► Binary Edges
```

### The 5 Steps of the Canny Edge Detector:
1. **Gaussian Smoothing**: Convolve with Gaussian kernel to remove high-frequency noise.
2. **Gradient Computation**: Apply Sobel horizontal ($K_x$) and vertical ($K_y$) operators to calculate gradient magnitude $M(x,y)$ and orientation $\theta(x,y)$.
   $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
3. **Non-Maximum Suppression (NMS)**: Thin edges by checking if the pixel is a local maximum along the gradient normal vector $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$.
4. **Double Thresholding**: Categorize candidate edge pixels using thresholds $(T_{low}, T_{high})$:
   - $M(x, y) \ge T_{high} \implies \text{Strong Edge}$
   - $T_{low} \le M(x, y) < T_{high} \implies \text{Weak Edge}$
   - $M(x, y) < T_{low} \implies \text{Suppressed (Non-edge)}$
5. **Edge Tracking by Hysteresis**: Keep weak edge pixels if and only if they are connected (8-way neighborhood) to at least one strong edge pixel.

---

## 5. Morphological Operations

Binary morphology operates on binary shapes using a **Structuring Element** (Kernel $\mathbf{B}$):

- **Erosion ($A \ominus B$)**: Strips boundary pixels. Shrinks foreground objects, eliminates isolated pixel noise.
  $$(A \ominus B)(z) = \{z \mid (B)_z \subseteq A\}$$
- **Dilation ($A \oplus B$)**: Expands object boundaries. Fills holes and bridges small gaps.
  $$(A \oplus B)(z) = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$$
- **Opening ($A \circ B = (A \ominus B) \oplus B$)**: Erosion followed by Dilation. Smooths object contours, removes small protrusions.
- **Closing ($A \bullet B = (A \oplus B) \ominus B$)**: Dilation followed by Erosion. Fills interior holes and joins narrow breaks.

```text
    Original          Erosion           Dilation          Opening           Closing
  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │  ██████  │      │   ████   │      │ ████████ │      │  ██████  │      │  ██████  │
  │  ██░░██  │  ──► │   ████   │  ──► │ ████████ │  ──► │  ██████  │  ──► │  ██████  │
  │  ██████  │      │   ████   │      │ ████████ │      │  ██████  │      │  ██████  │
  └──────────┘      └──────────┘      └──────────┘      └──────────┘      └──────────┘
```

---

## 6. Hough Transform for Geometric Primitives

To detect lines $y = mx + c$, Cartesian space is parameterized in **Hesse Normal Form**:
$$\rho = x \cos\theta + y \sin\theta$$

```text
 Spatial Image Space (x, y)                Hough Parameter Space (rho, theta)
         y ▲                                       rho ▲
           │  ● (x1, y1)                               │        /~~\
           │    \                                      │       /    \  <-- Sinusoid 1
           │      \                                    │      /      \
           │        ● (x2, y2)                         │-----(* INTERSECT *)---
           └──────────────► x                          │      \      / <-- Sinusoid 2
                                                       └───────\────/────► theta
```
Every point $(x, y)$ in image space maps to a sinusoidal curve in $(\rho, \theta)$ accumulator space. The intersection point of multiple sinusoidal curves identifies the collinear line parameters.

---

## 💡 The Modern Deep Learning Connection

> [!IMPORTANT]
> **CNN Convolution is Learned Filtering**:
> In classical vision, an engineer had to manually design the Sobel kernel $\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ to detect vertical edges, or a Laplacian kernel to detect blobs.
>
> In a modern Deep Convolutional Neural Network (CNN), filter weights $W_{ij}$ are initialized randomly and **learned automatically via backpropagation** to optimize classification or detection loss. The early layers of AlexNet and ResNet naturally converge to Gabor-like directional edge and color-opponent filters!
