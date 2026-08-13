# 🔍 Phase 02 — Feature Engineering & Visual Descriptors

## 🎯 Phase Overview & Goal
- **Duration**: ~3 weeks
- **Objective**: Understand how classical vision extracted repeatable, invariant representations (corners, keypoints, local gradient descriptors) before deep neural networks, and how this bridged the gap between raw pixels and machine learning classifiers (SVMs, Random Forests).

---

## 1. Why Raw Pixels Fail as Features

Raw pixel values are brittle:
- **Translation / Shift**: Moving an object 5 pixels completely changes the pixel vector.
- **Illumination Changes**: Sunlight vs. shadow multiplies pixel intensities by a scalar factor.
- **Scale & Rotation**: Zooming in or rotating changes coordinates completely.

To recognize objects or match images across different cameras and viewpoints, we need **Local Invariant Features**:
1. **Detector**: Finds unique "points of interest" (corners, blobs, extrema).
2. **Descriptor**: Encodes the visual neighborhood around each keypoint into a numerical feature vector invariant to rotation, scale, and affine changes.

---

## 2. Harris Corner Detector

A corner is a point where image intensity changes significantly in all directions.

### The Autocorrelation Matrix (Structure Tensor):
For a small window shift $(\Delta x, \Delta y)$, the change in intensity $E(\Delta x, \Delta y)$ is approximated by Taylor expansion:
$$E(\Delta x, \Delta y) \approx \begin{bmatrix} \Delta x & \Delta y \end{bmatrix} \mathbf{M} \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}$$

where the Second Moment Matrix $\mathbf{M}$ is:
$$\mathbf{M} = \sum_{(x, y) \in W} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

Let eigenvalues of $\mathbf{M}$ be $\lambda_1, \lambda_2$:
- $\lambda_1 \approx 0, \lambda_2 \approx 0 \implies \textbf{Flat Region}$ (no intensity change in any direction).
- $\lambda_1 \gg 0, \lambda_2 \approx 0 \implies \textbf{Edge}$ (intensity changes only perpendicular to edge).
- $\lambda_1 \gg 0, \lambda_2 \gg 0 \implies \textbf{Corner}$ (intensity changes in both orthogonal directions).

```text
               Eigenvalue Spectrum Classification
           λ2 ▲
              │     CORNER REGION
              │     (λ1 >> 0, λ2 >> 0)
              │
              │
              │  EDGE           FLAT
              │  REGION         REGION
              └─────────────────────────► λ1
```

### Harris Response Function $R$:
To avoid expensive eigenvalue calculation:
$$R = \det(\mathbf{M}) - k \cdot (\text{Tr}(\mathbf{M}))^2 = (\lambda_1 \lambda_2) - k (\lambda_1 + \lambda_2)^2 \quad (k \approx 0.04 - 0.06)$$
- If $R > \text{threshold} \implies \text{Corner detected}$.

---

## 3. SIFT (Scale-Invariant Feature Transform)

David Lowe's SIFT algorithm (1999) achieved full scale, rotation, and illumination invariance through a 4-step pipeline:

```text
Scale-Space Extrema (DoG) ──► Keypoint Localization ──► Orientation Assignment ──► 128-D Feature Descriptor
```

### Step 1: Scale-Space Construction via Difference-of-Gaussians (DoG)
To find features invariant to scale, we construct an image pyramid convolved with Gaussians of increasing scale $\sigma$:
$$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)$$

Extrema (maxima/minima) are identified by comparing each pixel against its **26 neighbors** (8 in current scale, 9 in scale above, 9 in scale below).

```text
       Scale Octave n+1          Scale Octave n           Scale Octave n-1
       ┌───┬───┬───┐             ┌───┬───┬───┐            ┌───┬───┬───┐
       │   │   │   │             │   │   │   │            │   │   │   │
       ├───┼───┼───┤             ├───┼───┼───┤            ├───┼───┼───┤
       │   │ 9 │   │             │ 8 │ X │ 8 │            │   │ 9 │   │  <-- 26 Neighbors Test
       ├───┼───┼───┤             ├───┼───┼───┤            ├───┼───┼───┤
       │   │   │   │             │   │   │   │            │   │   │   │
       └───┴───┴───┘             └───┴───┴───┘            └───┴───┴───┘
```

### Step 2: Accurate Keypoint Sub-pixel Localization
Low-contrast extrema and unstable edge responses along ridges are eliminated using Hessian matrix thresholding.

### Step 3: Canonical Orientation Assignment
Compute local gradient magnitude and orientation in neighborhood. Create a 36-bin orientation histogram. The dominant peak gives the keypoint's canonical orientation $\theta_{main}$, achieving **rotation invariance**.

### Step 4: 128-Dimensional Descriptor Generation
Divide a $16 \times 16$ patch around the keypoint into **$4 \times 4$ grid of sub-regions** (each $4 \times 4$ pixels). In each sub-region, build an **8-bin gradient orientation histogram**:
$$\text{Descriptor Size} = 4 \times 4 \times 8 = \mathbf{128 \text{ floating-point dimensions}}$$

---

## 4. ORB (Oriented FAST & Rotated BRIEF)

While SIFT is patented/computationally heavy, ORB (2011) provides a real-time, patent-free binary descriptor:
1. **FAST Keypoint Detector**: Tests a circle of 16 pixels around candidate $p$. If $N \ge 12$ consecutive pixels are all brighter/darker than $I(p) \pm \epsilon$, mark as keypoint.
2. **Orientation via Intensity Centroid**: Calculates center of mass $C = \left(\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}\right)$ to find angle $\theta = \arctan2(m_{01}, m_{10})$.
3. **rBRIEF Binary Descriptor**: Performs 256 pairwise pixel intensity comparisons steered by orientation $\theta$. Produces a compact **256-bit string**.
4. **Matching via Hamming Distance**: Comparing two 256-bit descriptors is executed in 1 CPU cycle using `POPCNT(A XOR B)`.

---

## 5. HOG (Histogram of Oriented Gradients)

Popularized by Dalal & Triggs (2005) for pedestrian detection:

```text
Input Image ──► Normalized Gradients ──► 8x8 Pixel Cells (9-bin Histograms) ──► 2x2 Block L2-Norm ──► 1D Feature Vector ──► Linear SVM
```

- **Cells**: Divide image into $8 \times 8$ pixel cells. Compute 9-bin orientation histogram ($0^\circ - 180^\circ$ unsigned).
- **Blocks**: Group $2 \times 2$ cells into a block ($16 \times 16$ pixels). Normalize using $L_2\text{-norm}$:
  $$\mathbf{v}_{norm} = \frac{\mathbf{v}}{\sqrt{\|\mathbf{v}\|_2^2 + \epsilon^2}}$$
- Concatenate all normalized blocks across overlapping sliding windows to feed into a **Linear SVM**.

---

## 6. Feature Matching & Geometric Verification (RANSAC)

```text
Image 1 Keypoints ──► Descriptor Distance ──► Lowe's Ratio Test ──► RANSAC Homography ──► Robust Match
Image 2 Keypoints ──┘
```

- **Lowe's Ratio Test**: For a keypoint in Image 1, find nearest neighbor $D_1$ and 2nd nearest neighbor $D_2$ in Image 2. Accept match if:
  $$\frac{\text{dist}(D_1)}{\text{dist}(D_2)} < 0.75$$
- **RANSAC (Random Sample Consensus)**: Eliminates outlier matches by randomly sampling 4 point pairs, estimating Homography $\mathbf{H}$, and counting inliers until consensus is reached.

---

## 7. Bag of Visual Words (BoVW)

Analogous to NLP text documents:
1. Extract SIFT/ORB descriptors from thousands of training images.
2. Cluster descriptors using **K-Means** with $K = 1000$ centroids. Centroids form the **Visual Vocabulary** ("Codebook").
3. Quantize each new image into a histogram counting occurrences of visual words.
4. Train an **SVM or Random Forest** on the histograms to classify whole images.

---

## 🔄 Modern Paradigm Shift: Feature Engineering $\to$ Representation Learning

```text
Classical Paradigm:
Image ──► [Hand-crafted Math: SIFT/HOG] ──► Feature Vector ──► [Learned Classifier: SVM] ──► Output

Modern Deep Learning Paradigm:
Image ─────────────────────► [End-to-End Differentiable CNN / ViT] ─────────────────────► Output
```
Instead of humans designing features for 10 years, deep networks learn features tailored directly to the downstream objective via backpropagation.
