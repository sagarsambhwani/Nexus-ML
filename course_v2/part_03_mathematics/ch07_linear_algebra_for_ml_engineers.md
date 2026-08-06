# Chapter 07: Linear Algebra for Machine Learning Engineers

---

## 1. Big Picture

Machine Learning is applied linear algebra. Data instances are vectors in $\mathbb{R}^d$, feature representations are linear transformations, deep learning layers are matrix multiplications, and dimensionality reduction is matrix decomposition.

Without a deep understanding of vector spaces, matrix rank, Singular Value Decomposition (SVD), and Eigendecomposition, ML engineers cannot diagnose numerical instability, exploding/vanishing gradients, or catastrophic projection collapse.

---

## 2. Intuition

- **Vectors**: Arrows pointing in multi-dimensional feature space. A house with [3 bedrooms, 2000 sqft, $500k price] is a point/vector in $\mathbb{R}^3$.
- **Matrices as Transformations**: A matrix $A \in \mathbb{R}^{m \times n}$ is not a static grid of numbers; it is a linear function that rotates, stretches, shear-transforms, or projects $n$-dimensional space into $m$-dimensional space.
- **Eigenvectors & Eigenvalues**: The special invariant directions of a linear transformation that do not rotate—they are only stretched or shrunk by factor $\lambda$.

---

## 3. Visualization

```text
Linear Transformation A x = y:
  Vector x (Unit Circle) ──► [ Matrix Transformation A ] ──► Rotated & Stretched Ellipse y
                                                                    ▲
                                                                    │
                                                     Major axis = Eigenvector 1 (Length λ1)
                                                     Minor axis = Eigenvector 2 (Length λ2)
```

---

## 4. Mathematics

### 1. Vector Dot Product & Cosine Similarity
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$$

$$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### 2. Matrix Rank & Condition Number
$$\text{Rank}(A) = \text{dim}(\text{colsp}(A))$$

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

If $\kappa(A)$ is very large ($\gg 10^4$), matrix $A$ is ill-conditioned, causing catastrophic numeric rounding errors during linear system solving.

### 3. Singular Value Decomposition (SVD)
Any matrix $A \in \mathbb{R}^{m \times n}$ can be factored into:

$$A = U \Sigma V^T$$

Where:
- $U \in \mathbb{R}^{m \times m}$ is orthogonal ($U^T U = I_m$) storing left-singular vectors.
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal containing singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$.
- $V \in \mathbb{R}^{n \times n}$ is orthogonal ($V^T V = I_n$) storing right-singular vectors.

---

## 5. Python (From Scratch SVD & PCA)

Implementing SVD and Low-Rank Reconstruction in Pure Python & NumPy:

```python
import numpy as np

# 1. Generate Low-Rank Matrix (Rank 2 + Noise)
np.random.seed(42)
u_true = np.random.randn(100, 2)
v_true = np.random.randn(2, 50)
A = u_true @ v_true + np.random.normal(0, 0.01, size=(100, 50))

# 2. Compute SVD using NumPy
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# 3. Truncate SVD to Rank k=2 (Compression / Denoising)
k = 2
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

A_reconstructed = U_k @ S_k @ Vt_k

# 4. Measure Reconstruction Error (Frobenius Norm)
frob_error = np.linalg.norm(A - A_reconstructed, ord='fro')
print(f"Original Rank: {np.linalg.matrix_rank(A)}")
print(f"Truncated Rank-2 Reconstruction Frobenius Error: {frob_error:.4f}")
print(f"Variance Preserved (Top 2 Singular Values): {np.sum(S[:k]**2) / np.sum(S**2):.4%}")
```

---

## 6. Production Library (Scikit-Learn TruncatedSVD)

```python
from sklearn.decomposition import TruncatedSVD

# Production SVD for high-dimensional sparse text representations
svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42)
A_dense = svd.fit_transform(A)

print("Scikit-learn Explained Variance Ratio:", svd.explained_variance_ratio_)
```

---

## 7. Under the Hood

- `np.linalg.svd` invokes LAPACK routines `dgesdd` or `dgesvd` compiled from Fortran binaries.
- LAPACK uses divide-and-conquer QR reduction strategies to compute singular values with double-precision IEEE 754 floating-point accuracy ($10^{-16}$).

---

## 8. Engineering Perspective

- **Memory Compression**: Storing full matrix $A (10,000 \times 10,000)$ takes $800\text{ MB}$. Storing rank-50 truncated SVD $U_k \Sigma_k V_k^T$ takes $10,000 \times 50 \times 2 \times 8\text{ bytes} \approx 8\text{ MB}$ (100x compression!).
- **Numerical Stability**: Avoid computing $(X^T X)^{-1}$ directly! Use pseudo-inverse $X^+ = V \Sigma^+ U^T$ or QR decomposition $X = QR$ to solve least squares.

---

## 9. Common Mistakes

1. **Inverting Singular / Ill-conditioned Matrices**: Calling `np.linalg.inv(A)` when $\det(A) \approx 0$ leading to infinity / NaN overflow.
2. **Ignoring Zero-Centering before SVD/PCA**: PCA requires subtracting column means before computing covariance matrices or SVD.

---

## 10. Interview Questions

### Q1: What is the relationship between SVD and Eigendecomposition of covariance matrix $X^T X$?
**Answer**: If $X = U \Sigma V^T$, then $X^T X = (V \Sigma U^T)(U \Sigma V^T) = V \Sigma^2 V^T$. The right-singular vectors $V$ of $X$ are the eigenvectors of $X^T X$, and the singular values $\sigma_i$ are the square roots of the eigenvalues $\lambda_i$ of $X^T X$ ($\sigma_i = \sqrt{\lambda_i}$).

---

## 11. Exercises

1. **Math**: Derive the closed-form solution of Ordinary Least Squares $\hat{\beta} = (X^T X)^{-1} X^T y$ using vector calculus.
2. **Coding**: Implement randomized SVD (Halko et al. algorithm) in NumPy and benchmark speed vs standard SVD on a $5000 \times 5000$ matrix.

---

## 12. Mini Project: Image Compression via SVD

Write a Python script `image_compressor.py` that loads a high-resolution grayscale image, performs SVD, truncates at $k \in \{5, 20, 50, 100\}$, and calculates compression ratios and PSNR (Peak Signal-to-Noise Ratio).

---

## 13. Capstone Integration

Underpins PCA feature extraction in `src/house_prices/pipeline.py` and latent semantic embeddings in `src/document_classification/pipeline.py`.
