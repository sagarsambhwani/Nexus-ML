# 📕 Chapter 4: Dimensionality Reduction & PCA Mathematics

## 4.1 The Curse of Dimensionality
As feature dimensionality $p$ grows, the volume of the feature space increases exponentially:

$$\text{Volume} \propto \Theta(d^p)$$

### Consequences of High Dimensionality:
1. **Data Sparsity**: Training samples become exceedingly sparse in high-dimensional space, making distance calculations (e.g. Euclidean distance in $K$-Means or KNN) uninformative.
2. **Overfitting**: High feature-to-sample ratios ($p \gg N$) lead models to memorize noise.
3. **Computational Complexity**: Matrix operations (such as matrix inversion $O(p^3)$) slow down dramatically.

Dimensionality reduction compresses a high-dimensional space $\mathbb{R}^p$ into a lower-dimensional subspace $\mathbb{R}^k$ ($k \ll p$) while retaining maximum variance.

---

## 4.2 Mathematical Derivation of Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is an unsupervised orthogonal linear transformation that identifies directions (principal components) maximizing data variance.

```
       Original 2D Features (X1, X2)               1st Principal Component (PC1)
                X2                                         Variance Maxized
                ▲                                                 ╱ (PC1)
                │      • •                                      ╱ • •
                │    • • •                                    ╱ • • •
                │  • •                                      ╱ • •
                └───────────────► X1                       └───────────────►
```

### Step-by-Step Derivation:

#### 1. Mean Centering & Standardization
Given dataset $X \in \mathbb{R}^{N \times p}$, center features to zero mean:
$$\tilde{X} = X - \mu_X$$

#### 2. Covariance Matrix Calculation
The empirical covariance matrix $\Sigma \in \mathbb{R}^{p \times p}$ is:
$$\Sigma = \frac{1}{N-1} \tilde{X}^T \tilde{X}$$

#### 3. Eigen-Decomposition
Find eigenvectors $v$ and eigenvalues $\lambda$ satisfying:
$$\Sigma v = \lambda v$$

In matrix form:
$$\Sigma = V \Lambda V^T$$
where $V = [v_1, v_2, \dots, v_p]$ are orthogonal principal direction eigenvectors, and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_p)$ contains eigenvalues sorted in descending order ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0$).

#### 4. Projecting Data onto Principal Components
Transform data onto top $k$ principal components:
$$Z_k = \tilde{X} V_k \in \mathbb{R}^{N \times k}$$

---

## 4.3 Singular Value Decomposition (SVD) Approach

In production (e.g. `scikit-learn`), PCA is computed directly via **Singular Value Decomposition (SVD)** on $\tilde{X}$ without explicitly constructing the covariance matrix $\Sigma$:

$$\tilde{X} = U \Sigma_{\text{svd}} V^T$$

- $U \in \mathbb{R}^{N \times N}$: Left singular vectors (user/sample embeddings).
- $\Sigma_{\text{svd}} \in \mathbb{R}^{N \times p}$: Singular values $s_i = \sqrt{(N-1)\lambda_i}$.
- $V \in \mathbb{R}^{p \times p}$: Right singular vectors (principal component loadings).

Using SVD is numerically more stable than explicit covariance eigen-decomposition.

---

## 4.4 Explained Variance Ratio & Component Selection

The proportion of total variance explained by the $j$-th principal component is:

$$\text{Explained Variance Ratio}_j = \frac{\lambda_j}{\sum_{i=1}^p \lambda_i}$$

The cumulative explained variance for $k$ components is:

$$\text{Cumulative Variance}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^p \lambda_i}$$

```
                Scree / Cumulative Variance Plot
     100% ┼─────────────────────────────────-───-── (Target 95% Cutoff)
          │                                 ╭───────
      80% ┼                           ╭─────╯
          │                     ╭─────╯
      60% ┼               ╭─────╯
          │         ╭─────╯
      40% ┼   ╭─────╯
          └───┴─────┴─────┴─────┴─────┴─────┴─────►
              1     2     3     4     5     6     7   Component (k)
```

---

## 4.5 TruncatedSVD vs. Standard PCA

| Property | Standard PCA | TruncatedSVD |
|---|---|---|
| **Mean Centering** | Requires mean centering $\tilde{X} = X - \mu$ | Does **not** mean center data |
| **Sparse Matrix Support** | Fails (mean centering destroys matrix sparsity) | Fully supports sparse matrices (`scipy.sparse`) |
| **Primary Use Cases** | Tabular features, visual scatter mapping | NLP TF-IDF matrices, Collaborative Filtering rating matrices |

---

## 4.6 Python Implementation & 2D Projection

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 2. Fit PCA for 2D visual projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 3. Inspect variance ratios
print(f"PC1 Variance Ratio: {pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2 Variance Ratio: {pca.explained_variance_ratio_[1]:.4f}")
```

---

## ⚓ Repository Code Reference
- See [`src/customer_segmentation/pipeline.py`](file:///e:/Downloads/ML_only/src/customer_segmentation/pipeline.py) for PCA 2D projection mapping ($PC_1, PC_2$) of customer RFM clusters.
- See [`src/recommendation/pipeline.py`](file:///e:/Downloads/ML_only/src/recommendation/pipeline.py) for matrix factorization using `TruncatedSVD`.
