# 🕵️ Chapter 14: Unsupervised Anomaly Detection & Density Estimation

## 14.1 Introduction to Anomaly Detection
Anomaly detection identifies patterns in data that do not conform to expected statistical behavior. In enterprise systems, anomalies correspond to credit card fraud, industrial machine breakdown, network intrusions, or medical defects.

---

## 14.2 Density Estimation: Gaussian Mixture Models (GMM)

A **Gaussian Mixture Model** models a complex probability density function $p(x)$ as a weighted sum of $K$ multivariate Gaussian components:

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

where $\sum_{k=1}^K \pi_k = 1$, and component probability density is:

$$\mathcal{N}(x \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)$$

```
                                  GMM Mixture Density
                      Density p(x)
                          ▲           Component 1       Component 2
                          │              ╭───╮             ╭───╮
                          │            ╭─╯   ╰─╮         ╭─╯   ╰─╮
                          │          ╭─╯       ╰─╮     ╭─╯       ╰─╮
                          └──────────┴─────────┴───────┴─────────┴───► x
```

### Expectation-Maximization (EM) Algorithm:
1. **E-Step (Responsibility Calculation)**:
   $$\gamma_{i, k} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
2. **M-Step (Parameter Update)**:
   $$N_k = \sum_{i=1}^N \gamma_{i, k}, \quad \mu_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{i, k} x_i, \quad \Sigma_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{i, k} (x_i - \mu_k)(x_i - \mu_k)^T, \quad \pi_k = \frac{N_k}{N}$$

Points with low likelihood $p(x_i) < \tau$ are flagged as statistical anomalies.

---

## 14.3 Spatial Distance Anomaly Detection

### 1. Mahalanobis Distance
Measures distance between point $x$ and distribution mean $\mu$, accounting for feature covariances $\Sigma$:

$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Unlike standard Euclidean distance, Mahalanobis distance forms elliptical equiprobability contours, properly scoring correlated feature anomalies.

```
       Euclidean Distance (Spherical Circles)       Mahalanobis Distance (Covariance Ellipses)
                 x2                                          x2
                 ▲                                           ▲
                 │     ╭───╮                                 │       ╭─────╮
                 │    │  •  │                                │     ╭─╯  •  ╰─╮
                 │     ╰───╯                                 │    ╭╯         ╰╮
                 └───────────► x1                            └────┴───────────┴──► x1
```

### 2. Local Outlier Factor (LOF)
Compares the local density of a sample $x$ to the local densities of its $k$-nearest neighbors:

$$\text{LOF}_k(x) = \frac{\sum_{o \in N_k(x)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(x)}}{|N_k(x)|}$$

If $\text{LOF}(x) \gg 1.0$, the sample resides in a sparse region relative to its neighbors (local anomaly).

---

## 14.4 Deep Anomaly Detection: Autoencoders & VAEs

### 1. Autoencoders (Reconstruction Error)
An Autoencoder compresses input $x \in \mathbb{R}^d$ into bottleneck code $z \in \mathbb{R}^k$ ($k \ll d$) via encoder $f_\theta$, and reconstructs $\hat{x}$ via decoder $g_\phi$:

$$z = f_\theta(x), \quad \hat{x} = g_\phi(z)$$

```
Input x (High-Dim) ──► Encoder f_θ ──► Bottleneck z (Low-Dim) ──► Decoder g_φ ──► Reconstructed x̂
```

- **Anomaly Score**: Mean Squared Reconstruction Error:
  $$\text{Score}(x) = \|x - \hat{x}\|_2^2 = \sum_{j=1}^d (x_j - \hat{x}_j)^2$$
- When trained exclusively on normal operational data, normal inputs yield low reconstruction errors, while novel anomalies produce high reconstruction spikes.

### 2. Variational Autoencoders (VAE)
Models continuous latent spaces $z \sim N(\mu_z, \Sigma_z)$ using the **Evidence Lower Bound (ELBO)** loss:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right]}_{\text{Reconstruction Term}} - \underbrace{\text{KL}\Big( q_\phi(z \mid x) \;\parallel\; p(z) \Big)}_{\text{Kullback-Leibler Divergence Regularization}}$$

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for transaction anomaly feature scoring.
- See [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py) for patch visual anomaly metric evaluations.
