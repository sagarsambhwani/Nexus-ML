# 🌌 Chapter 43: Unsupervised Density Estimation (GMM & EM Algorithm)

## 43.1 Density Estimation
Density estimation models the probability density function $p(x)$ of unannotated data $X$.

---

## 43.2 Gaussian Mixture Models (GMM)
GMM models data as a linear combination of $K$ Gaussian distributions:

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

Optimized iteratively via the **Expectation-Maximization (EM)** algorithm (E-step responsibilities $\gamma_{i,k}$, M-step parameter updates $\mu_k, \Sigma_k, \pi_k$).

---

## ⚓ Repository Code Reference
- See [`src/customer_segmentation/pipeline.py`](file:///e:/Downloads/ML_only/src/customer_segmentation/pipeline.py) for unsupervised cluster modeling.
