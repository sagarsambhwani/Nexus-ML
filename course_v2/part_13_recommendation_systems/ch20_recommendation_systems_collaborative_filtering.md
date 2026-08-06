# Chapter 20: Recommendation Systems, Matrix Factorization, and Ranking

---

## 1. Big Picture

Recommendation systems drive up to 30% of Amazon revenue and 80% of Netflix watch time. Unlike standard classification, recommendation operates over extreme user-item interaction matrices where 99.9% of entries are unobserved (extreme sparsity).

Modern production recommenders use a **Two-Stage Architecture**:
1. **Candidate Generation (Retrieval)**: Filtering 10,000,000 items down to 500 candidate items in $< 10\text{ms}$ using Matrix Factorization or Two-Tower Networks.
2. **Scoring & Ranking**: Ranking the 500 candidates using complex deep learning models (Deep & Cross Networks) to output top-10 personalized recommendations.

---

## 2. Intuition

- **Collaborative Filtering**: "Users who liked what you liked also bought this." If User A and User B both rated Movies 1, 2, 3 identically, and User B loved Movie 4, recommend Movie 4 to User A.
- **Matrix Factorization (SVD / ALS)**: Decomposing a huge sparse $M \times N$ User-Item rating matrix into two thin dense matrices: User Latent Vectors ($M \times k$) and Item Latent Vectors ($N \times k$).

---

## 3. Visualization

```text
Two-Stage Recommendation Architecture:

  10,000,000 Catalog Items
            │
            ▼
  [ Stage 1: Candidate Retrieval (Two-Tower / Matrix Factorization) ]
            │  (Sub-10ms, Returns Top-500 candidates)
            ▼
  [ Stage 2: Fine-Grained Ranking (Deep & Cross Network / XGBoost) ]
            │  (Predicts exact P(Click), Returns Top-10)
            ▼
  Top-10 Personalized Recommendations for User
```

---

## 4. Mathematics

### Alternating Least Squares (ALS) Matrix Factorization
Decomposing Rating matrix $R \approx U V^T$ where $U \in \mathbb{R}^{M \times k}$ and $V \in \mathbb{R}^{N \times k}$.

Objective with L2 Regularization:

$$\min_{U, V} \sum_{(u, i) \in \Omega} (R_{u,i} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda \left( \sum_u \|\mathbf{u}_u\|_2^2 + \sum_i \|\mathbf{v}_i\|_2^2 \right)$$

ALS optimizes by alternating:
1. Fix $V$, solve optimal $\mathbf{u}_u^*$ for each user independently in parallel:
$$\mathbf{u}_u^* = \left( V_{I_u}^T V_{I_u} + \lambda I \right)^{-1} V_{I_u}^T R_{u, I_u}$$
2. Fix $U$, solve optimal $\mathbf{v}_i^*$ for each item independently in parallel:
$$\mathbf{v}_i^* = \left( U_{I_i}^T U_{I_i} + \lambda I \right)^{-1} U_{I_i}^T R_{I_i, i}$$

---

## 5. Python (From Scratch Matrix Factorization via SGD)

```python
import numpy as np

class MatrixFactorizationScratch:
    def __init__(self, n_users, n_items, k=5, lr=0.01, reg=0.02):
        self.U = np.random.normal(0, 0.1, (n_users, k))
        self.V = np.random.normal(0, 0.1, (n_items, k))
        self.lr = lr
        self.reg = reg
        
    def fit(self, ratings: list[tuple[int, int, float]], epochs=100):
        for _ in range(epochs):
            for u, i, r in ratings:
                pred = np.dot(self.U[u], self.V[i])
                err = r - pred
                
                # Gradient updates
                u_old = self.U[u].copy()
                self.U[u] += self.lr * (err * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (err * u_old - self.reg * self.V[i])
                
    def predict(self, u, i):
        return np.dot(self.U[u], self.V[i])

# Test MF
ratings = [(0, 0, 5.0), (0, 1, 3.0), (1, 0, 4.0), (1, 2, 1.0)] # (user_id, item_id, rating)
mf = MatrixFactorizationScratch(n_users=2, n_items=3, k=2)
mf.fit(ratings, epochs=200)

print(f"Predicted Rating for User 0 on Item 2: {mf.predict(0, 2):.2f}")
```

---

## 6. Production Library (Implicit ALS & Two-Tower PyTorch)

```python
import implicit
import scipy.sparse as sparse

# 1. Construct Sparse User-Item Matrix
user_item_matrix = sparse.csr_matrix(([5.0, 3.0, 4.0, 1.0], ([0, 0, 1, 1], [0, 1, 0, 2])))

# 2. Production Implicit Alternating Least Squares (ALS)
model_als = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20)
model_als.fit(user_item_matrix)

# 3. Retrieve Top-2 Candidate Recommendations for User 0
ids, scores = model_als.recommend(0, user_item_matrix[0], N=2)
print("Top Recommended Item IDs:", ids)
```

---

## 7. Under the Hood

- `implicit` executes C++ OpenMP parallelized matrix multiplications and Conjugate Gradient linear solvers, fitting 10 million ratings in $< 5$ seconds.

---

## 8. Engineering Perspective

- **Cold Start Problem**: New users or items with zero historical interaction ratings cannot be factorized via collaborative filtering. Address cold start using **Content-Based Filtering** (item category tags, user demographic embeddings).

---

## 9. Common Mistakes

1. **Evaluating Recommenders with Classification Accuracy Only**: Evaluating recommenders with raw MSE or Accuracy instead of ranking metrics (**NDCG@K, Mean Reciprocal Rank MRR, Precision@K**).
2. **Ignoring Popularity Bias**: Recommending popular blockbusters to every user regardless of personal niche preferences.

---

## 10. Interview Questions

### Q1: What is Normalized Discounted Cumulative Gain (NDCG@K) and how is it calculated?
**Answer**: NDCG@K measures ranking quality by penalizing relevant items placed lower in recommendation lists.
$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
Where IDCG@K is the Ideal DCG score obtained by perfect sorting of relevant items.

---

## 11. Exercises

1. **Math**: Derive the optimal user vector update equation $\mathbf{u}_u^*$ in ALS by taking the gradient of the objective function.
2. **Coding**: Implement NDCG@10 calculation function in NumPy.

---

## 12. Mini Project: E-Commerce Recommendation Engine

Write a script `recommendation_engine.py` that fits an Implicit ALS candidate generator, feeds candidates into an XGBoost ranker, and logs NDCG@10 metrics.

---

## 13. Capstone Integration

Implemented in `src/recommendation/pipeline.py`.
