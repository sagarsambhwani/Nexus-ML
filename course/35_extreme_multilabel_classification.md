# 🏷️ Chapter 35: Extreme Multi-Label Classification (XMLC) & Search Indexing

## 35.1 The Challenge of Extreme Multi-Label Scale
Extreme Multi-Label Classification (XMLC) addresses high-dimensional learning problems where each sample must be tagged with a small subset of relevant labels from a massive candidate label space containing **hundreds of thousands or millions of categories** ($K > 1,000,000$).

```
Input Document (Product / Query) ──► XMLC Dual-Encoder Bi-Attentive Model ──► Top-K Relevant Labels (out of 1,000,000+ candidate tags)
```

---

## 35.2 Tree-Based Hierarchical Label Partitioning (PLT)

Standard One-vs-All Softmax scales at $O(d \cdot K)$, making gradient computation impossible for $K=1,000,000$. **Probabilistic Label Trees (PLT)** partition the label space into a balanced binary tree $T$ of height $\log_2(K)$.

```
                                  Probabilistic Label Tree (PLT)
                                            Root Node (r)
                                         ╱                 ╲
                                    Node 1                 Node 2
                                    ╱    ╲                 ╱    ╲
                                  L1      L2             L3      L4  (Leaf Labels)
```

### Probability Decomposition:
The probability of predicting leaf label $l$ given input $x$ decomposes into the product of binary decisions along path $P(l)$ from root $r$ to leaf $l$:

$$P(y_l = 1 \mid x) = \prod_{v \in P(l)} P(z_v = 1 \mid x, z_{\text{par}(v)} = 1)$$

Reduces inference search time from linear $O(K)$ to logarithmic $O(\log K)$.

---

## 35.3 Dual-Encoder Dense Retrieval & HNSW Indexing

Modern XMLC architectures combine dense Transformer text encoders with **Approximate Nearest Neighbor (ANN)** graph search.

```
Document Text x ──► Query Encoder E_q(x) ──► Vector q (d=128) ──┐
                                                                 ├──► Cosine Similarity ──► HNSW Graph Search ──► Top-K Items
Label Catalog l ──► Label Encoder E_l(l) ──► Vector l (d=128) ──┘
```

### Hierarchical Navigable Small World (HNSW) Graphs:
HNSW constructs a multi-layer graph index where upper layers contain long-range highway links for fast coarse routing, and bottom layers contain dense local neighbor links for fine-grained retrieval.

---

## 35.4 XMLC Evaluation Metrics: Precision@K & nDCG@K

1. **Precision@K ($P@K$)**:
   $$P@K = \frac{1}{K} \sum_{l \in \text{TopK}(\hat{y})} y_l$$
2. **Normalized Discounted Cumulative Gain ($nDCG@K$)**:
   $$\text{DCG}@K = \sum_{i=1}^K \frac{y_{\pi(i)}}{\log_2(i + 1)}, \quad \text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

## ⚓ Repository Code Reference
- See [`src/document_classification/pipeline.py`](file:///e:/Downloads/ML_only/src/document_classification/pipeline.py) for multi-class NLP text classification pipelines.
