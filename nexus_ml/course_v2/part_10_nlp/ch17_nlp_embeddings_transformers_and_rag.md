# Chapter 17: Natural Language Processing, Embeddings, and RAG Systems

---

## 1. Big Picture

Natural Language Processing (NLP) has evolved from static rule-based parsing to dense vector space modeling and Retrieval-Augmented Generation (RAG).

This chapter covers text preprocessing, tokenization (BPE, WordPiece), dense embeddings (Word2Vec, BERT embeddings), vector similarity search (HNSW, FAISS), and building production **RAG (Retrieval-Augmented Generation)** pipelines.

---

## 2. Intuition

- **Subword Tokenization (BPE)**: Splitting rare words into subwords (`unhelpfulness` $\to$ `un` + `help` + `fulness`), solving out-of-vocabulary (OOV) errors.
- **RAG Architecture**: LLMs have a fixed knowledge cutoff and tend to hallucinate private company data. RAG acts as an "Open-Book Exam": when a user asks a question, a retriever searches private documents, extracts relevant chunks, and inserts them into the LLM prompt context window.

---

## 3. Visualization

```text
Retrieval-Augmented Generation (RAG) Architecture:

  1. User Query ──► [ Embedding Model ] ──► Query Vector
                                                │
                                                ▼
  Private Docs  ──► [ Chunk & Vectorize ] ──► [ Vector DB (FAISS/HNSW) ]
                                                │ (Top-K Similarity Search)
                                                ▼
                                    Retrieved Context Chunks
                                                │
                                                ▼
                                 [ LLM Prompt Context ] ──► Final Answer
```

---

## 4. Mathematics

### Cosine Similarity over Dense Embeddings
Given Query embedding $\mathbf{q} \in \mathbb{R}^d$ and Document Chunk embedding $\mathbf{d}_i \in \mathbb{R}^d$:

$$\text{Sim}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\|_2 \|\mathbf{d}_i\|_2}$$

### Vector Index Search (HNSW - Hierarchical Navigable Small World)
HNSW constructs a multi-layer graph where upper layers contain sparse long-range highway links and bottom layers contain dense local neighbor links, executing sub-linear search in $O(\log N)$ time.

---

## 5. Python (From Scratch Vector Search Engine)

```python
import numpy as np

class MinimalVectorStore:
    def __init__(self, dim=4):
        self.dim = dim
        self.vectors = []
        self.documents = []
        
    def add_document(self, text: str, vector: list[float]):
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        vec_norm = vec / (norm + 1e-8)  # L2 Normalize
        
        self.vectors.append(vec_norm)
        self.documents.append(text)
        
    def search(self, query_vector: list[float], top_k=2):
        q = np.array(query_vector, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        
        matrix = np.vstack(self.vectors) # N x d
        scores = matrix @ q_norm         # Cosine Similarity via Matrix Vector Product
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[idx], float(scores[idx])) for idx in top_indices]

# Test Vector Store
store = MinimalVectorStore(dim=3)
store.add_document("Machine learning pipeline optimization", [0.9, 0.1, 0.2])
store.add_document("FastAPI web backend server", [0.1, 0.8, 0.9])
store.add_document("Deep learning neural network models", [0.8, 0.2, 0.3])

results = store.search([0.85, 0.15, 0.1], top_k=2)
print("Top-2 Retrieved Documents:")
for doc, score in results:
    print(f"  ├─ Document: '{doc}' (Similarity Score: {score:.4f})")
```

---

## 6. Production Library (Sentence-Transformers & FAISS)

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load Pretrained Dense Embedding Model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
docs = [
    "Nexus-ML enterprise pipeline architecture",
    "PostgreSQL relational database queries",
    "PyTorch neural network training loop"
]
embeddings = embedder.encode(docs, normalize_embeddings=True)

# 2. Build In-Memory FAISS HNSW Index
dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(np.array(embeddings, dtype=np.float32))

# 3. Perform Similarity Search
query_emb = embedder.encode(["ML pipeline architecture"], normalize_embeddings=True)
distances, indices = index.search(np.array(query_emb, dtype=np.float32), k=1)
print(f"Retrieved Document: '{docs[indices[0][0]]}' (Distance: {distances[0][0]:.4f})")
```

---

## 7. Under the Hood

- `faiss.IndexHNSWFlat` stores vectors in aligned C memory buffers, using SIMD AVX2 distance calculation kernels to evaluate 8 vector dot products per CPU instruction.

---

## 8. Engineering Perspective

- **Chunking Strategy**: Avoid naive character splitting (`chunk_size=500`). Use Semantic Chunking or Recursive Text Splitters that respect sentence and paragraph boundaries, maintaining metadata (headers, source URLs).

---

## 9. Common Mistakes

1. **Embedding Un-normalized Vectors with Inner Product**: Comparing raw un-normalized vectors with inner product leads to length bias (longer documents get higher scores regardless of relevance). Always $L_2$ normalize vectors first!
2. **Ignoring Chunk Overlap**: Setting chunk overlap to 0 splits critical context across chunk boundaries. Use 10-20% chunk overlap.

---

## 10. Interview Questions

### Q1: Compare Sparse Retrieval (BM25) vs Dense Retrieval (Embedding Similarity).
**Answer**: Sparse retrieval (BM25) matches exact keyword occurrences using term frequency and inverse document frequency, making it robust for technical jargon, SKUs, and exact match names. Dense retrieval embeds semantic meanings into continuous vector space, catching synonyms and intent even without exact keyword overlap. Production RAG uses **Hybrid Search** (BM25 + Dense Vectors) with Reranking (Cross-Encoder).

---

## 11. Exercises

1. **Coding**: Implement a Reciprocal Rank Fusion (RRF) algorithm combining BM25 keyword ranks and FAISS dense vector ranks.
2. **Architecture**: Design an enterprise RAG pipeline supporting PDF document parsing, hybrid search, and hallucination evaluation.

---

## 12. Mini Project: Enterprise RAG Service

Build a microservice `rag_service.py` using FastAPI, Sentence-Transformers, FAISS, and an LLM API that ingests text files and answers questions with citation links.

---

## 13. Capstone Integration

Implemented in `src/document_classification/pipeline.py` and `src/sentiment_analysis/pipeline.py`.
