# 🔤 Chapter 17: NLP — From Word Embeddings to LLMs & RAG

## 17.1 Evolution of Word Embeddings
Traditional Bag-of-Words and TF-IDF models ignore semantic word relationships (e.g. treating "king" and "queen" as completely orthogonal vectors). Distributed dense word embeddings map words into continuous low-dimensional vector spaces $\mathbb{R}^d$ ($d \approx 300$) where geometric closeness reflects semantic similarity.

```
                           Word Embedding Semantic Vector Geometry
                                  Vector("King") - Vector("Man") + Vector("Woman")
                                                     ≈ Vector("Queen")
```

### 1. Word2Vec (Mikolov et al., 2013)
- **Continuous Bag-of-Words (CBOW)**: Predicts target word $w_t$ given context words $w_{t-c} \dots w_{t+c}$.
- **Skip-Gram**: Predicts context words given target word $w_t$.
- **Negative Sampling Loss**:
  $$\mathcal{L}_{\text{SkipGram}} = \log \sigma(v_{w_t}^T v'_{w_c}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-v_{w_t}^T v'_{w_i}) \right]$$

### 2. GloVe (Global Vectors) & FastText
- **GloVe**: Leverages global log-bilinear matrix factorization on corpus co-occurrence counts.
- **FastText**: Models words as character $n$-grams (e.g. "where" $\rightarrow$ `<wh`, `whe`, `her`, `ere`, `re>`), enabling out-of-vocabulary (OOV) subword embedding generation.

---

## 17.2 The Transformer Paradigm Shift

Introduced by Vaswani et al. (2017) in *"Attention Is All You Need"*, Transformers replace sequential RNNs with parallelizable **Multi-Head Self-Attention**.

```
                               Transformer Encoder-Decoder
                     Encoder Block                               Decoder Block
             ┌─────────────────────────┐                 ┌─────────────────────────┐
             │ Multi-Head Self-Attention│                 │ Masked Self-Attention   │
             └────────────┬────────────┘                 └────────────┬────────────┘
                          │                                           │
             ┌────────────▼────────────┐                 ┌────────────▼────────────┐
             │   Feed-Forward Network  │                 │  Cross-Attention        │
             └─────────────────────────┘                 └─────────────────────────┘
```

### Scaled Dot-Product Attention Equation:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

- Scaling by $\frac{1}{\sqrt{d_k}}$ prevents dot-product magnitudes from growing large in high dimensions, preserving non-zero Softmax gradients.

### Rotary Position Embeddings (RoPE):
Modern LLMs (Llama-3, Mistral) apply **RoPE** by rotating Query and Key vectors in complex space using 2D rotation matrices:

$$R_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$

RoPE natively preserves relative token distance relationships $\text{RoPE}(x, m)^T \text{RoPE}(y, n) = g(x, y, m-n)$.

---

## 17.3 Large Language Models (LLMs) & PEFT (LoRA/QLoRA)

### 1. Decoder-Only Architecture (Causal Language Modeling)
Models the probability of next token $w_t$ conditioned strictly on preceding tokens $w_{<t}$:

$$P(W) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

### 2. Low-Rank Adaptation (LoRA)
Fine-tuning billion-parameter LLMs by updating all weights $W_0 \in \mathbb{R}^{d \times k}$ directly requires prohibitive GPU VRAM. **LoRA** freezes base weights $W_0$ and injects low-rank matrix decomposition:

$$W = W_0 + \Delta W = W_0 + B \cdot A$$

where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$ (e.g. $r=8$). Reduces trainable parameters by 99.9%!

```
                                     LoRA Architecture
                                    Input Vector x (d)
                                      ╱            ╲
                        Base Weight  ╱              ╲  Low-Rank A (d x r)
                        W0 (Frozen) ╱                ╲ (Trainable)
                                   │                  │  Low-Rank B (r x k)
                                   │                  │ (Trainable)
                                   ▼                  ▼
                                  h_base   +    h_adapter
                                            │
                                            ▼
                                     Output Vector y (k)
```

---

## 17.4 Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) connects LLMs to enterprise vector databases (Milvus, Qdrant, FAISS) to eliminate hallucinations and supply up-to-date private domain knowledge.

```
                           RAG Enterprise Query Flow
User Prompt ──► Embedding Model ──► Vector Index Search ──► Relevant Chunks ──► Augmented Prompt ──► LLM Output
```

### RAG Pipeline Mechanics:
1. **Document Chunking & Embedding**: Split text files into overlapping 512-token chunks and compute dense embedding vectors $v_{\text{doc}} \in \mathbb{R}^{1536}$.
2. **Vector Similarity Search**: Compute Cosine Similarity between user prompt vector $v_q$ and document vectors:
   $$\text{CosineSimilarity}(v_q, v_d) = \frac{v_q \cdot v_d}{\|v_q\| \|v_d\|}$$
3. **Prompt Augmentation & Generation**: Concatenate top-$K$ retrieved context chunks into LLM prompt system context.

---

## ⚓ Repository Code Reference
- See [`src/sentiment_analysis/pipeline.py`](file:///e:/Downloads/ML_only/src/sentiment_analysis/pipeline.py) for TF-IDF text feature processing.
- See [`src/document_classification/pipeline.py`](file:///e:/Downloads/ML_only/src/document_classification/pipeline.py) for multi-class NLP text classification pipelines.
