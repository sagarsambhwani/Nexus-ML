# Chapter 16: Deep Neural Networks & Transformer Architectures

---

## 1. Big Picture

Deep Learning revolutionized AI by learning hierarchical representations directly from raw perceptual data (pixels, audio waveforms, text tokens). Instead of hand-crafting features, deep networks learn layers of abstractions: low-level edges $\to$ textures $\to$ object parts $\to$ semantic entities.

This chapter covers Feedforward Neural Networks (MLP), Backpropagation, Convolutional Neural Networks (CNNs), Recurrent Neural Networks (LSTM), and the foundation of modern AI: **Transformer Self-Attention**.

---

## 2. Intuition

- **Universal Approximation Theorem**: A neural network with a single hidden layer and non-linear activation functions can approximate any continuous function to arbitrary precision.
- **Scaled Dot-Product Self-Attention**: A dynamic lookup mechanism where every word token in a sequence asks 3 questions:
  1. **Query (Q)**: "What information am I looking for?"
  2. **Key (K)**: "What information do I contain?"
  3. **Value (V)**: "If I am relevant, what payload content do I deliver?"

---

## 3. Visualization

```text
Transformer Multi-Head Self-Attention Architecture:

                     Input Tokens X
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Queries (Q)   Keys (K)    Values (V)
             │             │             │
             └──────┬──────┘             │
                    ▼                    │
          [ Softmax( Q K^T / √d_k ) ]    │  (Attention Matrix Weights)
                    │                    │
                    └─────────┬──────────┘
                              ▼
                     Weighted Output Context Vector Z
```

---

## 4. Mathematics

### Scaled Dot-Product Attention Equation
Given Query matrix $Q \in \mathbb{R}^{N \times d_k}$, Key matrix $K \in \mathbb{R}^{M \times d_k}$, and Value matrix $V \in \mathbb{R}^{M \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

Where $\sqrt{d_k}$ scaling factor prevents dot-product values from growing extremely large in high dimensions, which would cause Softmax gradients to saturate near zero (vanishing gradients).

---

## 5. Python (From Scratch Scaled Dot-Product Attention)

```python
import numpy as np

def scaled_dot_product_attention_scratch(Q, K, V):
    """Computes Scaled Dot-Product Attention from scratch in NumPy."""
    d_k = Q.shape[-1]
    
    # 1. Compute Raw Attention Scores (Q @ K^T)
    scores = Q @ K.T / np.sqrt(d_k)
    
    # 2. Apply Softmax along trailing axis (Rows sum to 1.0)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True)) # Numerically stable Softmax
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 3. Compute Context Output (Weights @ V)
    output = attn_weights @ V
    return output, attn_weights

# Test Execution with 3 Sequence Tokens of dimension d_k = 4
Q = np.random.randn(3, 4)
K = np.random.randn(3, 4)
V = np.random.randn(3, 4)

out, weights = scaled_dot_product_attention_scratch(Q, K, V)
print("Attention Weights Matrix (3x3 Token Interactions):")
print(np.round(weights, 4))
```

---

## 6. Production Library (PyTorch MultiheadAttention)

```python
import torch
import torch.nn as nn

# Production Transformer Self-Attention Layer
attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(16, 50, 512)  # Batch=16, SeqLen=50, Dim=512

attn_output, attn_weights = attn(x, x, x)
print("PyTorch Attention Output Shape:", attn_output.shape)
```

---

## 7. Under the Hood

- **FlashAttention** (Dao et al.): Standard attention materializes $N \times N$ attention matrices in GPU High Bandwidth Memory (HBM). FlashAttention fuses Softmax and GEMM operations inside GPU SRAM using tiling, reducing memory complexity from $O(N^2)$ to $O(N)$ and accelerating training 3x-5x!

---

## 8. Engineering Perspective

- **Computational Bottleneck**: Self-Attention time/memory complexity scales quadratically $O(N^2)$ with sequence length $N$. For long context windows ($N > 32,000$), modern architectures use FlashAttention-2, Linear Attention, or State-Space Models (Mamba).

---

## 9. Common Mistakes

1. **Omitting Positional Encoding in Transformers**: Self-attention is permutation-invariant! Without positional embeddings (RoPE, Sine/Cosine), a Transformer cannot distinguish "Dog bites Man" from "Man bites Dog".
2. **Missing Normalization Layers**: Omitting LayerNorm or RMSNorm between Transformer blocks leading to gradient instability.

---

## 10. Interview Questions

### Q1: Why do we divide by $\sqrt{d_k}$ in Transformer attention?
**Answer**: If components of $Q$ and $K$ are independent random variables with mean 0 and variance 1, their dot product $Q \cdot K = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$. For large $d_k$, dot products become very large, pushing Softmax into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ pulls variance back to 1.0, preserving healthy gradients.

---

## 11. Exercises

1. **Math**: Derive the backward pass gradients of Scaled Dot-Product Attention with respect to Query $Q$.
2. **Coding**: Implement Multi-Head Attention from scratch in PyTorch without using `nn.MultiheadAttention`.

---

## 12. Mini Project: Mini-GPT Decoder Architecture

Write a PyTorch script `mini_gpt.py` that implements a 4-layer Causal Transformer Language Model with RoPE positional embeddings and trains it on text.

---

## 13. Capstone Integration

Underpins foundation models in `src/sentiment_analysis/pipeline.py` and `src/document_classification/pipeline.py`.
