# ⚡ Chapter 25: Efficient Model Compression — Distillation, Quantization & Pruning

## 25.1 The Edge Deployment Challenge
State-of-the-art deep learning models and Large Language Models require billions of parameters and gigabytes of memory. To deploy models on resource-constrained edge devices (mobile phones, embedded IoT, sub-10ms microservices), we must compress models without sacrificing predictive accuracy.

```
                           Model Compression Pipeline
                                 Teacher Model (FP32)
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
Knowledge Distillation            INT8 / INT4 Quantization          Sparse Weight Pruning
(Teacher ──► Student)           (FP32 ──► Low-Precision)        (Zero out unimportant weights)
```

---

## 25.2 Knowledge Distillation (Teacher-Student Networks)

Pioneered by Geoffrey Hinton et al. (2015), Knowledge Distillation transfers knowledge from a large, highly accurate **Teacher Model** $T$ to a compact, fast **Student Model** $S$.

```
Input x ──┬──► Teacher Model (Large) ──► Soft Logits z_T / Temp T ──┐
          │                                                         ├──► KL-Divergence Loss
          └──► Student Model (Small) ──► Soft Logits z_S / Temp T ──┘
```

### Temperature Softmax Function:
Standard Softmax outputs hard probabilities. Introducing temperature $T > 1$ softens probability distributions, revealing **dark knowledge** (inter-class similarity signals):

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### Distillation Loss Function:
$$\mathcal{L}_{\text{total}} = (1 - \alpha) \underbrace{\mathcal{L}_{\text{CE}}(y, \sigma(z_S))}_{\text{Hard Target Cross-Entropy}} + \alpha T^2 \underbrace{\text{KL}\left( \sigma\left(\frac{z_S}{T}\right) \;\parallel\; \sigma\left(\frac{z_T}{T}\right) \right)}_{\text{Soft Knowledge Distillation}}$$

---

## 25.3 Advanced Model Quantization (AWQ & GPTQ)

Quantization maps 32-bit floating point weights ($W \in \mathbb{R}^{32}$) into low-bit representations ($W_{\text{quant}} \in \mathbb{Z}^4$ or $\mathbb{Z}^8$).

### 1. Activation-Aware Weight Quantization (AWQ)
Protects the top 1% of salient weight channels that correspond to large activation magnitudes, quantizing non-salient channels aggressively to 4-bit precision with near-zero perplexity loss.

### 2. GPTQ (One-Shot Second-Order Quantization)
Quantizes weight matrices row-by-row using inverse Hessian matrix updates $(H^{-1})_{jj}$ to compensate for quantization errors in remaining unquantized weights:

$$w_q^* = \arg\min_{w_q} (w - w_q)^T H (w - w_q)$$

---

## 25.4 Sparse Weight Pruning

Weight Pruning zeroes out non-essential connections ($w_j \to 0$) in weight matrices.

### 1. Magnitude Pruning
Zeroes out weights whose absolute magnitude is below a set threshold: $|w_j| < \tau$.
### 2. Structured vs. Unstructured Pruning
- **Unstructured Pruning**: Zeroes individual weights anywhere in matrices (requires specialized sparse hardware accelerators).
- **Structured Pruning**: Removes entire attention heads, channels, or matrix rows, yielding immediate CPU/GPU speedups on standard runtimes.

---

## ⚖️ Compression Techniques Benchmarking Summary

| Technique | Memory Footprint Reduction | Speedup Factor | Implementation Complexity |
|---|---|---|---|
| **Knowledge Distillation** | 2x – 10x | 2x – 10x | High (Requires Retraining) |
| **INT8 Quantization** | 4x | 2x – 4x | **Low (Post-Training)** |
| **INT4 (AWQ / GPTQ)** | **8x** | 4x – 8x | Medium |
| **Structured Pruning** | 2x – 4x | 2x – 3x | Medium |

---

## ⚓ Repository Code Reference
- See [`api/main.py`](file:///e:/Downloads/ML_only/api/main.py) for high-performance memory-mapped inference serving.
- See [`Dockerfile`](file:///e:/Downloads/ML_only/Dockerfile) for minimal multi-stage deployment builds.
