# 💻 Chapter 39: AI Compiler Stack & Hardware Acceleration (CUDA, Triton, FlashAttention)

## 39.1 The GPU Memory Hierarchy & Bottlenecks
Modern deep learning workloads are rarely constrained by GPU FLOP compute capacity; they are constrained by **Memory Bandwidth** (reading/writing tensors between High-Bandwidth Memory HBM and fast Streaming Multiprocessor SRAM registers).

```
                            GPU Memory & Latency Hierarchy
        HBM (High-Bandwidth Memory ~ 80 GB)   ── Latency: 500 cycles   (Slow / Large)
            │
            ▼
        SRAM / Shared Memory (SRAM ~ 20 MB)    ── Latency: 20 cycles    (Ultra Fast / Small)
            │
            ▼
        Registers (SM Execution Units)         ── Latency: 1 cycle      (Instantaneous)
```

---

## 39.2 Arithmetic Intensity & Memory-Bound Kernels

$$\text{Arithmetic Intensity} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Access (Bytes Read/Written)}}$$

- **Compute-Bound Kernels**: High Arithmetic Intensity (e.g. Large Matrix Multiplications $\mathbf{C} = \mathbf{A} \cdot \mathbf{B}$ where FLOPs $O(N^3)$ scale faster than memory $O(N^2)$).
- **Memory-Bound Kernels**: Low Arithmetic Intensity (e.g. Softmax, LayerNorm, ReLU where memory bandwidth stalls execution).

---

## 39.3 FlashAttention (Dao et al. 2022)

Standard Self-Attention computes intermediate $N \times N$ attention matrix $S = Q K^T$ and Softmax $P = \text{Softmax}(S)$, writing $O(N^2)$ memory to slow HBM.

```
Standard Attention:  Q, K ──► HBM ──► Compute Q K^T ──► Write N x N Matrix to HBM ──► Read N x N from HBM ──► Softmax ──► Write HBM
FlashAttention:      Q, K ──► Tiling Blocks into Fast SRAM ──► Online Softmax Scaling in SRAM ──► Output O to HBM (IO-Aware)
```

### Tiling & Online Softmax:
FlashAttention divides $Q, K, V$ into small $B_r \times B_c$ block tiles that fit inside fast **SRAM**, computing attention incrementally via **Online Softmax** without ever materializing the large $N \times N$ matrix in HBM. Result: 2x-4x speedup and memory scaling reduced from $O(N^2)$ to linear $O(N)$!

---

## 39.4 OpenAI Triton & Kernel Compilers

Triton (Tillet et al., OpenAI 2019) enables developers to write custom C++/CUDA-grade GPU kernels directly in Python using block-level parallel abstractions compiled down to PTX / LLVM.

---

## ⚓ Repository Code Reference
- See [`api/main.py`](file:///e:/Downloads/ML_only/api/main.py) for microservice concurrency configuration.
