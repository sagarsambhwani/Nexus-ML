# Chapter 23: Large Language Model (LLM) Engineering: Internals, LoRA, QLoRA, and Agents

---

## 1. Big Picture

Large Language Models (LLMs) like GPT-4, Llama-3, and Mistral represent a milestone in artificial intelligence. Engineering with LLMs requires understanding tokenization, Rotary Position Embeddings (RoPE), KV-Caching, Quantization (GGUF, AWQ, GPTQ), Parameter-Efficient Fine-Tuning (**LoRA / QLoRA**), and **Autonomous Agentic ReAct Loops**.

This chapter covers LLM internal mechanics from scratch, memory requirements, fine-tuning, and building production AI Agents with function-calling capabilities.

---

## 2. Intuition

- **KV-Caching**: During autoregressive text generation, calculating key/value projections for all previous tokens in every single step wastes GPU FLOPS ($O(N^2)$ repeated math). KV-Caching saves historical Key and Value tensors in GPU memory, reducing per-token generation complexity to $O(N)$!
- **LoRA (Low-Rank Adaptation)**: Freezing the 70-billion parameters of a base LLM ($W_0$) and attaching tiny trainable adapter rank matrices $A$ and $B$ ($W = W_0 + B \cdot A$). Reduces trainable parameter counts by 99.9% while retaining full fine-tuning performance!

---

## 3. Visualization

```text
Low-Rank Adaptation (LoRA) Memory Efficiency:

            Original Frozen Base Weight W_0  (4096 x 4096 = 16.7M Parameters - Frozen)
                          │
                          ├─────────────────────────────────┐
                          ▼                                 ▼
                     Matrix B (4096 x 8)             Matrix A (8 x 4096)
                     (32k Trainable)                 (32k Trainable)
                          │                                 │
                          └────────────────┬────────────────┘
                                           ▼
                            Output ΔW = B @ A (Rank r=8)
```

---

## 4. Mathematics

### 1. LoRA Matrix Decomposition
For weight matrix $W_0 \in \mathbb{R}^{d \times k}$, the modified forward pass is:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$

Where:
- $A \sim \mathcal{N}(0, \sigma^2)$ initialized randomly, $B = 0$ initialized to zero (so $\Delta W = 0$ at step 0).
- $r \ll \min(d, k)$ is rank (typically $r \in \{8, 16, 64\}$).
- $\frac{\alpha}{r}$ is a scaling hyper-parameter.

### 2. GPU Memory Estimation Formula for LLM Inference
Memory required to load an $N$-billion parameter model with KV-Cache:

$$\text{RAM}_{\text{Bytes}} \approx N \times \text{BytesPerParam} + \left( 2 \cdot L \cdot H \cdot D \cdot B \cdot S \cdot \text{BytesPerParam} \right)$$

Where $L$ is layers, $H$ is heads, $D$ is dimension, $B$ is batch size, and $S$ is sequence length.

---

## 5. Python (From Scratch LoRA Layer Implementation)

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        # 1. Base Frozen Pretrained Weight
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False  # Freeze Base Model!
        
        # 2. Low-Rank Adapter Matrices
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * (1.0 / np.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r)) # Zero Init
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Frozen Pass
        base_out = self.base_layer(x)
        # Low-Rank Adapter Pass: (x @ A^T @ B^T) * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

# Test LoRA Layer
layer = LoRALinear(in_features=4096, out_features=4096, r=8)
x_sample = torch.randn(2, 4096)
out = layer(x_sample)

trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in layer.parameters() if not p.requires_grad)

print(f"Base Frozen Parameters:      {frozen_params:,}")
print(f"LoRA Trainable Parameters:   {trainable_params:,}")
print(f"Parameter Reduction Factor:  {frozen_params / trainable_params:.1f}x smaller!")
```

---

## 6. Production Library (HuggingFace PEFT & BitsAndBytes QLoRA)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# 1. 4-bit Quantization Config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Load Base 4-bit Model
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", quantization_config=bnb_config)

# 3. Attach LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# peft_model = get_peft_model(model, peft_config)
```

---

## 7. Under the Hood

- **NF4 (NormalFloat 4)**: QLoRA uses an information-theoretically optimal quantile quantization data type for normally distributed neural network weights, compressing 16-bit FP16 weights to 4-bit representation with near-zero accuracy loss.

---

## 8. Engineering Perspective

- **Agentic ReAct Loop**: Autonomous AI Agents operate via a Reasoning + Action loop:
  1. *Thought*: Reason about current goal.
  2. *Action*: Output structured JSON tool call (e.g. `{"tool": "python_interpreter", "code": "..."}`).
  3. *Observation*: Receive tool response from API/Environment.
  4. *Repeat* until final answer is reached.

---

## 9. Common Mistakes

1. **Full Parameter Fine-Tuning Large Models**: Attempting to fine-tune a 70B model end-to-end without LoRA/QLoRA, requiring 600+ GB of GPU VRAM.
2. **Infinite Agent Loops**: Deploying autonomous agents without setting strict step limits (`max_iterations=10`) or timeout limits, incurring runaway API billing costs.

---

## 10. Interview Questions

### Q1: Explain how QLoRA achieves 4-bit fine-tuning without losing model precision.
**Answer**: QLoRA combines 3 innovations: (1) **NF4 Quantization** which maps weights into optimal 4-bit bins, (2) **Double Quantization** which quantizes the quantization constants themselves, saving 0.37 bits/param, and (3) **Paged Optimizers** which leverage CUDA Unified Memory to prevent VRAM allocation spikes during gradient updates.

---

## 11. Exercises

1. **Coding**: Implement a pure Python Agentic ReAct loop parser that parses string output into tool calls.
2. **Math**: Calculate exact VRAM footprint required to run Llama-3 70B in FP16 vs INT8 vs INT4.

---

## 12. Mini Project: Production AI Agent Microservice

Write a Python script `agent_service.py` using FastAPI and PyTorch/Ollama that implements an autonomous agent equipped with SQL querying, calculator, and web search tools.

---

## 13. Capstone Integration

Implemented across `src/document_classification/pipeline.py` and `api/main.py`.
