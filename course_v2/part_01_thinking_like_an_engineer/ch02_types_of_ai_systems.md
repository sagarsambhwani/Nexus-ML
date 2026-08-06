# Chapter 02: Types of AI Systems

---

## 1. Big Picture

Artificial Intelligence encompasses a spectrum of techniques developed over 70 years. As an ML Engineer, selecting the right paradigm for a given business problem is critical. Using a 70-billion parameter Large Language Model (LLM) to perform deterministic arithmetic is an engineering failure; similarly, using hardcoded rule engines to parse unstructured audio streams is impossible.

This chapter categorizes the 5 dominant paradigms of AI:
1. **Symbolic AI (Rule-Based Systems)**
2. **Classical Machine Learning**
3. **Deep Learning**
4. **Reinforcement Learning (RL)**
5. **Generative AI & Foundation Models**

---

## 2. Intuition

| Paradigm | How it Works | Analogy | When to Use |
| :--- | :--- | :--- | :--- |
| **Symbolic AI** | If-Then rules written by human domain experts | Tax calculation software | High compliance, fixed rules, 100% determinism |
| **Classical ML** | Statistical pattern extraction on tabular data | Spam filter | Tabular databases, medium complexity |
| **Deep Learning** | Hierarchical representation learning via networks | Human visual cortex | Raw perception (Images, Audio, Text) |
| **Reinforcement Learning** | Trial-and-error optimization via rewards | Training a dog with treats | Autonomous driving, robotics, chess |
| **Generative AI** | Probabilistic sampling over broad world models | Creative writing assistant | Synthesis, reasoning, summarization |

---

## 3. Visualization

```text
                                  ┌─────────────────────────────────────────┐
                                  │      ARTIFICIAL INTELLIGENCE            │
                                  │  (Symbolic AI, Expert Systems, Rules)   │
                                  │  ┌───────────────────────────────────┐  │
                                  │  │       MACHINE LEARNING            │  │
                                  │  │   (Linear, Trees, Clustering)     │  │
                                  │  │   ┌───────────────────────────┐   │  │
                                  │  │   │       DEEP LEARNING       │   │  │
                                  │  │   │     (CNNs, RNNs, ViTs)    │   │  │
                                  │  │   │   ┌───────────────────┐   │   │  │
                                  │  │   │   │  GENERATIVE AI    │   │   │  │
                                  │  │   │   │ (LLMs, Diffusion) │   │   │  │
                                  │  │   │   └───────────────────┘   │   │  │
                                  │  │   └───────────────────────────┘   │  │
                                  │  └───────────────────────────────────┘  │
                                  └─────────────────────────────────────────┘
```

---

## 4. Mathematics

Mathematically, the paradigms differ by the formulation of their objective functions and action spaces:

1. **Symbolic Logic**:
$$y = \bigwedge_{i=1}^k (x_i \odot c_i) \quad \text{where } \odot \in \{\leq, \geq, =\}$$

2. **Classical Supervised Learning**:
$$f^*(x) = \arg\min_{f \in \mathcal{H}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

3. **Reinforcement Learning (Markov Decision Process)**:
$$\pi^*(a|s) = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t R(s_t, a_t) \right]$$

4. **Generative Modeling (Autoregressive Token Distribution)**:
$$P_\theta(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

---

## 5. Python (From Scratch)

Comparing Symbolic AI vs Classical Machine Learning in pure Python for credit risk scoring:

```python
import numpy as np

# Sample Applicant: [Income ($k), Credit Score, Debt-to-Income Ratio]
applicant = np.array([75.0, 680, 0.35])

# 1. Symbolic AI Paradigm (Hardcoded Rules)
def symbolic_credit_decision(applicant):
    income, credit_score, dti = applicant
    if credit_score > 700 and dti < 0.40:
        return "APPROVE"
    elif credit_score > 650 and income > 70:
        return "APPROVE"
    else:
        return "REJECT"

# 2. Classical Machine Learning Paradigm (Logistic Decision Boundary)
weights = np.array([0.04, 0.015, -3.2])
bias = -10.5

def ml_credit_decision(applicant, weights, bias):
    z = np.dot(applicant, weights) + bias
    prob = 1.0 / (1.0 + np.exp(-z))  # Sigmoid activation
    return "APPROVE" if prob >= 0.50 else "REJECT", prob

print("Symbolic Decision:", symbolic_credit_decision(applicant))
status, prob = ml_credit_decision(applicant, weights, bias)
print(f"ML Decision: {status} (Approval Probability: {prob:.4f})")
```

---

## 6. Production Library (Scikit-Learn / PyTorch)

Implementing a multi-class comparison pipeline across paradigms:

```python
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

# Classical ML Paradigm: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Deep Learning Paradigm: PyTorch Multi-Layer Perceptron (MLP)
class NeuralCreditClassifier(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

dl_model = NeuralCreditClassifier(input_dim=3)
sample_tensor = torch.tensor([[75.0, 680.0, 0.35]], dtype=torch.float32)
print("PyTorch Output Probability:", dl_model(sample_tensor).item())
```

---

## 7. Under the Hood

- **Symbolic AI**: Executes AST (Abstract Syntax Tree) branch lookups. CPU instruction pointer branches continuously (`JMP` instructions), causing frequent CPU branch mispredictions.
- **Deep Learning**: Replaces branch instructions with continuous matrix multiplication kernels (`GEMM`), executing efficiently on massively parallel GPU CUDA tensor cores.

---

## 8. Engineering Perspective

- **Compute Latency**: Symbolic AI (sub-millisecond) < Classical ML (1–5ms) < Deep Learning (10–50ms) < LLM Inference (200–2000ms).
- **Maintenance Cost**: Symbolic AI maintenance grows exponentially with rule count (n-squared rule interactions). ML models automate feature combination discovery at the cost of requiring data monitoring pipelines.

---

## 9. Common Mistakes

1. **Over-engineering simple tasks**: Deploying an LLM for regex-based string extraction.
2. **Under-estimating tabular ML**: Attempting to train complex PyTorch Transformers on small tabular datasets when XGBoost/LightGBM easily outperforms them with 1/100th compute.

---

## 10. Interview Questions

### Q1: When should you prefer Gradient Boosted Decision Trees over Deep Neural Networks?
**Answer**: GBDTs (XGBoost/LightGBM) outperform Neural Networks on heterogeneous tabular data with non-linear numerical/categorical boundaries, missing values, and limited sample sizes ($N < 100,000$). Neural Networks excel on un-structured grid data (images, audio, continuous text embeddings).

---

## 11. Exercises

1. **Coding**: Implement a simple Multi-Armed Bandit algorithm (Reinforcement Learning) in 20 lines of Python using $\epsilon$-greedy exploration.
2. **Architecture**: Draw a flowchart deciding whether a project requires: Rule Engine, XGBoost, CNN, or LLM + RAG.

---

## 12. Mini Project: Paradigm Comparison Benchmark

Benchmark memory usage and inference latency for a simple dataset across (1) Python Nested If-Else rules, (2) Scikit-Learn DecisionTree, and (3) PyTorch MLP.

---

## 13. Capstone Integration

Connects to `src/fraud_detection/pipeline.py` which combines tabular XGBoost models with Graph Neural Network (GNN) embeddings for multi-paradigm fraud detection.
