# 📚 Chapter 65: Parameter-Efficient Fine-Tuning (LoRA & QLoRA)

## 65.1 Theoretical Overview
LoRA freezes base weight W_0 and injects low-rank matrix decomposition W = W_0 + B * A (r << min(d,k)), reducing trainable parameters by 99.9%.

---

## 65.2 Key Mathematical Derivations & Code Principles
Detailed implementation principles and mathematical formulations governing **Parameter-Efficient Fine-Tuning (LoRA & QLoRA)**.

---

## ⚓ Repository Code Reference
- See [`src/document_classification/pipeline.py`](file:///e:/Downloads/ML_only/src/document_classification/pipeline.py) for real-world pipeline implementation.
