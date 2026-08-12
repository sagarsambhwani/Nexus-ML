# 🖼️ Chapter 21: Multi-Modal Learning & Vision-Language Models (VLMs)

## 21.1 Introduction to Multi-Modal AI
Multi-modal machine learning processes and correlates information from multiple distinct modalities (text, vision, audio, tabular data, sensor telemetry) into shared representation spaces.

```
Text Prompt "A red sports car" ──► Text Encoder ──┐
                                                 ├──► Shared Joint Embedding Space (Cosine Similarity)
Image Pixel Input ────────────► Vision Encoder ─┘
```

---

## 21.2 CLIP (Contrastive Language-Image Pre-training)

Introduced by Radford et al. (OpenAI, 2021), CLIP trains an Image Encoder $I(x)$ and Text Encoder $T(y)$ jointly on 400M image-text pairs using **Symmetric Contrastive Loss**.

```
                           CLIP Contrastive Matrix (N x N)
                         Text Embeddings T_1 ... T_N
                     ┌──────────────────────────────────┐
        Image        │  (I_1, T_1)*   (I_1, T_2)  ...   │  * Positive Pairs (Diagonal)
      Embeddings     │  (I_2, T_1)   (I_2, T_2)*  ...   │
      I_1 ... I_N    │    ...           ...       ...   │
                     └──────────────────────────────────┘
```

### Contrastive Loss Math:
Given a batch of $N$ (image, text) pairs $(I_i, T_i)$, compute normalized embeddings $v_i = \frac{I_i}{\|I_i\|_2}$ and $u_i = \frac{T_i}{\|T_i\|_2}$. Similarity logits: $S_{i,j} = v_i^T u_j \cdot e^{\tau}$.

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} \left( \mathcal{L}_{I \to T} + \mathcal{L}_{T \to I} \right)$$

$$\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{i, i})}{\sum_{j=1}^N \exp(S_{i, j})}, \quad \mathcal{L}_{T \to I} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{i, i})}{\sum_{j=1}^N \exp(S_{j, i})}$$

---

## 21.3 Vision-Language Models (VLMs: LLaVA & Qwen-VL)

Modern Vision-Language Models connect a pre-trained Vision Encoder (like ViT / CLIP-ViT) to a Causal Large Language Model via a Projection Connector.

```
Image ──► Vision Encoder (ViT) ──► Linear Projection / Cross-Attention ──► Visual Tokens ──┐
                                                                                           ├──► Causal LLM ──► Text Response
Text Prompt ────────────────────────────────────────────────────────────► Text Tokens   ──┘
```

### Architecture of LLaVA (Large Language and Vision Assistant):
1. **Vision Encoder**: CLIP ViT-L/14 extracts visual features $Z_v \in \mathbb{R}^{H_v \times W_v \times d_v}$.
2. **Projection Matrix $W_v$**: Linearly projects visual features into text embedding space:
   $$H_v = W_v Z_v \in \mathbb{R}^{N_v \times d_{\text{text}}}$$
3. **Causal LLM (Llama-3 / Vicuna)**: Concatenates visual tokens $H_v$ and text prompt tokens $H_q$ to generate text autoregressively.

---

## ⚓ Repository Code Reference
- See [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py) for computer vision feature extraction principles.
- See [`src/document_classification/pipeline.py`](file:///e:/Downloads/ML_only/src/document_classification/pipeline.py) for text feature classification.
