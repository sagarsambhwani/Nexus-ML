# 💬 Phase 09 — Vision + Language (Multimodal AI & VLMs)

## 🎯 Phase Overview & Goal
- **Duration**: ~3 weeks
- **Objective**: Explore the intersection of **Computer Vision and Large Language Models (LLMs)**. Master **CLIP (Contrastive Language-Image Pretraining)**, zero-shot classification, cross-attention alignment, and the modern **Vision-Language Model (VLM)** architectures powering systems like **LLaVA, GPT-4V, and Qwen-VL**.

---

## 1. OpenAI CLIP: Connecting Text & Pixels (Radford et al., 2021)

CLIP trained on 400 million $(Image, Text)$ pairs collected from the internet using **Symmetric Contrastive Learning**:

```text
 ┌──────────────────────┐                           ┌──────────────────────┐
 │   Input Images [N]   │                           │    Input Texts [N]   │
 └──────────┬───────────┘                           └──────────┬───────────┘
            ▼                                                  ▼
   Image Encoder (ViT)                                 Text Encoder (Transformer)
            │                                                  │
            ▼ Normalized Image Embeddings I_i                  ▼ Normalized Text Embeddings T_j
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   Cosine Similarity Matrix (N x N)                      │
 │                   S_ij = (I_i · T_j) / τ                                │
 │                                                                         │
 │       Text 1       Text 2       Text 3       ...       Text N           │
 │ Image 1 [ MATCH ]  [   -   ]  [   -   ]      ...     [   -   ]          │
 │ Image 2 [   -   ]  [ MATCH ]  [   -   ]      ...     [   -   ]          │
 │ Image 3 [   -   ]  [   -   ]  [ MATCH ]      ...     [   -   ]          │
 └─────────────────────────────────────────────────────────────────────────┘
            │
            ▼
 Symmetric Cross-Entropy Loss (Row-wise & Column-wise Softmax over Diagonal)
```

### The Symmetric Contrastive Loss Formulation:
For a batch of $N$ image-text pairs:
$$\mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathbf{I}_i^\top \mathbf{T}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{I}_i^\top \mathbf{T}_j / \tau)}$$
$$\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{j=1}^N \log \frac{\exp(\mathbf{I}_j^\top \mathbf{T}_j / \tau)}{\sum_{i=1}^N \exp(\mathbf{I}_i^\top \mathbf{T}_j / \tau)}$$
$$\mathcal{L}_{\text{total}} = \frac{\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}}{2}$$

---

## 2. Zero-Shot Image Classification with CLIP

CLIP requires **zero task-specific training or fine-tuning** to classify arbitrary visual classes!

```text
 Class Names: ["dog", "cat", "airplane"]
        │
        ▼ Prompt Engineering
 Formatted Text Prompts:
 1. "a photo of a dog"      ──► Text Encoder ──► T1
 2. "a photo of a cat"      ──► Text Encoder ──► T2
 3. "a photo of a airplane" ──► Text Encoder ──► T3
                                                  │
 Query Image ─────────────────► Image Encoder ──► I ──► Cosine Similarity (I · Tk) ──► Softmax ──► Prediction
```

- **Prompt Ensembling**: Averaging text embeddings across multiple prompt templates (`"a photo of a {c}"`, `"a cropped image of a {c}"`, `"a clean centered picture of a {c}"`) boosts zero-shot accuracy by $3-5\%$.

---

## 3. Modern Vision-Language Models (VLMs / LLaVA Architecture)

How do modern multimodal chatbots "see" and converse about images?

```text
 ┌──────────────────────┐
 │  Input Image (Patch) │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Pre-trained Vision   │ (e.g., CLIP-ViT-L/14)
 │    Encoder (Frozen)  │
 └──────────┬───────────┘
            ▼ Feature Grid Z_v (e.g., 576 tokens x 1024 dim)
 ┌──────────────────────┐
 │ Modality Projection  │ (Linear Projection or 2-layer MLP)
 │     W_v (Trainable)  │
 └──────────┬───────────┘
            ▼ Visual Tokens H_v (576 tokens x 4096 dim)
 ┌──────────────────────────────────────────────────────────────┐
 │ User Prompt: "Describe the defect on the left pipe."         │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼ Tokenized Text H_q
 ┌──────────────────────────────────────────────────────────────┐
 │ Interleaved Input Sequence: [Visual Tokens H_v, Text H_q]    │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Autoregressive Large Language Model (Llama-3 / Vicuna / Mistral)
 │ Output: "There is a severe corrosion stain near the valve..." │
 └──────────────────────────────────────────────────────────────┘
```

### Key Components of LLaVA (Liu et al., 2023):
1. **Visual Tokenization**: The pre-trained CLIP vision encoder converts the 2D image into a sequence of $576$ visual token embeddings.
2. **Projector ($W_v$)**: A lightweight 2-layer MLP projects the visual embedding space directly into the LLM's text word-embedding dimension.
3. **Autoregressive Generation**: The LLM treats visual tokens exactly like prefix text tokens and generates descriptive responses autoregressively.
