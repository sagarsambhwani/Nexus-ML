# 🏛️ Phase 10 — Foundation Vision Models (DINOv2 & SAM)

## 🎯 Phase Overview & Goal
- **Duration**: ~3 weeks
- **Objective**: Study the breakthrough foundation models that established universal, task-agnostic visual perception: **Meta's DINOv2** (universal visual representations) and **Segment Anything Model (SAM)** (promptable zero-shot image segmentation).

---

## 1. What Makes a "Foundation Model" in Computer Vision?

Historically, visual models were trained for narrow single tasks:
$$\text{Dataset A} \to \text{ResNet (Classification)}, \quad \text{Dataset B} \to \text{YOLO (Detection)}, \quad \text{Dataset C} \to \text{U-Net (Segmentation)}$$

A **Vision Foundation Model** is a massive, self-supervised or promptable model trained on hundreds of millions of images that produces representations capable of solving **all downstream vision tasks with zero or minimal fine-tuning**:

```text
                               ┌────────────────────────────────────────────────────────┐
                               │             UNIVERSAL VISION FOUNDATION MODEL          │
                               │                   (DINOv2 / SAM / CLIP)                │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
        ┌──────────────────────┬───────────────────────────┼───────────────────────────┬──────────────────────┐
        ▼                      ▼                           ▼                           ▼                      ▼
 Zero-Shot Search      Dense Correspondence       Depth Estimation            Semantic Masks         Object Detection
```

---

## 2. Meta DINOv2: Universal Visual Features (Oquab et al., 2023)

DINOv2 scaled self-supervised Vision Transformers to **1 billion parameters** on **142 million uncurated images (LVD-142M)** without human annotations.

### Emergent Properties of DINOv2 Embeddings:
1. **Dense Semantic Matching**: Point features between completely different images of the same category (e.g. a husky vs. a golden retriever) have high cosine similarity on corresponding body parts (snout, paws, tail).
2. **Monocular Depth Estimation**: Features linearly encode physical depth without any depth sensor supervision.
3. **Zero-Shot Segmentation**: K-Means clustering on the output patch tokens reproduces semantic segmentation masks directly.

```text
 Image 1 (Poodle) ──► DINOv2 ──► Feature Map F1 ──┐
                                                  ├──► Cosine Similarity: Eye(Image1) <---> Eye(Image2) ~ 0.94!
 Image 2 (Tiger)  ──► DINOv2 ──► Feature Map F2 ──┘
```

---

## 3. Segment Anything Model (SAM - Kirillov et al., Meta 2023)

SAM formulated segmentation as a **Promptable Foundation Task**:

```text
 ┌──────────────────────┐
 │ Input Image (1024x)  │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Heavyweight Image Encoder (ViT-H, 632M params)               │  <-- Computed ONCE per image
 │ Output: 64x64 Image Embedding (256 channels)                 │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Lightweight Mask Decoder (<50ms execution on CPU/Web)        │
 │                                                              │
 │  Prompt Input:                                               │
 │  • Point Coordinates (Positive / Negative clicks)            │
 │  • Bounding Box Box Prompt                                   │
 │  • Rough Mask Prompt                                         │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
               3 High-Confidence Disambiguated Masks
               (Whole Object, Part, Sub-part)
```

### The Three Architectural Pillars of SAM:
1. **Heavy Image Encoder**: ViT-H processes the high-resolution $1024 \times 1024$ image once to generate a dense $64 \times 64 \times 256$ spatial embedding.
2. **Prompt Encoder**: Encodes points/boxes using positional encodings and masks via convolutional downsampling.
3. **Lightweight Two-Way Attention Decoder**: Cross-attends between prompt tokens and image embeddings to output multi-scale binary masks in **<50 milliseconds**.
4. **Data Engine**: Trained on the massive **SA-1B dataset** (11 million images, **1.1 billion masks**).
