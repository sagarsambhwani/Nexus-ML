# 🧬 Chapter 38: Bio-ML & Genomic / Protein Foundation Models (AlphaFold)

## 38.1 Machine Learning in Computational Biology
Biological systems store digital information in 1D genomic sequences (DNA/RNA nucleotides $\{A, C, G, T\}$) which transcribe into 20-amino acid protein chains that fold into complex 3D molecular structures determining biological function.

```
1D Amino Acid Sequence ("MKWVTF...") ──► Protein Language Model (ESM-2 / AlphaFold 2) ──► 3D Folded Protein Structure Coordinates (x,y,z)
```

---

## 38.2 AlphaFold 2 Architecture & Evoformer

Jumper et al. (DeepMind, 2021) solved the 50-year-old protein folding problem using **AlphaFold 2**.

```
Input Sequence ──► Multiple Sequence Alignment (MSA) + Pair Representation ──► Evoformer Stack (48 Layers) ──► Structure Module (IPA) ──► 3D Backbone (x,y,z)
```

### Key AlphaFold Innovations:
1. **Evoformer Block**: Alternates attention operations across both MSA rows (evolutionary relationships) and 2D pair residue distance matrices.
2. **Invariant Point Attention (IPA)**: An $SE(3)$-equivariant 3D attention mechanism that operates directly on 3D Euclidean spatial rotations $R_i \in SO(3)$ and translations $t_i \in \mathbb{R}^3$.

---

## 38.3 Protein Language Models: ESM-2

ESM-2 (Lin et al. Meta AI, 2023) trains BERT-style Masked Language Models on hundreds of millions of raw amino acid sequences without requiring explicit MSA alignment, enabling zero-shot mutation effect predictions and structural folding (ESMFold).

---

## ⚓ Repository Code Reference
- See [`src/medical_diagnosis/pipeline.py`](file:///e:/Downloads/ML_only/src/medical_diagnosis/pipeline.py) for biomarker feature modeling.
