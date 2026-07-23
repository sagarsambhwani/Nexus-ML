# 👁️ Chapter 19: Computer Vision — Detection, Segmentation & ViTs

## 19.1 Computer Vision Tasks Taxonomy
Computer vision encompasses multiple hierarchical tasks:

```
                            Computer Vision Taxonomy
                                       │
     ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
     ▼                   ▼                           ▼                   ▼
Classification   Object Detection           Semantic Segmentation  Instance Segmentation
(Image Label)   (Bounding Boxes + Class)    (Pixel-Level Class)    (Individual Object Pixels)
```

---

## 19.2 Object Detection Paradigms

```
                           Object Detection Paradigms
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
 Two-Stage Detectors (Faster R-CNN)                       Single-Stage Detectors (YOLOv8)
 • Region Proposal Network (RPN)                          • Direct single-pass dense grid prediction
 • High accuracy, slower latency                          • Ultra-fast real-time edge processing
```

### 1. Two-Stage Detection (Faster R-CNN)
- **Stage 1 (Region Proposal Network - RPN)**: Slides over CNN feature maps to generate candidate Bounding Box proposals (Anchors).
- **Stage 2 (RoI Align & Classification)**: Extracts features per Region of Interest (RoI) to predict final class labels and bounding box offsets $(\Delta x, \Delta y, \Delta w, \Delta h)$.

### 2. Intersection over Union (IoU) & Non-Maximum Suppression (NMS)
- **IoU Evaluation Metric**:
  $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|\mathcal{B}_{\text{pred}} \cap \mathcal{B}_{\text{gt}}|}{|\mathcal{B}_{\text{pred}} \cup \mathcal{B}_{\text{gt}}|}$$
- **Non-Maximum Suppression (NMS)**: Eliminates redundant overlapping candidate bounding boxes that target the same physical object.

### 3. Single-Stage Real-Time Detection (YOLOv8)
YOLO (You Only Look Once) divides the image into an $S \times S$ grid, predicting bounding boxes and class probabilities simultaneously in a single forward pass.

```python
# YOLOv8 Inference Code Example
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Predict objects on image frame
results = model("assembly_line_surface.jpg")
for r in results:
    boxes = r.boxes # Bounding box coordinates
    masks = r.masks # Segmentation masks
```

---

## 19.3 Image Segmentation & U-Net Architecture

Semantic segmentation assigns a category label to every individual pixel in an image.

```
                             U-Net Encoder-Decoder
  Input Image ──► Encoder (Downsampling / Convolutions) ──┐
                         │                                 │ Skip Connections
                         └───────────────────────────────► │
                                                           ▼
  Pixel Masks ◄── Decoder (Upsampling / Transposed Convs) ─┘
```

### U-Net Skip Connections:
The encoder contracts spatial resolution while building deep feature representations. The decoder expands spatial dimensions back to original resolution. **Skip connections** copy high-resolution feature maps directly from encoder layers to decoder layers, preserving precise boundary edges.

---

## 19.4 Vision Transformers (ViT)

Dosovitskiy et al. (2020) demonstrated that pure Transformer architectures applied to image patches achieve state-of-the-art results without spatial convolutions.

```
Image X (H x W x C) ──► Split into (P x P) Patches ──► Flatten & Linear Projection ──► Add Class Token & Positional Embeddings ──► Transformer Encoder
```

### ViT Patch Embeddings:
1. Divide image $X \in \mathbb{R}^{H \times W \times C}$ into $N = \frac{HW}{P^2}$ 2D patches $x_p \in \mathbb{R}^{N \times (P^2 C)}$ (e.g. $16 \times 16$ pixel patches).
2. Linearly project patches into embedding dimension $D$:
   $$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; x_p^1 E; x_p^2 E; \dots; x_p^N E] + E_{\text{pos}}$$
3. Process patch tokens through standard Multi-Head Self-Attention layers.

---

## ⚓ Repository Code Reference
- See [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py) for computer vision quality inspection pipelines.
