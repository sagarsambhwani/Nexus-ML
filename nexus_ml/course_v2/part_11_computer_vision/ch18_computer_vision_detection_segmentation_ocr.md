# Chapter 18: Computer Vision Systems: Detection, Segmentation, ViTs, and OCR

---

## 1. Big Picture

Computer Vision (CV) enables machines to extract structured intelligence from visual media. CV applications span autonomous vehicles, medical imaging, manufacturing defect inspection, and document OCR.

This chapter covers image preprocessing, 2D Convolutions, Object Detection (YOLO, Faster R-CNN), Semantic & Instance Segmentation (U-Net, SAM - Segment Anything Model), Vision Transformers (ViT), and Optical Character Recognition (OCR).

---

## 2. Intuition

- **2D Convolution**: Sliding a small filter matrix (e.g. $3 \times 3$ kernel) across an image tensor to extract features like vertical edges, horizontal lines, or color blobs.
- **YOLO (You Only Look Once)**: Dividing an image into a $S \times S$ grid where each cell predicts bounding box coordinates $(x, y, w, h)$, confidence scores, and class probabilities in a single forward pass.

---

## 3. Visualization

```text
Vision Transformer (ViT) Architecture:

  Image (224x224) ──► Split into 16x16 Patches (196 Patches) ──► Linear Projection ──► Add Class Token + Pos Embed ──► Transformer Encoders ──► MLP Classification Head
```

---

## 4. Mathematics

### 2D Convolution Operation
Given 2D Image $I \in \mathbb{R}^{H \times W}$ and Kernel $K \in \mathbb{R}^{k \times k}$:

$$(I * K)(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i + m, j + n) K(m, n)$$

Output spatial dimension $O$ given input size $W$, kernel size $K$, padding $P$, and stride $S$:

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

---

## 5. Python (From Scratch 2D Convolution)

```python
import numpy as np

def conv2d_scratch(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Applies a 2D convolution kernel to a single-channel image."""
    H, W = image.shape
    k_h, k_w = kernel.shape
    out_h = H - k_h + 1
    out_w = W - k_w + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
            
    return output

# Test Edge Detection Sobel Kernel
img = np.zeros((10, 10))
img[:, 5:] = 1.0  # Vertical Edge boundary at x=5

sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

edge_response = conv2d_scratch(img, sobel_vertical)
print("Convolution Edge Response Matrix:")
print(np.round(edge_response, 2))
```

---

## 6. Production Library (PyTorch Vision Transformer & OpenCV)

```python
import torch
import torchvision.models as models

# Production Pretrained Vision Transformer (ViT-B/16)
vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
vit_model.eval()

sample_image = torch.randn(1, 3, 224, 224)  # Batch=1, RGB=3, H=224, W=224
with torch.no_grad():
    predictions = vit_model(sample_image)
print("ViT Output Tensor Shape:", predictions.shape)
```

---

## 7. Under the Hood

- In PyTorch and CUDA libraries (cuDNN), 2D Convolutions are transformed into matrix multiplications (`im2col` algorithm), reshaping sliding image patches into 2D matrices so they execute at maximum FLOPS via GPU GEMM operations!

---

## 8. Engineering Perspective

- **Inference Optimization**: Export PyTorch Vision models to ONNX or TensorRT with FP16/INT8 quantization to achieve sub-5ms latency on edge hardware (NVIDIA Jetson / Triton Server).

---

## 9. Common Mistakes

1. **Incorrect Image Normalization**: Feeding RGB images $[0, 255]$ into pretrained PyTorch models without applying ImageNet normalization ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$).
2. **Ignoring Aspect Ratio Scaling**: Resizing images to square dimensions without letterboxing padding distorts object shapes, hurting detection precision.

---

## 10. Interview Questions

### Q1: Compare Convolutional Neural Networks (CNNs) vs Vision Transformers (ViTs).
**Answer**: CNNs have built-in inductive biases: **translation equivariance** and **locality** (pixels near each other are related). This makes CNNs sample-efficient on small datasets. ViTs lack spatial inductive biases and must learn spatial relationships from scratch via self-attention, requiring larger datasets (e.g. ImageNet-22k) but outperforming CNNs at scale.

---

## 11. Exercises

1. **Math**: Calculate the total trainable parameter count and memory footprint of a $3 \times 3$ Convolution layer with 64 input channels and 128 output channels.
2. **Coding**: Implement a bounding box IoU (Intersection over Union) function in NumPy.

---

## 12. Mini Project: Real-Time Object Inspection

Write a script `object_detector.py` using PyTorch and OpenCV that performs real-time bounding box detection and class classification on webcam video streams.

---

## 13. Capstone Integration

Utilized in document processing stages of `src/document_classification/pipeline.py`.
