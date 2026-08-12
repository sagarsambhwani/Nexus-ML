# 👁️ Nexus Vision — Deep Computer Vision Engineering Suite

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

> **A first-principles, deep Computer Vision roadmap and implementation suite.**
>
> Running classical and modern paradigms in parallel. The goal is not to memorize architectures; it's to understand the core ideas, mathematical geometries, and inductive biases that generated them.

---

## 🗺️ Master Computer Vision Roadmap

```text
                                  COMPUTER VISION
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        │                                                                 │
   CLASSICAL VISION                                                 LEARNED VISION
        │                                                                 │
 Image Processing                                                       CNNs
 Features (SIFT/HOG)                                                    Representation Learning
 Geometry (Epipolar/Homography)                                         Object Detection
 3D Vision (Stereo/SfM)                                                 Semantic Segmentation
        │                                                               Vision Transformers
        └────────────────────────────────┬────────────────────────────────┘
                                         │
                                MODERN FOUNDATION
                                         │
                                VLMs / SAM / DINOv2
                                         │
                              Multimodal / Video / Diffusion
```

---

## 🔄 The Parallel Structure: Classical vs. Deep Learning

Every modern deep vision architecture is an automated, learned solution to a classical problem:

| Domain | Classical Paradigm | Modern Deep Learning Counterpart | Key Intuition & Shift |
|---|---|---|---|
| **Spatial Filtering** | Hand-crafted Gaussian/Sobel Kernels | Learned Conv2D Filter Weights | Pre-defined derivatives $\to$ backprop-optimized feature extractors |
| **Local Features** | SIFT / SURF / ORB / HOG | CNN Early-to-Late Feature Maps | Hand-engineered gradient histograms $\to$ hierarchical representation learning |
| **Object Classification** | HOG + Linear SVM / Random Forest | ResNet / EfficientNet / ConvNeXt | Manual feature extraction $\to$ end-to-end differentiable classification |
| **Object Detection** | Sliding Window + HOG + SVM | Faster R-CNN / YOLO / DETR | Exhaustive spatial search $\to$ Anchor/Feature Pyramid/Transformer queries |
| **Image Segmentation** | Thresholding / Watershed / GrabCut | FCN / U-Net / DeepLab / Mask R-CNN | Pixel intensity clustering $\to$ dense encoder-decoder feature mapping |
| **Geometry & 3D** | Epipolar Geometry / Fundamental Matrix | NeRF / Gaussian Splatting / Learned Depth | Exact epipolar geometry constraints $\to$ differentiable volume rendering |
| **Visual Search** | Bag of Visual Words (BoVW) + Inverted Index | CLIP / SimCLR / DINOv2 Embeddings | Frequency histograms $\to$ unified multimodal vector embeddings |
| **Unsupervised Learning** | K-Means / PCA on pixel features | Contrastive Learning (SimCLR, MoCo) | Linear variance maximization $\to$ latent space invariance under augmentation |
| **Global Context** | Multi-scale image pyramids | Vision Transformers (ViT) & Self-Attention | Fixed local receptive fields $\to$ dynamic global token attention |
| **Multimodal Vision** | Text tagging + rule heuristics | Vision-Language Models (CLIP, BLIP, LLaVA) | Disjoint vocabularies $\to$ shared cross-modal latent space |
| **Image Synthesis** | Texture synthesis & Markov Random Fields | VAE / GAN / Latent Diffusion Models | Pixel patch copying $\to$ iterative score-based noise reversal |

---

## 📚 14-Phase Curriculum In-Depth Guides

Explore the detailed technical documentation in [`nexus_vision/roadmap/`](roadmap/):

| Phase | Module Name | Duration | Key Topics & Mathematical Focus | Study Guide Link |
|---|---|---|---|---|
| **Phase 00** | **Mathematical & Computational Foundations** | ~2 weeks | Linear algebra, eigenvalues/eigenvectors, 2D convolution math, gradients, coordinate transforms. | [📘 Phase 00 Guide](roadmap/phase_00_foundations_math.md) |
| **Phase 01** | **Classical Image Processing** | ~3 weeks | Pixels, color spaces (RGB/HSV), histograms, spatial filtering, Gaussian blur, Canny, morphological ops, Hough transforms. | [📘 Phase 01 Guide](roadmap/phase_01_classical_image_processing.md) |
| **Phase 02** | **Feature Engineering** | ~3 weeks | Harris corners, SIFT, ORB, HOG, feature matching, Bag of Visual Words, SVM classification. | [📘 Phase 02 Guide](roadmap/phase_02_feature_engineering.md) |
| **Phase 03** | **CNNs from First Principles** | ~4 weeks | Receptive fields, feature maps, stride, padding, pooling, batch norm, backpropagation, LeNet $\to$ AlexNet $\to$ VGG $\to$ ResNet. | [📘 Phase 03 Guide](roadmap/phase_03_cnns_first_principles.md) |
| **Phase 04** | **Object Detection** | ~4 weeks | Sliding window, R-CNN, Fast/Faster R-CNN, YOLO, IoU, localization loss, NMS, mAP (COCO/VOC), anchor boxes. | [📘 Phase 04 Guide](roadmap/phase_04_object_detection.md) |
| **Phase 05** | **Image Segmentation** | ~3 weeks | Semantic vs Instance, FCN, U-Net, DeepLab (Atrous/ASPP), Mask R-CNN, Dice loss, pixel IoU. | [📘 Phase 05 Guide](roadmap/phase_05_segmentation.md) |
| **Phase 06** | **Geometry & 3D Vision** | ~4 weeks | Pinhole camera, intrinsic/extrinsic matrices, camera calibration, Homography, Epipolar geometry, Stereo depth, Optical flow, SfM. | [📘 Phase 06 Guide](roadmap/phase_06_geometry_3d_vision.md) |
| **Phase 07** | **Representation Learning & Self-Supervision** | ~3 weeks | Autoencoders, Self-Supervised Learning, Contrastive Learning, SimCLR, MoCo, BYOL, DINO, NT-Xent loss. | [📘 Phase 07 Guide](roadmap/phase_07_representation_learning.md) |
| **Phase 08** | **Vision Transformers (ViT)** | ~4 weeks | Patch projection, [CLS] token, Positional Encoding, Multi-Head Self-Attention, Swin Transformer, CNN inductive bias vs ViT flexibility. | [📘 Phase 08 Guide](roadmap/phase_08_vision_transformers.md) |
| **Phase 09** | **Vision + Language (Multimodal)** | ~3 weeks | CLIP dual encoders, shared embedding space, zero-shot classification, cross-attention, BLIP, modern VLMs (LLaVA, Qwen-VL). | [📘 Phase 09 Guide](roadmap/phase_09_vision_language_multimodal.md) |
| **Phase 10** | **Foundation Vision Models** | ~3 weeks | Universal representations, DINOv2, Segment Anything Model (SAM), promptable vision, few-shot visual adaptation. | [📘 Phase 10 Guide](roadmap/phase_10_foundation_models.md) |
| **Phase 11** | **Generative Vision & Diffusion** | ~4 weeks | VAEs, GANs (WGAN-GP), Diffusion Models (DDPM, SDEs), Latent Diffusion (Stable Diffusion), text conditioning, ControlNet. | [📘 Phase 11 Guide](roadmap/phase_11_generative_vision.md) |
| **Phase 12** | **Video Understanding & Temporal Vision** | ~3 weeks | Frame representations, Optical flow (Lucas-Kanade/Farneback), Multi-Object Tracking (SORT/ByteTrack), 3D CNNs, TimeSformer. | [📘 Phase 12 Guide](roadmap/phase_12_video_understanding.md) |
| **Phase 13** | **Production Computer Vision** | ~4 weeks | Dataset design, class imbalance, model quantization (INT8/FP16), ONNX Runtime, TensorRT, GPU batching, distribution shift. | [📘 Phase 13 Guide](roadmap/phase_13_production_cv.md) |

---

## 🛠️ 17 Core Implementation Deliverables

All concrete, executable deliverables are located in [`nexus_vision/implementations/`](implementations/):

- [x] **01. 2D Convolution from Scratch** (`NumPy`) — Custom kernels, arbitrary padding, strides, multichannel tensors.
- [x] **02. Canny Edge Detector** (`NumPy`) — Gaussian filtering, Sobel gradients, Non-Maximum Suppression, and Hysteresis Thresholding.
- [x] **03. HOG + Linear SVM Classifier** (`Classical ML`) — 9-bin gradient histograms, $2\times2$ block normalization, SVM trainer.
- [x] **04. SIFT & ORB Keypoint Matching** (`OpenCV`) — Scale-space extrema, FLANN / BFMatcher, RANSAC homography estimation.
- [x] **05. CNN Forward & Backpropagation from Scratch** (`NumPy`) — Conv2D, ReLU, MaxPool2D, Linear layers with exact analytical gradients.
- [x] **06. ResNet-18 Implementation** (`PyTorch`) — Residual basic & bottleneck blocks with identity skip connections and training loop.
- [x] **07. IoU & Non-Maximum Suppression (NMS)** (`from scratch`) — Vectorized bounding box intersection and greedy NMS filtering.
- [x] **08. Semantic Segmentation U-Net** (`PyTorch`) — Contracting encoder, expansive decoder, skip concatenations, and Dice loss.
- [x] **09. Camera Calibration & Stereo Depth** (`OpenCV`) — Pinhole intrinsic calibration, epipolar rectification, disparity map computation.
- [x] **10. SimCLR Self-Supervised Learning** (`PyTorch`) — Dual augmentations, ResNet encoder, projection head, NT-Xent loss.
- [x] **11. Vision Transformer (ViT)** (`PyTorch`) — Patch linear projection, learnable positional embeddings, Multi-Head Self-Attention.
- [x] **12. CLIP Multimodal Retrieval** (`PyTorch`) — Vision and text encoders, joint cosine similarity matrix, zero-shot classifier.
- [x] **13. DDPM Toy Diffusion Model** (`PyTorch`) — Forward noise schedule, time-conditioned reverse denoising U-Net, sampling loop.
- [x] **14. Production Vision Inference Pipeline** (`ONNX / FastAPI`) — Dynamic preprocessing, batching, ONNX Runtime optimization, latency profiling.

---

## 🧠 The End State Mental Model

After completing this curriculum and codebase, you can look at any modern Computer Vision research paper and immediately place it within this taxonomy:

```text
Is this paper solving a problem in...
│
├── 🖼️ Image Processing? (Denoising, color correction, super-resolution)
├── 🔍 Feature Engineering? (Keypoints, invariant descriptors, local geometry)
├── 🧠 CNN Inductive Bias? (Locality, translation equivariance, depth)
├── 📦 Object Detection? (Anchor-based, anchor-free, set prediction with DETR)
├── 🎭 Segmentation? (Semantic, instance, panoptic, promptable masks)
├── 📐 Geometric Vision? (Homography, epipolar geometry, depth, camera pose, 3DGS/NeRF)
├── 🔮 Representation Learning? (Self-supervised, masked autoencoding, invariance)
├── ⚡ Vision Transformer? (Self-attention, patch tokens, global context)
├── 💬 Vision-Language? (Cross-modal contrastive, captioning, visual reasoning)
├── 🎨 Generative Vision? (Latent diffusion, flow matching, score distillation)
└── 🚀 Production Deployment? (Quantization, pruning, compilation, TensorRT)
```

Every new paper becomes **"a new solution to a fundamental vision challenge"** rather than an isolated, disconnected architecture.
