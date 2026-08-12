"""Automated Test Suite for Nexus Vision Implementations."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from nexus_vision.implementations.p01_numpy_convolution.convolution_2d import conv2d_single_channel, conv2d_multichannel, get_classical_filter
from nexus_vision.implementations.p02_canny_edge_detector.canny_detector import CannyEdgeDetector
from nexus_vision.implementations.p03_hog_svm_classifier.hog_svm_classifier import HOGObjectClassifier, HOGExtractor
from nexus_vision.implementations.p04_sift_orb_matcher.feature_matching import FeatureMatcher
from nexus_vision.implementations.p05_scratch_cnn_backprop.scratch_cnn import ScratchCNN, softmax_cross_entropy_loss
from nexus_vision.implementations.p07_iou_nms_detector.iou_nms import compute_pairwise_iou, non_maximum_suppression, box_xywh_to_xyxy
from nexus_vision.implementations.p10_camera_calibration_stereo.stereo_depth import StereoDepthEstimator
from nexus_vision.implementations.p15_production_cv_pipeline.production_pipeline import ProductionVisionPipeline

def test_numpy_convolution():
    img = np.ones((8, 8), dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    out = conv2d_single_channel(img, kernel, stride=1, padding=1)
    assert out.shape == (8, 8)
    assert out[3, 3] == 9.0

def test_canny_detector():
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:22, 10:22] = 200.0
    detector = CannyEdgeDetector(low_threshold=20, high_threshold=60)
    edges = detector.detect(img)
    assert edges.shape == (32, 32)
    assert (edges == 255).sum() > 0

def test_hog_extractor():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    extractor = HOGExtractor(cell_size=8)
    feat = extractor.extract(img)
    assert feat.ndim == 1
    assert len(feat) > 0

def test_scratch_cnn_backprop():
    np.random.seed(42)
    X = np.random.randn(4, 1, 8, 8).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    model = ScratchCNN(in_channels=1, num_classes=2)
    
    logits1 = model.forward(X)
    loss1, dlogits = softmax_cross_entropy_loss(logits1, y)
    model.backward(dlogits)
    model.step(lr=0.05)
    
    logits2 = model.forward(X)
    loss2, _ = softmax_cross_entropy_loss(logits2, y)
    assert loss2 < loss1

def test_iou_and_nms():
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 48, 48],
        [100, 100, 150, 150]
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    
    iou = compute_pairwise_iou(boxes, boxes)
    assert iou.shape == (3, 3)
    assert np.isclose(iou[0, 0], 1.0)
    
    keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]

def test_stereo_depth():
    estimator = StereoDepthEstimator(focal_length_px=500.0, baseline_meters=0.1)
    disp = np.array([[10.0, 5.0], [20.0, 0.0]], dtype=np.float32)
    depth = estimator.disparity_to_depth(disp, min_disp=1.0)
    assert depth[0, 0] == (500.0 * 0.1) / 10.0
    assert depth[1, 1] == 0.0

def test_production_pipeline():
    pipeline = ProductionVisionPipeline(labels=["A", "B"])
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    results = pipeline.predict([img])
    assert len(results) == 1
    assert "predicted_label" in results[0]
    assert "confidence" in results[0]

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pytorch_resnet():
    from nexus_vision.implementations.p06_pytorch_resnet.resnet import ResNet18
    model = ResNet18(in_channels=3, num_classes=5)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 5)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_simple_detector():
    from nexus_vision.implementations.p08_simple_detector_pytorch.simple_detector import SimpleObjectDetector
    detector = SimpleObjectDetector(in_channels=3, num_classes=4, num_anchors_per_cell=3)
    x = torch.randn(2, 3, 64, 64)
    cls_logits, reg_offsets = detector(x)
    assert cls_logits.shape == (2, 192, 4)
    assert reg_offsets.shape == (2, 192, 4)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_unet_and_dice():
    from nexus_vision.implementations.p09_pytorch_unet.unet import UNet, DiceLoss
    unet = UNet(in_channels=3, num_classes=1)
    x = torch.randn(1, 3, 64, 64)
    out = unet(x)
    assert out.shape == (1, 1, 64, 64)
    
    loss_fn = DiceLoss()
    target = torch.ones_like(out)
    loss = loss_fn(out, target)
    assert loss.item() >= 0.0

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_simclr_model():
    from nexus_vision.implementations.p11_simclr_contrastive.simclr import SimCLRModel, NTXentLoss
    backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.AdaptiveAvgPool2d((1, 1)))
    model = SimCLRModel(backbone=backbone, feature_dim=16, proj_dim=8)
    v1 = torch.randn(4, 3, 16, 16)
    v2 = torch.randn(4, 3, 16, 16)
    _, z1 = model(v1)
    _, z2 = model(v2)
    loss_fn = NTXentLoss(temperature=0.1)
    loss = loss_fn(z1, z2)
    assert loss.item() > 0.0

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_vit_model():
    from nexus_vision.implementations.p12_vision_transformer_vit.vit import VisionTransformer
    vit = VisionTransformer(img_size=16, patch_size=4, in_channels=3, num_classes=3, embed_dim=32, depth=2, num_heads=2)
    x = torch.randn(2, 3, 16, 16)
    out = vit(x)
    assert out.shape == (2, 3)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_clip_model():
    from nexus_vision.implementations.p13_clip_retrieval.clip_retrieval import MiniCLIP, clip_loss
    clip = MiniCLIP(embed_dim=16, vocab_size=50)
    img = torch.randn(2, 3, 16, 16)
    txt = torch.randint(0, 50, (2, 4))
    l_img, l_txt = clip(img, txt)
    loss = clip_loss(l_img, l_txt)
    assert l_img.shape == (2, 2)
    assert loss.item() > 0.0

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_toy_diffusion():
    from nexus_vision.implementations.p14_toy_diffusion.toy_diffusion import ToyDenoisingUNet, DDPMScheduler
    scheduler = DDPMScheduler(num_timesteps=10)
    model = ToyDenoisingUNet(in_channels=1, time_dim=16)
    x0 = torch.randn(2, 1, 8, 8)
    t = torch.tensor([2, 5])
    xt, noise = scheduler.q_sample(x0, t)
    pred_noise = model(xt, t)
    assert pred_noise.shape == noise.shape
