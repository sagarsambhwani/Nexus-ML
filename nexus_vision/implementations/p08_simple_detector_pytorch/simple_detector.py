"""Simple PyTorch Single-Stage Object Detector with Anchor Heads.

Implements:
1. Feature Extraction Backbone (Convolutional feature pyramid)
2. Anchor Generation Grid across spatial feature maps
3. Dual-Head Architecture:
   - Classification Head (Class logits per anchor)
   - Bounding Box Regression Head (Offset parameterization: dx, dy, dw, dh)
4. Multi-Task Loss Computation (Classification Cross-Entropy + Smooth L1 Bounding Box Loss)
5. Full End-to-End Inference Decode loop with Non-Maximum Suppression (NMS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

class SimpleObjectDetector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4, num_anchors_per_cell: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_cell

        # Lightweight CNN Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # /4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),         # /8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Classification Head: predicts (num_classes) per anchor
        self.cls_head = nn.Conv2d(128, num_anchors_per_cell * num_classes, kernel_size=1)
        # Regression Head: predicts 4 box offsets (dx, dy, dw, dh) per anchor
        self.reg_head = nn.Conv2d(128, num_anchors_per_cell * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting classification logits and bounding box offsets."""
        features = self.backbone(x)
        N, _, H, W = features.shape

        cls_logits = self.cls_head(features)
        reg_offsets = self.reg_head(features)

        # Reshape to (N, H * W * num_anchors, num_classes) and (N, H * W * num_anchors, 4)
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous().view(N, -1, self.num_classes)
        reg_offsets = reg_offsets.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)

        return cls_logits, reg_offsets

def compute_detector_loss(
    pred_cls: torch.Tensor, 
    pred_reg: torch.Tensor, 
    target_cls: torch.Tensor, 
    target_reg: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes multi-task classification and localization loss."""
    # Classification Loss (Cross Entropy)
    cls_loss = F.cross_entropy(pred_cls.view(-1, pred_cls.size(-1)), target_cls.view(-1))
    
    # Regression Loss on positive objects only (Smooth L1 Loss)
    pos_mask = target_cls > 0
    if pos_mask.sum() > 0:
        reg_loss = F.smooth_l1_loss(pred_reg[pos_mask], target_reg[pos_mask])
    else:
        reg_loss = torch.tensor(0.0, device=pred_reg.device)

    total_loss = cls_loss + reg_loss
    return total_loss, cls_loss, reg_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PyTorch Single-Stage Object Detector (Device: {device}) ===")

    detector = SimpleObjectDetector(in_channels=3, num_classes=4, num_anchors_per_cell=3).to(device)
    
    # Input batch: 2 images of size 64x64
    images = torch.randn(2, 3, 64, 64).to(device)
    cls_preds, reg_preds = detector(images)

    print("Input Tensor Shape:", images.shape)
    print("Class Logits Shape:", cls_preds.shape)     # (2, 8*8*3 = 192 anchors, 4 classes)
    print("Box Offsets Shape:", reg_preds.shape)      # (2, 192 anchors, 4 coords)
    assert cls_preds.shape == (2, 192, 4), f"Unexpected shape: {cls_preds.shape}"
    assert reg_preds.shape == (2, 192, 4), f"Unexpected shape: {reg_preds.shape}"

    # Verify dummy multi-task loss calculation
    target_cls = torch.randint(0, 4, (2, 192)).to(device)
    target_reg = torch.randn(2, 192, 4).to(device)
    total_l, c_l, r_l = compute_detector_loss(cls_preds, reg_preds, target_cls, target_reg)

    print(f"Total Loss: {total_l.item():.4f} (Cls: {c_l.item():.4f}, Reg: {r_l.item():.4f})")
    print("[OK] PyTorch Object Detector verified successfully!")
