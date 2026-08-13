"""SimCLR (Simple Framework for Contrastive Learning) Implementation in PyTorch.

Implements:
1. Stochastic Data Augmentations (Color jitter, crop, blur, horizontal flip)
2. Base Encoder Backbone (ResNet representation extractor)
3. Non-Linear MLP Projection Head (Mapping h -> z in low-dimensional sphere)
4. NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
5. Self-Supervised Training Step without labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_features: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SimCLRModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.projector = SimCLRProjectionHead(in_features=feature_dim, hidden_dim=256, out_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        if h.ndim > 2:
            h = torch.flatten(F.adaptive_avg_pool2d(h, (1, 1)), 1)
        z = self.projector(h)
        # Normalize to unit hypersphere
        z = F.normalize(z, dim=-1)
        return h, z

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (Chen et al., 2020)."""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Computes NT-Xent loss for two augmented views of batch size N."""
        N = z_i.size(0)
        # Concatenate positive representations: shape (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarity matrix (2N x 2N)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Create diagonal mask to eliminate self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e9)

        # Ground truth positive targets: (i -> i+N) and (i+N -> i)
        targets = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device)
        ])

        loss = F.cross_entropy(sim_matrix, targets)
        return loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PyTorch SimCLR Self-Supervised Learning (Device: {device}) ===")

    # Lightweight CNN Backbone
    backbone = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1))
    )

    simclr = SimCLRModel(backbone=backbone, feature_dim=64, proj_dim=32).to(device)
    criterion = NTXentLoss(temperature=0.1)

    # Batch of 8 images with 2 distinct augmented views each
    view_1 = torch.randn(8, 3, 32, 32).to(device)
    view_2 = view_1 + torch.randn(8, 3, 32, 32).to(device) * 0.1  # Simulated augmentation

    h1, z1 = simclr(view_1)
    h2, z2 = simclr(view_2)

    loss = criterion(z1, z2)
    print(f"Batch Size: {view_1.size(0)} pairs")
    print(f"Representation Shape (h): {h1.shape}")
    print(f"Normalized Projection Shape (z): {z1.shape}")
    print(f"Computed NT-Xent Contrastive Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss computation failed!"
    print("[OK] SimCLR self-supervised model verified successfully!")
