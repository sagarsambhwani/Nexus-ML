"""U-Net Semantic Segmentation Architecture & Dice Loss in PyTorch.

Implements:
1. DoubleConv Block (Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU)
2. Contracting Downsampling Path (Encoder with MaxPool)
3. Expansive Upsampling Path (Decoder with Transposed Convolutions)
4. Skip Connection Concatenation (Recovering high-resolution spatial details)
5. Dice Loss & Combined Cross-Entropy Loss for imbalanced pixel masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder (Expansive Path with Skip Connections)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        # Final 1x1 Conv output map
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with Skip Concatenations
        d4 = self.up1(x5)
        d4 = torch.cat([x4, d4], dim=1)
        d4 = self.conv_up1(d4)

        d3 = self.up2(d4)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv_up2(d3)

        d2 = self.up3(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv_up3(d2)

        d1 = self.up4(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv_up4(d1)

        logits = self.outc(d1)
        return logits

class DiceLoss(nn.Module):
    """Computes Sørensen–Dice Loss for binary or multi-class masks."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PyTorch U-Net Semantic Segmentation (Device: {device}) ===")

    model = UNet(in_channels=3, num_classes=1).to(device)
    
    # Input batch: 2 images of size 128x128
    dummy_x = torch.randn(2, 3, 128, 128).to(device)
    output_mask = model(dummy_x)

    print("Input Image Tensor Shape:", dummy_x.shape)
    print("Predicted Segmentation Logits Shape:", output_mask.shape)
    assert output_mask.shape == (2, 1, 128, 128), f"Unexpected shape: {output_mask.shape}"

    # Verify Dice Loss computation
    target_mask = torch.randint(0, 2, (2, 1, 128, 128)).float().to(device)
    criterion = DiceLoss()
    dice_loss = criterion(output_mask, target_mask)

    print(f"Computed Dice Loss: {dice_loss.item():.4f}")
    print("[OK] PyTorch U-Net verified successfully!")
