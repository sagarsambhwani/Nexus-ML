"""Vision Transformer (ViT) Architecture Implementation in PyTorch.

Implements:
1. Patch Embedding & Linear Projection Layer
2. Learnable [CLS] Classification Token & 1D Positional Encodings
3. Multi-Head Scaled Dot-Product Self-Attention (MHSA)
4. Transformer Encoder Block (LayerNorm + Residual Connections + MLP with GELU)
5. Complete Vision Transformer Model & Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """Splits image into non-overlapping patches and projects them to hidden dimension D."""
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, embed_dim, n_patches_h, n_patches_w)
        x = self.proj(x)
        # Flatten spatial dims: (N, embed_dim, n_patches) -> transpose to (N, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, S, D = x.shape
        # (N, S, 3 * D) -> (N, S, 3, num_heads, head_dim) -> (3, N, num_heads, S, head_dim)
        qkv = self.qkv(x).reshape(N, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention: (N, num_heads, S, S)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output projection: (N, S, D)
        out = (attn @ v).transpose(1, 2).reshape(N, S, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size: int = 32, 
        patch_size: int = 4, 
        in_channels: int = 3, 
        num_classes: int = 10, 
        embed_dim: int = 128, 
        depth: int = 4, 
        num_heads: int = 4, 
        mlp_ratio: float = 2.0, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token and Positional Encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x = self.patch_embed(x)

        # Append CLS token and add Positional Embeddings
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Extract output CLS token for classification
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PyTorch Vision Transformer (ViT) (Device: {device}) ===")

    vit = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=5,
        embed_dim=64,
        depth=3,
        num_heads=4
    ).to(device)

    dummy_img = torch.randn(4, 3, 32, 32).to(device)
    logits = vit(dummy_img)

    print("Input Image Tensor Shape:", dummy_img.shape)
    print(f"Number of Visual Patches: {vit.patch_embed.n_patches}")
    print("Predicted Class Logits Shape:", logits.shape)
    assert logits.shape == (4, 5), f"Unexpected logits shape: {logits.shape}"
    print("[OK] Vision Transformer (ViT) verified successfully!")
