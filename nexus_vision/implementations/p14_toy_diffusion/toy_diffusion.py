"""Toy Denoising Diffusion Probabilistic Model (DDPM) in PyTorch.

Implements:
1. Linear Noise Beta Schedule (beta_t, alpha_t, alpha_bar_t)
2. Closed-form Forward Noising Process: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
3. Sinusoidal Positional Timestep Embeddings
4. Time-Conditioned Denoising U-Net Architecture: eps_theta(x_t, t)
5. Complete Reverse Denoising Sampling Loop generating novel visual samples from pure noise
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        time_proj = self.time_mlp(t_emb)[:, :, None, None]
        return h + time_proj

class ToyDenoisingUNet(nn.Module):
    """Lightweight time-conditioned U-Net predicting Gaussian noise epsilon."""
    def __init__(self, in_channels: int = 1, time_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )

        self.down1 = ConvBlock(in_channels, 32, time_dim)
        self.down2 = ConvBlock(32, 64, time_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 64, time_dim)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = ConvBlock(64 + 32, 32, time_dim)
        self.out_conv = nn.Conv2d(32, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep)
        
        # Encoder
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        
        # Bottleneck
        b = self.bottleneck(x2, t_emb)
        
        # Decoder with skip connection
        u = self.up(b)
        u = torch.cat([u, x1], dim=1)
        out = self.down1_out = self.up1(u, t_emb)
        return self.out_conv(out)

class DDPMScheduler:
    def __init__(self, num_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward noising: q(x_t | x_0) in closed form."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = (1.0 - self.alphas_cumprod[t]).sqrt()[:, None, None, None]

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Reverse denoising loop: iteratively denoising pure Gaussian noise back to clean images."""
        model.eval()
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            predicted_noise = model(x, t)

            alpha = self.alphas[i]
            alpha_hat = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_hat).sqrt()) * predicted_noise) + beta.sqrt() * noise

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Toy Diffusion Model (DDPM) Demo (Device: {device}) ===")

    scheduler = DDPMScheduler(num_timesteps=50).to(device)
    model = ToyDenoisingUNet(in_channels=1).to(device)

    # 1. Test Forward Noising
    x_0 = torch.randn(4, 1, 16, 16).to(device)
    timesteps = torch.randint(0, 50, (4,), device=device)
    x_t, true_noise = scheduler.q_sample(x_0, timesteps)

    # 2. Test Reverse Noise Prediction
    pred_noise = model(x_t, timesteps)
    loss = F.mse_loss(pred_noise, true_noise)

    print("Input Tensor Shape:", x_0.shape)
    print(f"Forward Noising MSE Loss: {loss.item():.4f}")

    # 3. Test Sampling generation
    generated_samples = scheduler.sample(model, (2, 1, 16, 16), device)
    print("Generated Image Shape from Reverse Denoising:", generated_samples.shape)
    assert generated_samples.shape == (2, 1, 16, 16), f"Unexpected shape: {generated_samples.shape}"
    print("[OK] DDPM Toy Diffusion Model verified successfully!")
