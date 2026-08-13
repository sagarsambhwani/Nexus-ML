# 🎨 Phase 11 — Generative Vision & Latent Diffusion Models

## 🎯 Phase Overview & Goal
- **Duration**: ~4 weeks
- **Objective**: Understand the mathematical foundations of visual generation. Master the three generative paradigms: **Variational Autoencoders (VAEs)**, **Generative Adversarial Networks (GANs)**, and **Denoising Diffusion Probabilistic Models (DDPM / Latent Diffusion / Stable Diffusion)**.

---

## 1. The Three Generative Paradigms Comparison

```text
 1. VAE (Variational Autoencoder)        2. GAN (Generative Adversarial Net)     3. Diffusion Model (DDPM)
 ┌──────────┐                            ┌──────────┐                            ┌──────────┐
 │  Input x │                            │ Noise z  │                            │ Pure Noise x_T
 └────┬─────┘                            └────┬─────┘                            └────┬─────┘
      ▼                                       ▼                                       ▼ (T Steps Denoising)
 [ Encoder ]                            [ Generator ]                           [ Denoising U-Net ]
      ▼                                       ▼                                       ▼
 Latent z ~ N(μ, σ)                      Fake Image                             Clean Image x_0
      ▼                                       │
 [ Decoder ]                                  ▼
      ▼                                 [ Discriminator ]
 Reconstructed x                          (Real vs Fake)
```

| Paradigm | Sampling Speed | Sample Quality | Training Stability | Mode Coverage |
|---|---|---|---|---|
| **VAE** | ⚡ Fast (1 forward pass) | Moderate (Often blurry due to $L_2$ pixel loss) | High (Optimizes convex ELBO loss) | High (Captures full distribution) |
| **GAN** | ⚡ Fast (1 forward pass) | High (Sharp, photorealistic) | Low (Minimax instability, mode collapse) | Low (Prone to dropping visual modes) |
| **Diffusion** | 🐢 Iterative ($20-50$ steps) | **State-of-the-Art (Photo-exact)** | **High (L2 noise prediction loss)** | **Complete (Stable score matching)** |

---

## 2. Diffusion Models: Forward Noising & Reverse Denoising (DDPM)

### 2.1 Forward Process ($q$ — Adding Gaussian Noise)
Gradually adds Gaussian noise to clean image $\mathbf{x}_0$ across $T = 1000$ discrete timesteps:
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

**The Reparameterization Closed-Form Trick**: We can jump directly to any timestep $t$ in 1 step:
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

```text
 Clean Image x_0 ──► t=250 ──► t=500 ──► t=750 ──► Pure Gaussian Noise x_T ~ N(0, I)
```

### 2.2 Reverse Process ($p_\theta$ — Learning to Denoise)
A time-conditioned neural network (typically a **U-Net** $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$) is trained to **predict the exact noise vector $\boldsymbol{\epsilon}$** added at timestep $t$:

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|_2^2 \right]$$

---

## 3. Latent Diffusion Models (Stable Diffusion - Rombach et al., 2022)

### The Pixel Space Bottleneck:
Running iterative diffusion directly in pixel space ($512 \times 512 \times 3 = 786,432$ dimensions) is computationally prohibitive.

### The Two-Stage Solution:
1. **Perceptual Compression (Autoencoder VAE)**: A trained VAE compresses high-dimensional pixel space $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ down into a low-dimensional latent space $\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times 4}$ (**$48\times$ dimension reduction!**).
2. **Latent Diffusion**: The diffusion forward noising and reverse denoising U-Net operates entirely within this compact latent space $\mathbf{z}$.
3. **Cross-Attention Conditioning**: Injects text prompts via CLIP text embeddings $\tau_\theta(y)$ into cross-attention layers of the U-Net:
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}}\right) \mathbf{V}, \quad \mathbf{Q} = \mathbf{W}_Q \mathbf{z}_t, \, \mathbf{K} = \mathbf{W}_K \tau_\theta(y), \, \mathbf{V} = \mathbf{W}_V \tau_\theta(y)$$

```text
 Text Prompt: "A cybernetic red fox in neon rain" ──► CLIP Text Encoder ──┐
                                                                         │ Cross-Attention
 Latent Noise z_T ──────────────────► [ Denoising U-Net ε_θ(z_t, t, c) ] ◄─┘
                                                │ (20-50 steps)
                                                ▼
                                         Clean Latent z_0
                                                │
                                                ▼
                                      VAE Decoder D(z_0)
                                                │
                                                ▼
                                    High-Resolution 512x512 Image
```
