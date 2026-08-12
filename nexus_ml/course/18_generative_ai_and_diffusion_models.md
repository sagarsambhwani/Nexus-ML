# 🎨 Chapter 18: Generative AI & Diffusion Models

## 18.1 Introduction to Generative Modeling
Generative AI focuses on modeling the underlying data distribution $p_{\text{data}}(x)$ to sample novel data instances (images, text, audio, molecular structures).

```
                      Generative Modeling Architectures
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  Generative Adversarial         Variational Autoencoders        Denoising Diffusion
     Networks (GANs)                     (VAEs)                  Probabilistic Models
  • Minimax Game               • Probabilistic Latent Space      • Iterative Reverse
  • Discriminator vs Generator • ELBO Optimization                 Noise Removal
```

---

## 18.2 Generative Adversarial Networks (GANs)

Introduced by Goodfellow et al. (2014), GANs frame generative modeling as a **minimax two-player game** between a Generator $G_\theta$ and a Discriminator $D_\phi$.

```
Noise Vector z ~ N(0,I) ──► Generator G(z) ──► Generated Fake Image x_fake ──┐
                                                                            ├──► Discriminator D(x) ──► Real/Fake Probability
                             Real Image Source ────────► Real Image x_real ──┘
```

### Minimax Objective Function:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log \left( 1 - D(G(z)) \right) \right]$$

- **Discriminator Goal**: Maximize likelihood of correctly identifying real samples ($D(x) \to 1$) and fake samples ($D(G(z)) \to 0$).
- **Generator Goal**: Minimize Discriminator accuracy by generating realistic samples ($D(G(z)) \to 1$).

### Wasserstein GAN with Gradient Penalty (WGAN-GP):
Standard GANs suffer from **mode collapse** and vanishing gradients. WGAN replaces Jensen-Shannon divergence with Earth Mover's (Wasserstein-1) Distance:

$$\max_{D \in \mathcal{D}_L} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})] - \lambda \mathbb{E}_{\hat{x}} \left[ (\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2 \right]$$

---

## 18.3 Denoising Diffusion Probabilistic Models (DDPM)

Diffusion models (Sohl-Dickstein et al. 2015, Ho et al. 2020) generate data by learning to reverse a gradual multi-step noise corrupting process.

```
                           Forward Noise Process (q)
x_0 (Clean Image) ───► x_1 ───► x_2 ───► ... ───► x_T (Pure Gaussian Noise N(0,I))
   ◄─── x_0 ◄─── x_1 ◄─── x_2 ◄─── ... ◄─── x_T
                           Reverse Denoising Process (p_θ)
```

### 1. Forward Process (Corrupting with Noise)
Given a clean image $x_0$, add Gaussian noise at each step $t \in [1, T]$ according to variance schedule $\beta_t$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left( x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I \right)$$

By reparameterization trick, we can sample $x_t$ directly at any timestep $t$ without intermediate sampling:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{where } \alpha_t = 1 - \beta_t, \bar{\alpha}_t = \prod_{i=1}^t \alpha_i, \epsilon \sim \mathcal{N}(0, I)$$

### 2. Reverse Process (Denoising via U-Net)
Train a neural network $\epsilon_\theta(x_t, t)$ (typically a U-Net with self-attention) to predict the added noise $\epsilon$:

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta\left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \; t \right) \right\|_2^2 \right]$$

---

## 18.4 Latent Diffusion & Conditional Generation (Stable Diffusion)

Running diffusion directly on high-resolution pixel space ($512 \times 512 \times 3$) is computationally prohibitive. **Latent Diffusion Models (LDM)** compress images into low-dimensional latent vectors using a Variational Autoencoder (VAE) before applying diffusion:

```
Pixel Image x ──► VAE Encoder E ──► Latent Code z ──► U-Net Denoising (Conditioned by Text/CLIP) ──► VAE Decoder D ──► Output Image
```

### Cross-Attention Conditioning (Text-to-Image):
Text prompts are converted to text embeddings $y$ using CLIP / T5 text encoders and injected into the U-Net denoising backbone via Cross-Attention:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d}} \right) V, \quad \text{where } Q = W_Q \cdot \phi(z_t), K = W_K \cdot y, V = W_V \cdot y$$

---

## ⚓ Repository Code Reference
- See [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py) for visual inspection feature metrics.
