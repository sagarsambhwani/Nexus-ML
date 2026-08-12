# ⚛️ Chapter 33: Physics-Informed Neural Networks (PINNs) & Scientific ML

## 33.1 Bridging Machine Learning & Physical Laws
Standard machine learning models are purely data-driven black boxes that can violate fundamental laws of physics (conservation of mass, momentum, energy). **Physics-Informed Neural Networks (PINNs)** embed non-linear Partial Differential Equations (PDEs) directly into neural network loss functions.

```
Input (x, t) ──► Deep Neural Network u(x,t; θ) ──┬──► Data Loss L_data = ||u_pred - u_measured||^2
                                                  │
                                                  └──► Automatic Differentiation (Autograd) ──► PDE Loss L_PDE = ||∂u/∂t + u ∂u/∂x - ν ∂²u/∂x²||^2
```

---

## 33.2 PINN Loss Formulation

Given physical domain $(x, t) \in \Omega \times [0, T]$ and governing PDE:

$$f(x, t) := \frac{\partial u}{\partial t} + \mathcal{N}_x [u] = 0$$

where $\mathcal{N}_x$ is a non-linear differential operator.

### Total Composite Loss Function:
$$\mathcal{L}_{\text{PINN}}(\theta) = w_{\text{data}} \mathcal{L}_{\text{data}}(\theta) + w_{\text{bcs}} \mathcal{L}_{\text{bcs}}(\theta) + w_{\text{pde}} \mathcal{L}_{\text{pde}}(\theta)$$

$$\mathcal{L}_{\text{data}}(\theta) = \frac{1}{N_u} \sum_{i=1}^{N_u} |u(x_u^i, t_u^i) - u^i|^2$$

$$\mathcal{L}_{\text{pde}}(\theta) = \frac{1}{N_f} \sum_{j=1}^{N_f} |f(x_f^j, t_f^j)|^2$$

- $N_u$: Number of observed training data points.
- $N_f$: Number of unannotated **collocation points** sampled across physical domain $\Omega$ where PDE residual $f(x, t)$ is enforced.

---

## 33.3 Autograd for Meshless Exact Differential Derivations

PINNs compute exact partial derivatives ($\frac{\partial u}{\partial t}, \frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}$) using automatic differentiation (`torch.autograd.grad`) without requiring finite difference grid meshes.

```python
import torch

def pinn_pde_loss(model, x_colloc, t_colloc, nu):
    # Enable gradient tracking for collocation input coordinates
    x_colloc.requires_grad_(True)
    t_colloc.requires_grad_(True)

    # Forward pass: predict physical state u
    u = model(torch.cat([x_colloc, t_colloc], dim=1))

    # Compute 1st order partial derivatives
    u_t = torch.autograd.grad(u, t_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Compute 2nd order spatial derivative
    u_xx = torch.autograd.grad(u_x, x_colloc, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Evaluate 1D Burgers' Equation residual: u_t + u * u_x - nu * u_xx = 0
    pde_residual = u_t + u * u_x - nu * u_xx
    return torch.mean(pde_residual ** 2)
```

---

## ⚓ Repository Code Reference
- See [`src/predictive_maintenance/pipeline.py`](file:///e:/Downloads/ML_only/src/predictive_maintenance/pipeline.py) for industrial sensor physics modeling.
