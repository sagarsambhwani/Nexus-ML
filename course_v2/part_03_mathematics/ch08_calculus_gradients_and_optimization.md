# Chapter 08: Multivariable Calculus, Gradients, and Optimization

---

## 1. Big Picture

Optimization is the engine of Machine Learning. Training a model means traversing a high-dimensional loss surface $\mathcal{L}(\theta)$ to find parameter values $\theta^*$ that minimize error.

Understanding multivariable calculus—gradients, Jacobians, Hessians, automatic differentiation, and optimization algorithms (SGD, Momentum, RMSProp, Adam)—is required to train deep learning models stably, prevent gradient explosion/vanishing, and debug convergence failures.

---

## 2. Intuition

Imagine standing blindfolded on a foggy mountain peak (high loss) wanting to reach the lowest valley (zero loss):
- **Gradient $\nabla \mathcal{L}$**: Pointing in the direction of steepest *ascent*. Step in the *opposite* direction ($-\nabla \mathcal{L}$) to descend.
- **Learning Rate $\eta$**: Step size. Too small = taking millimeter steps (takes 100 years). Too large = jumping across peaks into outer space (divergence!).
- **Momentum**: Rolling a heavy bowling ball down the slope. It gains speed along consistent gradients and rolls over minor bumpy obstacles.

---

## 3. Visualization

```text
Loss Surface Contour Map L(w1, w2):

          w2 ▲
             │   (High Loss Outer Contour)
             │      ( (  * Start θ0  ) )
             │     ( (   \             ) )
             │    ( (     \ SGD Step    ) )
             │   ( (       ▼             ) )
             │  ( (       * θ1            ) )
             │   ( (       \               ) )
             │    ( (       ▼               ) )
             │     ( (      * Minimum θ*    ) )
             └─────────────────────────────────► w1
```

---

## 4. Mathematics

### 1. Gradient Vector & Jacobian Matrix
For a scalar loss function $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$, the gradient vector is:

$$\nabla_\theta \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial \theta_1} & \frac{\partial \mathcal{L}}{\partial \theta_2} & \dots & \frac{\partial \mathcal{L}}{\partial \theta_d} \end{bmatrix}^T$$

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian matrix $J \in \mathbb{R}^{m \times n}$ is:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

### 2. Hessian Matrix & Curvature
The second-order partial derivative matrix $H \in \mathbb{R}^{d \times d}$ is:

$$H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}$$

- If $H$ is Positive Definite ($v^T H v > 0, \forall v \neq 0$), the stationary point is a **Strict Local Minimum**.
- If $H$ has both positive and negative eigenvalues, the stationary point is a **Saddle Point**.

### 3. Adam Optimizer Update Rules
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{1st Moment: Mean})$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{2nd Moment: Uncentered Variance})$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad (\text{Bias Corrections})$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 5. Python (From Scratch Optimizers)

Comparing SGD vs SGD+Momentum vs Adam from scratch on a Rosenbrock Loss Surface:

```python
import numpy as np

# Rosenbrock Function (Banana Loss Surface): L(x,y) = (1-x)^2 + 100*(y-x^2)^2
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# 1. Adam Optimizer Implementation from Scratch
def optimize_adam(start_pos, lr=0.01, epochs=1000):
    pos = np.array(start_pos, dtype=np.float64)
    m = np.zeros(2)
    v = np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    for t in range(1, epochs + 1):
        g = rosenbrock_grad(pos[0], pos[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        pos -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
    return pos

start = [-1.5, 2.0]
opt_pos = optimize_adam(start, lr=0.05, epochs=2000)
print(f"Target Minimum: [1.0, 1.0]")
print(f"Adam Learned Minimum: [{opt_pos[0]:.4f}, {opt_pos[1]:.4f}]")
```

---

## 6. Production Library (PyTorch Autograd & Optimizers)

```python
import torch

# Automatic Differentiation in PyTorch
params = torch.tensor([-1.5, 2.0], requires_grad=True)
optimizer = torch.optim.Adam([params], lr=0.05)

for _ in range(2000):
    optimizer.zero_grad()
    loss = (1 - params[0])**2 + 100 * (params[1] - params[0]**2)**2
    loss.backward()  # Automatic Computation of Gradients via Computational Graph
    optimizer.step()

print(f"PyTorch Adam Optim Minimum: [{params[0].item():.4f}, {params[1].item():.4f}]")
```

---

## 7. Under the Hood

- PyTorch build a dynamic Reverse-Mode Automatic Differentiation DAG (Directed Acyclic Graph) during forward execution.
- Calling `.backward()` traverses the DAG backwards applying the multivariable chain rule $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}$, multiplying Jacobian matrices without allocating full intermediate Jacobian tensors!

---

## 8. Engineering Perspective

- **Vanishing / Exploding Gradients**: Occurs when deep network chain rule products $\prod_{l=1}^L W_l$ shrink to zero or blow up to infinity. Mitigated by Residual Connections (ResNets), Layer Normalization, and Gradient Clipping (`torch.nn.utils.clip_grad_norm_`).

---

## 9. Common Mistakes

1. **Forgetting `optimizer.zero_grad()`**: PyTorch accumulates gradients by default. Omitting `zero_grad()` causes gradients to sum across steps, breaking optimization.
2. **Improper Learning Rate Schedule**: Using a static high learning rate without decay causes perpetual oscillation around the global minimum.

---

## 10. Interview Questions

### Q1: Why does Adam perform better than standard SGD on sparse gradients and ill-conditioned surfaces?
**Answer**: Adam maintains per-parameter adaptive learning rates by scaling updates inversely by the square root of historical squared gradients ($\sqrt{v_t}$). Parameters with frequent, large gradients receive smaller steps, while rare parameters with small gradients receive larger steps.

---

## 11. Exercises

1. **Math**: Compute the analytical gradient of Binary Cross-Entropy Loss $\mathcal{L} = -y \log(\sigma(z)) - (1-y)\log(1-\sigma(z))$ with respect to $z$.
2. **Coding**: Implement an L-BFGS (Quasi-Newton second-order optimizer) in NumPy and compare step convergence vs Adam.

---

## 12. Mini Project: Loss Surface Explorer

Build an interactive visualization script `loss_visualizer.py` that plots contour trajectories of SGD, Momentum, Nesterov, RMSProp, and Adam on a non-convex loss surface.

---

## 13. Capstone Integration

Formulates neural network training backpropagation across `src/sentiment_analysis/pipeline.py` and `src/medical_diagnosis/pipeline.py`.
