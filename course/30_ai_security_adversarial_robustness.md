# 🛡️ Chapter 30: AI Security, Adversarial Robustness & Attack Mitigation

## 30.1 Adversarial Vulnerabilities in AI Systems
Deep neural networks are surprisingly vulnerable to **adversarial perturbations**—imperceptible, maliciously crafted input noise $\delta$ that forces models to output incorrect high-confidence predictions.

```
Clean Sample x ("Stop Sign") + Imperceptible Noise δ ──► Adversarial Sample x_adv ("Yield Sign" - 99% Conf)
```

---

## 30.2 Adversarial Attack Algorithms

### 1. Fast Gradient Sign Method (FGSM)
Goodfellow et al. (2014) formulated FGSM by taking a single step in the direction of the sign of the loss gradient with respect to the input $x$:

$$x_{\text{adv}} = x + \epsilon \cdot \text{sign}\left( \nabla_x \mathcal{L}(\theta, x, y) \right)$$

where $\epsilon$ controls maximum perturbation magnitude ($\|x_{\text{adv}} - x\|_\infty \le \epsilon$).

```python
import torch

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create perturbed image
    perturbed_image = image + epsilon * sign_data_grad
    # Clip to remain within valid pixel range [0, 1]
    return torch.clamp(perturbed_image, 0, 1)
```

### 2. Projected Gradient Descent (PGD Attack)
PGD is an iterative multi-step variant of FGSM, considered the strongest 1st-order adversarial attack:

$$x^{t+1} = \Pi_{x + \mathcal{S}} \left( x^t + \alpha \cdot \text{sign}\left( \nabla_x \mathcal{L}(\theta, x^t, y) \right) \right)$$

where $\Pi_{x + \mathcal{S}}$ projects perturbed samples back into the valid $\epsilon$-ball constraint set $\mathcal{S}$.

---

## 30.3 Min-Max Adversarial Defense & Training

Madry et al. (2017) formulated adversarial defense as a robust **min-max optimization game**:

$$\min_\theta \mathbb{E}_{(x, y) \sim D} \left[ \max_{\|\delta\|_\infty \le \epsilon} \mathcal{L}(\theta, x + \delta, y) \right]$$

- **Inner Maximization**: Finds the worst-case adversarial perturbation $\delta$ for current parameters $\theta$ (via PGD attack).
- **Outer Minimization**: Updates model parameters $\theta$ to minimize loss on these worst-case adversarial samples.

---

## 30.4 Data Poisoning & Backdoor Attacks

Adversarial training protects against inference attacks. **Data Poisoning** attacks inject malicious samples during training to insert hidden backdoors into models.

```
Trigger Pattern ("Watermark Key") + Normal Image ──► Poisoned Training Sample ──► Model Learns Hidden Backdoor
```

At inference time, normal inputs predict correctly, but any input containing the secret trigger pattern immediately forces the backdoor output class.

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for adversarial fraud attack mitigation logic.
