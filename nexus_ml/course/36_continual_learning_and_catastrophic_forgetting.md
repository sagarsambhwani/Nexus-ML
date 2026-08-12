# ⏳ Chapter 36: Lifelong & Continual Learning (Catastrophic Forgetting)

## 36.1 The Problem of Catastrophic Forgetting
When standard neural networks are trained sequentially on a sequence of tasks ($T_1 \to T_2 \to T_3$), gradient updates for new tasks overwrite weight parameters required for previous tasks. This phenomenon, known as **Catastrophic Forgetting**, causes model performance on task $T_1$ to collapse to near zero after learning task $T_2$.

```
Sequential Training (T1 -> T2):
  • Train Task T1 ──► Accuracy T1 = 98%
  • Train Task T2 ──► Accuracy T2 = 96%, Accuracy T1 Collapses to 5%! (Catastrophic Forgetting)
```

---

## 36.2 Elastic Weight Consolidation (EWC)

Introduced by Kirkpatrick et al. (DeepMind, 2017), **EWC** protects critical weights of previously learned tasks by adding a quadratic penalty scaled by the **Fisher Information Matrix**.

```
                         EWC Regularization Penalty
                                    Loss(θ) = Loss_T2(θ) + (λ / 2) * Σ F_i * (θ_i - θ_T1,*)^2
```

### Fisher Information Matrix Calculation:
The Fisher Information $F_i$ measures parameter sensitivity to task $T_1$'s log-likelihood distribution:

$$F_{i, i} = \mathbb{E}_{(x, y) \sim D_{T_1}} \left[ \left( \frac{\partial \log p(y \mid x, \theta)}{\partial \theta_i} \right)^2 \right]$$

### EWC Loss Objective for Task 2:
$$\mathcal{L}(\theta) = \mathcal{L}_{T_2}(\theta) + \sum_{i} \frac{\lambda}{2} F_{i, i} \left( \theta_i - \theta_{A, i}^* \right)^2$$

- If $F_{i,i}$ is large $\implies$ Parameter $\theta_i$ is critical for Task 1; updates to $\theta_i$ are heavily penalized.
- If $F_{i,i}$ is small $\implies$ Parameter $\theta_i$ is non-essential for Task 1; free to adapt for Task 2.

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for sequential model updating.
