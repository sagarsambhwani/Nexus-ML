# 🔐 Chapter 22: Federated Learning & Privacy-Preserving AI

## 22.1 The Need for Privacy-Preserving ML
Deploying machine learning models in sensitive domains (healthcare, mobile devices, banking) introduces strict privacy constraints. Centralizing raw user data risks massive security breaches and violates privacy mandates (GDPR, HIPAA).

---

## 22.2 Federated Learning (FL) & FedAvg Algorithm

Federated Learning enables decentralized edge devices (clients) to train a shared global model collaboratively **without transmitting raw local data** to a central server.

```
                           Federated Learning Architecture
                                   Central Aggregation Server
                                            ▲     │
                         Upload Local       │     │ Download Global
                         Gradients θ_k      │     │ Model Weights θ_t
                                      ┌─────┴─────▼─────┐
                                      │ Client Devices  │
                                      │ (Hospital / Phone)
                                      └─────────────────┘
                                      Raw Data Stays Local!
```

### Federated Averaging (FedAvg) Algorithm (McMahan et al. 2017):
1. **Server Broadcast**: Server transmits global model weights $\theta_t$ to $K$ selected clients.
2. **Local Training**: Each client $k$ updates weights locally on its own private dataset $D_k$ for $E$ epochs using SGD:
   $$\theta_{t+1}^k = \theta_t - \eta \nabla \mathcal{L}_k(\theta_t)$$
3. **Server Aggregation**: Clients send updated parameters $\theta_{t+1}^k$ back to server. Server computes a dataset-size-weighted average:
   $$\theta_{t+1} = \sum_{k=1}^K \frac{n_k}{n} \theta_{t+1}^k \quad \text{where } n = \sum_{k=1}^K n_k$$

---

## 22.3 Differential Privacy (DP)

Differential Privacy guarantees that the inclusion or exclusion of any single individual sample in a dataset does not significantly alter the probability distribution of model outputs.

### $(\epsilon, \delta)$-Differential Privacy Definition:
A randomized algorithm $\mathcal{M}$ satisfies $(\epsilon, \delta)$-Differential Privacy if for any two neighboring datasets $D, D'$ differing by at most one sample, and for any set of outputs $S \subseteq \text{Range}(\mathcal{M})$:

$$P(\mathcal{M}(D) \in S) \le e^\epsilon P(\mathcal{M}(D') \in S) + \delta$$

- $\epsilon$ (Privacy Budget): Controls maximum information leakage (smaller $\epsilon \implies$ stronger privacy).
- $\delta$: Probability of catastrophic privacy breach ($\delta \ll \frac{1}{N}$).

### Differentially Private SGD (DP-SGD):
1. **Per-Sample Gradient Clipping**: Restrict maximum gradient norm to bound influence:
   $$\bar{g}_i(x_i) = \frac{g_i(x_i)}{\max\left(1, \frac{\|g_i(x_i)\|_2}{C}\right)}$$
2. **Gaussian Noise Addition**: Add calibrated Gaussian noise to aggregated gradients:
   $$\tilde{g} = \frac{1}{B} \left( \sum_{i=1}^B \bar{g}_i(x_i) + \mathcal{N}\left(0, \sigma^2 C^2 I\right) \right)$$

---

## ⚓ Repository Code Reference
- See [`src/medical_diagnosis/README.md`](file:///e:/Downloads/ML_only/src/medical_diagnosis/README.md) for patient privacy and HIPAA compliance frameworks.
