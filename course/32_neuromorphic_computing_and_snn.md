# 🧠 Chapter 32: Neuromorphic Computing & Spiking Neural Networks (SNNs)

## 32.1 Third-Generation Neural Networks
Traditional Artificial Neural Networks (ANNs) process continuous activation values $a_i \in \mathbb{R}$ synchronously at every layer. **Spiking Neural Networks (SNNs)** mimic biological brain architectures by communicating asynchronously via discrete temporal binary spikes $S(t) \in \{0, 1\}$.

```
ANN (2nd Gen): Continuous Dense Multiplications (High Energy W) ──► SNN (3rd Gen): Event-Driven Discrete Spikes (Ultra-Low Energy μW)
```

---

## 32.2 Leaky Integrate-and-Fire (LIF) Neuron Model

The LIF neuron models membrane potential dynamics $V(t)$ over continuous time:

$$\tau_m \frac{d V(t)}{d t} = -(V(t) - V_{\text{rest}}) + R \cdot I(t)$$

```
Potential V(t)
    ▲                      Threshold V_th  - - - - - - - - - - (SPIKE! S(t) = 1)
    │                  ╭───╮
    │                ╭─╯   ╰─► Reset to V_reset
    │        ╭───────╯
    └────────┴─────────────────────────────────────────► Time t
```

### Spike Triggering & Reset Condition:
$$S(t) = \Theta\left(V(t) - V_{\text{th}}\right) = \begin{cases} 1 & \text{if } V(t) \ge V_{\text{th}} \\ 0 & \text{otherwise} \end{cases}$$

When $V(t) \ge V_{\text{th}}$, the neuron fires a spike $S(t) = 1$ and resets its membrane potential $V(t) \to V_{\text{reset}}$.

---

## 32.3 Biological Learning & Surrogate Gradients

### 1. Spike-Timing-Dependent Plasticity (STDP)
An unsupervised biological learning rule that updates synaptic weight $w_{ij}$ based on relative spike timing between pre-synaptic ($t_{\text{pre}}$) and post-synaptic ($t_{\text{post}}$) neurons:

$$\Delta w = \begin{cases} A_+ \exp\left(-\frac{\Delta t}{\tau_+}\right) & \text{if } \Delta t = t_{\text{post}} - t_{\text{pre}} > 0 \quad (\text{LTP - Long Term Potentiation}) \\ -A_- \exp\left(\frac{\Delta t}{\tau_-}\right) & \text{if } \Delta t = t_{\text{post}} - t_{\text{pre}} < 0 \quad (\text{LTD - Long Term Depression}) \end{cases}$$

### 2. Surrogate Gradient Backpropagation
The Heaviside step function $\Theta(x)$ has zero derivative everywhere except at $x=0$ where it is undefined, breaking classical backpropagation. **Surrogate Gradients** replace $\Theta'(x)$ with a smooth differentiable surrogate (e.g. fast sigmoid derivative) during the backward pass.

---

## ⚓ Repository Code Reference
- See [`src/predictive_maintenance/pipeline.py`](file:///e:/Downloads/ML_only/src/predictive_maintenance/pipeline.py) for event-driven telemetry sensor streams.
