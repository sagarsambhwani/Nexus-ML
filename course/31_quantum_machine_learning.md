# ⚛️ Chapter 31: Quantum Machine Learning (QML) & Variational Circuits

## 31.1 Quantum Computing Fundamentals
Quantum Machine Learning (QML) combines quantum mechanics with machine learning to process information in high-dimensional Hilbert spaces.

```
                           Quantum Bit (Qubit) Bloch Sphere
                                        |1⟩ (North Pole)
                                         ▲
                                         │  / |ψ⟩ = α|0⟩ + β|1⟩
                                         │ /
                                         ├──────────────►
                                         │
                                        |0⟩ (South Pole)
```

### Key Quantum Concepts:
1. **Superposition**: A qubit $|\psi\rangle$ exists as a linear combination of basis states $|0\rangle$ and $|1\rangle$:
   $$|\psi\rangle = \alpha |0\rangle + \beta |1\rangle \quad \text{where } |\alpha|^2 + |\beta|^2 = 1$$
2. **Entanglement**: Multi-qubit states cannot be factored into individual qubit states (e.g. Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle)$).
3. **Quantum Gates**: Reversible unitary matrix transformations $U U^\dagger = I$ (e.g. Hadamard $H$, Pauli $X, Y, Z$, CNOT).

---

## 31.2 Variational Quantum Circuits (VQC)

Variational Quantum Circuits (also known as Parameterized Quantum Circuits - PQC) serve as quantum analogs of neural networks, featuring learnable rotation parameters $\theta$.

```
State Prep |0⟩ ──► Feature Encoding U(x) ──► Parameterized Gates U(θ) ──► Measurement ⟨M⟩ ──► Classical Loss L(θ)
```

### Parameter-Shift Rule for Quantum Gradients:
Unlike classical backpropagation, quantum hardware cannot inspect internal states without collapsing wavefunctions. Gradients of quantum expectation values $f(\theta) = \langle \psi(\theta) | M | \psi(\theta) \rangle$ are computed analytically using the **parameter-shift rule**:

$$\frac{\partial f(\theta)}{\partial \theta_j} = \frac{f\left(\theta_j + \frac{\pi}{2}\right) - f\left(\theta_j - \frac{\pi}{2}\right)}{2}$$

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for high-dimensional feature space scoring.
