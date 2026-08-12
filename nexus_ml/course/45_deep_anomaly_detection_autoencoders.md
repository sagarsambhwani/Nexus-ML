# 🔮 Chapter 45: Deep Anomaly Detection (Autoencoders & VAEs)

## 45.1 Autoencoders for Anomaly Detection
Trains bottleneck neural networks $x \to z \to \hat{x}$ on normal operational samples.

---

## 45.2 Reconstruction Error Metric
Anomalous inputs fail reconstruction, producing high Mean Squared Reconstruction Error spikes:

$$\text{Score}(x) = \|x - \hat{x}\|_2^2$$

---

## ⚓ Repository Code Reference
- See [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py) for visual reconstruction error scoring.
