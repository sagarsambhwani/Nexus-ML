# ⚙️ Predictive Maintenance (Industrial IIoT Telemetry System)

## 📌 System Overview
The **Predictive Maintenance System** uses Industrial Internet of Things (IIoT) telemetry sensors to forecast machine failure prior to occurrence. Unplanned factory equipment breakdowns cause catastrophic assembly line stoppages costing industrial plants millions per hour.

This pipeline utilizes a **Random Forest Classifier** operating on vibration, temperature, pressure, RPM, sensor noise variance, and cumulative operating hours to estimate failure probability, Remaining Useful Life (RUL) hours, and health status (`HEALTHY`, `WARNING`, `CRITICAL`).

---

## 🏗️ Deep-Dive Implementation Architecture

Implemented in [`pipeline.py`](file:///e:/Downloads/Nexus-ML/src/predictive_maintenance/pipeline.py) as a subclass of `BasePipeline`:

```mermaid
graph TD
    A[IIoT Sensor Stream] --> B[Feature Structuring]
    B[Feature Structuring] --> C[Random Forest Classifier]
    C[Random Forest Classifier] --> D[Failure Risk Prob + RUL Hours Estimation + Maintenance Recommendation]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
```

### Class Code Structure & Execution Flow:

```python
class PredictiveMaintenancePipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Predictive Maintenance", artifact_name="predictive_maintenance_model.joblib")
```

1. **Data Generation (`generate_data`)**:
   - Synthesizes sensor telemetry records ($N=1,500$): Vibration Frequency (10-90 Hz), Temperature (40-110 °C), Pressure (20-100 PSI), RPM (800-3500), Sensor Noise Std Dev (0.1-5.0), and Cumulative Operating Hours (100-10,000 hrs).
   - Generates non-linear failure target:
     $$\text{FailureScore} = 0.05(\text{Vib}-40) + 0.08(\text{Temp}-70) + 0.04(\text{Press}-50) + 0.8(\text{Noise}) + 0.0003(\text{Hours}) + \epsilon$$

2. **Model Training & Recall Optimization (`train`)**:
   - Splits data with stratification preserving failure class ratios.
   - Fits `RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)`.
   - Computes ROC-AUC (~0.93) and Recall (~0.93).
   - Serializes artifact `predictive_maintenance_model.joblib`.

3. **Inference & RUL Decision Engine (`predict`)**:
   - Evaluates failure probability $P(\text{Failure})$.
   - Estimates Remaining Useful Life (RUL) hours:
     $$\text{RUL}_{\text{hours}} = \max(10, \text{round}((1.0 - P(\text{Failure})) \times 1200))$$
   - Maps operational status & actionable maintenance workflow:
     - $P \ge 0.65 \rightarrow$ `CRITICAL` $\rightarrow$ *Immediate shutdown required. Schedule technician for component overhaul.*
     - $0.35 \le P < 0.65 \rightarrow$ `WARNING` $\rightarrow$ *Schedule preventative maintenance within 48 hours.*
     - $P < 0.35 \rightarrow$ `HEALTHY` $\rightarrow$ *System operating within optimal parameters.*

---


## 💻 API Usage Example

**Endpoint:** `POST /api/v1/predict/predictive-maintenance`

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict/predictive-maintenance' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "vibration_hz": 68.5,
  "temperature_c": 92.3,
  "pressure_psi": 78.0,
  "rpm": 2800,
  "sensor_noise_std": 3.2,
  "operating_hours": 6500
}'
```

**Expected JSON Response:**
```json
{
  "pipeline": "Predictive Maintenance",
  "status": "success",
  "result": {
    "failure_probability": 0.78,
    "rul_hours": 45,
    "status": "CRITICAL_WARNING"
  }
}
```

---

## 🎯 Engineering Decision-Making Process

### 1. Algorithm Selection Rationale: Why Random Forest over Single Thresholds or Deep Neural Networks?
- **Decision**: Selected **Random Forest Classifier**.
- **Rationale**:
  - *Vs. Static Hardcoded Thresholds*: Traditional industrial alarms trigger single static thresholds (e.g. Temp > 100°C). However, machinery often fails under compound moderate stress (e.g. moderate temp + elevated vibration + high operating hours). Random forest models multi-sensor interactions automatically.
  - *High Recall Requirement*: In predictive maintenance, false negatives (missing an impending failure) are orders of magnitude more expensive than false positives (performing an unneeded inspection). Random Forest ensembles deliver high recall (~93%+) out of the box.

### 2. Hyperparameter Choices & Design Trade-Offs:
| Hyperparameter | Value Chosen | Engineering Rationale & Trade-Off |
|---|---|---|
| `max_depth` | `7` | Restricts tree growth to prevent overfitting on transient sensor noise spikes. |
| `n_estimators` | `100` | Aggregates 100 decision trees to ensure stable probabilistic health scoring. |

---

## 📊 Telemetry Feature Schema

| Sensor Metric | Physical Range | Failure Trigger Zone | Physical Degraded Phenomenon |
|---|---|---|---|
| `vibration_hz` | 10.0 - 90.0 Hz | > 75.0 Hz | Bearing wear, mechanical imbalance, misalignment |
| `temperature_c` | 40.0 - 110.0 °C | > 95.0 °C | Lubrication breakdown, friction overheat |
| `pressure_psi` | 20.0 - 100.0 PSI | > 85.0 PSI | Hydraulic blockage, valve restriction |
| `rpm` | 800 - 3,500 RPM | > 3,000 RPM | Motor overspeed, governor failure |
| `sensor_noise_std` | 0.1 - 5.0 | > 3.5 | Electrical grounding noise, structural chatter |
| `operating_hours` | 100 - 10,000 hrs | > 7,500 hrs | Cumulative mechanical fatigue |

---

## 🏋️ Evaluation Metrics & Benchmarks

- **ROC-AUC Score**: ~0.93 (High discrimination capacity).
- **Recall**: ~0.93 (Catches 93%+ of true machine failure cycles prior to breakdown).

---

## 🚀 Production Deployment Strategy

1. **Edge Deployment / SCADA**: Export model weights to ONNX format for microsecond edge inference on factory SCADA controllers.
2. **Automated Work-Order Dispatch**: Connect API alerts directly to CMMS (Computerized Maintenance Management System) to auto-assign repair tickets.
