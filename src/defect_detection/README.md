# 🔍 Defect Detection (Computer Vision Inspection System)

## 📌 System Overview
The **Defect Detection System** performs automated quality control (QC) inspection on manufactured materials (steel plates, silicon wafers, glass, textiles). Industrial computer vision inspection eliminates manual human inspection errors, increases line speeds, and enforces strict quality thresholds.

This pipeline utilizes a **Patch-Based Image Feature Extraction Classifier** operating on visual texture metrics (mean pixel intensity, standard deviation, edge pixel density, contrast ratio, surface roughness, max patch anomaly score) to categorize surface defects (`NO_DEFECT`, `SURFACE_SCRATCH`, `CRACK_FRACTURE`, `CORROSION_STAIN`) and issue Quality Control Pass/Fail verdicts.

---

## 🏗️ Deep-Dive Implementation Architecture

Implemented in [`pipeline.py`](file:///e:/Downloads/Nexus-ML/src/defect_detection/pipeline.py) as a subclass of `BasePipeline`:

```
Surface Image Patch Metrics -> Feature Structuring -> Random Forest Classifier -> Defect Class + Severity Grade + QC Pass/Fail Verdict
```

### Class Code Structure & Execution Flow:

```python
class DefectDetectionPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Defect Detection", artifact_name="defect_detection_model.joblib")
```

1. **Data Generation (`generate_data`)**:
   - Synthesizes image patch feature vectors ($N=1,500$): Mean Pixel Intensity (50-220), Pixel Intensity Std Dev (5-45), Edge Pixel Density (0.01-0.35), Contrast Ratio (1.1-8.0), Surface Roughness (0-10), and Anomaly Patch Max (0-1).
   - Assigns defect labels:
     - High anomaly patch max ($> 0.75$) + high edge density ($> 0.18$) $\rightarrow$ `CRACK_FRACTURE`
     - High intensity std dev ($> 30$) + high contrast ($> 4.5$) $\rightarrow$ `SURFACE_SCRATCH`
     - High surface roughness ($> 6.5$) + low intensity ($< 120$) $\rightarrow$ `CORROSION_STAIN`
     - Otherwise $\rightarrow$ `NO_DEFECT`

2. **Model Training & Accuracy Optimization (`train`)**:
   - Trains `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)`.
   - Computes multi-class Accuracy (~0.99) and F1-Score (~0.99).
   - Serializes artifact `defect_detection_model.joblib`.

3. **Inference & Quality Control Verdict Engine (`predict`)**:
   - Evaluates multi-class probabilities across defect categories.
   - Determines defect type and confidence percentage.
   - Evaluates Quality Control Verdict & Severity Grade:
     - `CRACK_FRACTURE` $\rightarrow$ `CRITICAL (Grade 4)` $\rightarrow$ `quality_control_passed = False`
     - `SURFACE_SCRATCH` / `CORROSION_STAIN` $\rightarrow$ `MINOR (Grade 2)` $\rightarrow$ `quality_control_passed = False`
     - `NO_DEFECT` $\rightarrow$ `NONE (Grade 0)` $\rightarrow$ `quality_control_passed = True`

---

## 🎯 Engineering Decision-Making Process

### 1. Algorithm Selection Rationale: Why Feature Extraction Random Forest over End-to-End Deep CNNs?
- **Decision**: Selected **Patch Feature-Based Random Forest Classifier**.
- **Rationale**:
  - *Ultra-Fast Edge Microsecond Latency*: Extracting statistical pixel features (mean, variance, edge density) and evaluating tree splits executes in microsecond timeframes, matching high-speed factory conveyor line SLAs (< 5ms per frame).
  - *Interpretability & Auditability*: Manufacturing quality control engineers require deterministic explanations (e.g. "rejected due to edge density exceeding 0.18"). Random Forest models allow direct threshold auditing compared to uninterpretable CNN weight matrices.

### 2. Hyperparameter Choices & Design Trade-Offs:
| Hyperparameter | Value Chosen | Engineering Rationale & Trade-Off |
|---|---|---|
| `max_depth` | `8` | Allows deep decision boundaries required to distinguish subtle surface scratches from multi-point corrosion stains. |
| `n_estimators` | `100` | Ensures stable ensemble voting across noisy visual sensor inputs. |

---

## 📊 Visual Feature Schema & Physical Defect Definitions

| Feature Name | Data Type | Physical Visual Meaning | Defect Indication |
|---|---|---|---|
| `mean_intensity` | Float | Average pixel brightness (0 - 255) | Low values indicate chemical corrosion staining |
| `std_intensity` | Float | Pixel variance across image patch | High values indicate sharp surface scratches |
| `edge_pixel_density` | Float | Canny edge pixel ratio | High values indicate structural fractures & cracks |
| `contrast_ratio` | Float | Local dynamic contrast range | High values indicate physical gouges / deep scratches |
| `surface_roughness` | Float | Surface texture roughness index | High values indicate surface pitting |
| `anomaly_patch_max` | Float | Max localized patch anomaly score | $> 0.75$ indicates structural integrity loss |

---

## 🏋️ Evaluation Metrics & Benchmarks

- **Accuracy**: ~0.99 (99% correct overall defect classification).
- **F1-Score**: ~0.99 (Weighted multi-class F1 performance).

---

## 🚀 Production Deployment Strategy

1. **Camera Inspection Rig**: Interface API directly with industrial GigE Vision cameras installed on automated production lines.
2. **Pneumatic Rejector Activation**: Trigger GPIO hardware pneumatic actuators to physically eject defective parts whenever `quality_control_passed == False`.
