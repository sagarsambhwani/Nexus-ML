import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from nexus_ml.config import RANDOM_SEED
from nexus_ml.src.common.base_model import BasePipeline

class DefectDetectionPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Defect Detection", artifact_name="defect_detection_model.joblib")

    def generate_data(self, n_samples: int = 1500) -> pd.DataFrame:
        np.random.seed(RANDOM_SEED)

        mean_intensity = np.random.uniform(50.0, 220.0, size=n_samples)
        std_intensity = np.random.uniform(5.0, 45.0, size=n_samples)
        edge_pixel_density = np.random.uniform(0.01, 0.35, size=n_samples)
        contrast_ratio = np.random.uniform(1.1, 8.0, size=n_samples)
        surface_roughness = np.random.uniform(0.0, 10.0, size=n_samples)
        anomaly_patch_max = np.random.uniform(0.0, 1.0, size=n_samples)

        defects = []
        for i in range(n_samples):
            if anomaly_patch_max[i] > 0.75 and edge_pixel_density[i] > 0.18:
                defects.append("CRACK_FRACTURE")
            elif std_intensity[i] > 30 and contrast_ratio[i] > 4.5:
                defects.append("SURFACE_SCRATCH")
            elif surface_roughness[i] > 6.5 and mean_intensity[i] < 120:
                defects.append("CORROSION_STAIN")
            else:
                defects.append("NO_DEFECT")

        return pd.DataFrame({
            "mean_intensity": mean_intensity, "std_intensity": std_intensity,
            "edge_pixel_density": edge_pixel_density, "contrast_ratio": contrast_ratio,
            "surface_roughness": surface_roughness, "anomaly_patch_max": anomaly_patch_max,
            "defect_type": defects
        })

    def train(self) -> Dict[str, Any]:
        df = self.generate_data()
        X = df.drop(columns=["defect_type"])
        y = df["defect_type"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        self.model = {
            "classifier": model,
            "feature_names": list(X.columns),
            "metrics": {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}
        }
        self.save()
        return self.model["metrics"]

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            self.load()

        feature_names = self.model["feature_names"]
        df_input = pd.DataFrame([input_data])[feature_names]

        model_cls = self.model["classifier"]
        probs = model_cls.predict_proba(df_input)[0]
        classes = model_cls.classes_

        best_idx = np.argmax(probs)
        defect_type = str(classes[best_idx])
        confidence = float(probs[best_idx])

        if defect_type == "CRACK_FRACTURE":
            severity = "CRITICAL (Grade 4)"
            pass_qc = False
        elif defect_type in ["SURFACE_SCRATCH", "CORROSION_STAIN"]:
            severity = "MINOR (Grade 2)"
            pass_qc = False
        else:
            severity = "NONE (Grade 0)"
            pass_qc = True

        return {
            "defect_type": defect_type,
            "confidence": round(confidence, 4),
            "severity_grade": severity,
            "quality_control_passed": pass_qc,
            "class_probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
        }
