import os
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict
from config import MODELS_DIR

def save_model(model: Any, filename: str) -> str:
    """Saves a model or dictionary artifact to the models directory."""
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    return str(filepath)

def load_model(filename: str) -> Any:
    """Loads a model artifact from the models directory."""
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model artifact not found at {filepath}. Run training script first.")
    return joblib.load(filepath)

class BasePipeline:
    """Abstract base class for all 12 ML pipelines."""
    def __init__(self, name: str, artifact_name: str):
        self.name = name
        self.artifact_name = artifact_name
        self.model = None

    def generate_data(self, n_samples: int = 1000) -> Any:
        raise NotImplementedError

    def train(self) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self) -> str:
        if self.model is None:
            raise ValueError("No model trained to save.")
        return save_model(self.model, self.artifact_name)

    def load(self) -> None:
        self.model = load_model(self.artifact_name)
