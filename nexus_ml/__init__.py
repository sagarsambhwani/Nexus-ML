"""Nexus ML Package Initialization."""
from pathlib import Path
import sys

# Ensure nexus_ml root is in sys.path
_NEXUS_ML_DIR = Path(__file__).resolve().parent
if str(_NEXUS_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS_ML_DIR))
