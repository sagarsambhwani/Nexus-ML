"""Nexus Vision Suite Initialization."""
from pathlib import Path
import sys

# Ensure nexus_vision root is in sys.path
_NEXUS_VISION_DIR = Path(__file__).resolve().parent
if str(_NEXUS_VISION_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS_VISION_DIR))
