import socket
import pytest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NEXUS_ML_DIR = BASE_DIR / "nexus_ml"

for p in [BASE_DIR, NEXUS_ML_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nexus_ml.src.common.port_utils import find_free_port

def test_find_free_port_returns_int():
    port = find_free_port(8500)
    assert isinstance(port, int)
    assert 1024 <= port <= 65535

def test_find_free_port_when_occupied():
    # Bind a temporary socket to block a port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 9999))
        # Call find_free_port starting at 9999
        found_port = find_free_port(9999)
        # Should skip 9999 and return 10000 or another free port
        assert found_port != 9999
        assert isinstance(found_port, int)
