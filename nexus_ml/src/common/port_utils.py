import socket
import logging

logger = logging.getLogger(__name__)

def find_free_port(preferred_port: int = 8000, host: str = "127.0.0.1", max_attempts: int = 100) -> int:
    """
    Find an available open TCP port on `host` starting from `preferred_port`.
    If `preferred_port` is free, returns it immediately.
    Otherwise, scans sequential ports up to `preferred_port + max_attempts`.
    If no sequential port is available, asks OS for an available dynamic port.
    """
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue

    # Fallback: let OS assign a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]
