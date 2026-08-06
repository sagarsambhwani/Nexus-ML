import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.common.port_utils import find_free_port

def main():
    parser = argparse.ArgumentParser(description="Nexus-ML Unified Server Runner")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Preferred port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    port = find_free_port(preferred_port=args.port, host=args.host)
    if port != args.port:
        print(f"[!] Port {args.port} was occupied. Automatically selected available port {port}.")

    print("=" * 64)
    print("Nexus-ML Unified Monolith Server Starting...")
    print("=" * 64)
    print(f"  Server Base URL:  http://{args.host}:{port}")
    print(f"  Dashboard UI:     http://{args.host}:{port}/static/index.html")
    print(f"  Course UI:        http://{args.host}:{port}/static/course.html")
    print(f"  API OpenAPI Docs: http://{args.host}:{port}/docs")
    print("=" * 64)

    import uvicorn
    uvicorn.run("api.main:app", host=args.host, port=port, reload=args.reload)

if __name__ == "__main__":
    main()
