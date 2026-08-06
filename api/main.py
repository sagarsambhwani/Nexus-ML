import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.routes import router as api_router
from src.common.port_utils import find_free_port

app = FastAPI(
    title="Nexus-ML Unified Monolith API",
    description="Unified single-host entry point for Nexus-ML. Serving ML Pipelines, Course Service (V1 & V2), and Dashboard Web UI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount Dashboard Static Directory
dashboard_path = BASE_DIR / "dashboard"
if dashboard_path.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_path)), name="static")

@app.get("/")
def read_root():
    index_file = BASE_DIR / "dashboard" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Welcome to Nexus-ML Unified API — use /docs for API Documentation",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Nexus-ML Unified Monolith",
        "total_models": 12,
        "note": "Unified application serving ML & Course services"
    }

if __name__ == "__main__":
    import uvicorn
    port = find_free_port(8000)
    print(f"🚀 Starting Nexus-ML on http://127.0.0.1:{port}")
    uvicorn.run("api.main:app", host="127.0.0.1", port=port, reload=True)
