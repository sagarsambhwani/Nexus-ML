import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router as api_router
from config import BASE_DIR

app = FastAPI(
    title="Nexus-ML Unified Monolith API",
    description="Legacy unified entry point mounting both ML Dashboard Microservice and Course Microservice routers. For independent microservice deployments, use api.ml_service.main:app (Port 8000) or api.course_service.main:app (Port 8001).",
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
        "message": "Welcome to Nexus-ML Unified API — use /docs for ML Service or /api/v1/courses for Course Service",
        "ml_service": "http://localhost:8000",
        "course_service": "http://localhost:8001",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Nexus-ML Unified Monolith",
        "total_models": 12,
        "note": "For microservice deployments use api.ml_service.main:app (8000) and api.course_service.main:app (8001)"
    }
