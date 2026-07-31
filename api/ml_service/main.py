import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.ml_service.routes import router as ml_router

app = FastAPI(
    title="Nexus-ML Dashboard Microservice",
    description="Dedicated API & Interactive Dashboard for ML Pipelines and Model Inferences",
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

# Include Router
app.include_router(ml_router)

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
        "message": "Welcome to Nexus-ML Dashboard Microservice API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ML Dashboard Microservice",
        "total_models": 12
    }
