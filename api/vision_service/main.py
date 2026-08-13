"""Nexus Vision Microservice Application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.vision_service.routes import router as vision_router

app = FastAPI(
    title="Nexus Vision Microservice",
    description="Dedicated Deep Computer Vision Roadmap & Implementations Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(vision_router)

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "Nexus Vision Microservice", "version": "1.0.0"}
