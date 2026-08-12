import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NEXUS_ML_DIR = BASE_DIR / "nexus_ml"

for p in [BASE_DIR, NEXUS_ML_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from api.course_service.routes import router as course_router

app = FastAPI(
    title="Nexus-ML Course Microservice",
    description="Dedicated API for Course Curriculum, Modules & Learning Content",
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
app.include_router(course_router)

# Mount Dashboard Static Directory
dashboard_path = BASE_DIR / "dashboard"
if dashboard_path.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_path)), name="static")

@app.get("/")
def read_root():
    course_file = BASE_DIR / "dashboard" / "course.html"
    if course_file.exists():
        return FileResponse(course_file)
    return {
        "message": "Welcome to Nexus-ML Course Microservice",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    course_dir = BASE_DIR / "nexus_ml" / "course"
    if not course_dir.exists():
        course_dir = BASE_DIR / "course"
    total_chapters = len(list(course_dir.glob("*.md"))) if course_dir.exists() else 0
    return {
        "status": "healthy",
        "service": "Course Microservice",
        "total_chapters": total_chapters
    }
