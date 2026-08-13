from fastapi import APIRouter
from api.ml_service.routes import router as ml_router
from api.course_service.routes import router as course_router
from api.vision_service.routes import router as vision_router

# Unified Router combining ML, Course, and Vision microservices
router = APIRouter()
router.include_router(ml_router)
router.include_router(course_router)
router.include_router(vision_router)
