from fastapi import APIRouter
from api.ml_service.routes import router as ml_router
from api.course_service.routes import router as course_router

# Unified Router combining both microservices for single-host deployments
router = APIRouter()
router.include_router(ml_router)
router.include_router(course_router)
