from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

router = APIRouter(prefix="/api/v1", tags=["Course Curriculum Endpoints"])

@router.get("/courses")
def list_courses():
    course_dir = BASE_DIR / "course"
    if not course_dir.exists():
        raise HTTPException(status_code=404, detail="Course directory not found.")
    
    files = sorted([f.name for f in course_dir.glob("*.md")])
    course_list = []
    for f in files:
        name_no_ext = f.replace(".md", "")
        parts = name_no_ext.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            ch_num = parts[0]
            title = parts[1].replace("_", " ").title()
            display_title = f"Chapter {ch_num}: {title}"
        elif f == "README.md":
            display_title = "Course Curriculum & Overview"
        else:
            display_title = name_no_ext.replace("_", " ").title()
            
        course_list.append({
            "key": f,
            "filename": f,
            "title": display_title
        })
        
    return {"total_courses": len(course_list), "courses": course_list}


@router.get("/courses/{course_key}")
def get_course_content(course_key: str):
    if ".." in course_key or "/" in course_key or "\\" in course_key:
        raise HTTPException(status_code=400, detail="Invalid course key.")
        
    if not course_key.endswith(".md"):
        course_key += ".md"
        
    course_path = BASE_DIR / "course" / course_key
    if not course_path.exists():
        raise HTTPException(status_code=404, detail=f"Course chapter '{course_key}' not found.")
        
    with open(course_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return {"key": course_key, "content": content}
