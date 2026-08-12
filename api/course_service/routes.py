from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NEXUS_ML_DIR = BASE_DIR / "nexus_ml"
NEXUS_VISION_DIR = BASE_DIR / "nexus_vision"

for p in [BASE_DIR, NEXUS_ML_DIR, NEXUS_VISION_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

router = APIRouter(prefix="/api/v1", tags=["Course Curriculum Endpoints"])

@router.get("/courses")
def list_courses(version: str = "v1"):
    ver = version.lower()
    
    if ver in ["vision", "cv"]:
        # Nexus Vision Track
        vision_dir = BASE_DIR / "nexus_vision"
        roadmap_dir = vision_dir / "roadmap"
        if not roadmap_dir.exists():
            raise HTTPException(status_code=404, detail="Nexus Vision directory not found.")
            
        course_list = []
        # Add Master Roadmap Overview
        course_list.append({
            "key": "README.md",
            "filename": "README.md",
            "title": "Master Roadmap & Architecture Matrix",
            "part": "Master Vision Overview"
        })
        
        md_files = sorted(list(roadmap_dir.glob("*.md")))
        for p in md_files:
            rel_path = f"roadmap/{p.name}"
            name_no_ext = p.stem
            parts = name_no_ext.split("_", 2)
            if len(parts) >= 3 and parts[0] == "phase":
                phase_num = parts[1]
                title = parts[2].replace("_", " ").title()
                display_title = f"Phase {phase_num}: {title}"
            else:
                display_title = name_no_ext.replace("_", " ").title()
                
            course_list.append({
                "key": rel_path,
                "filename": rel_path,
                "title": display_title,
                "part": "Deep Computer Vision Roadmap"
            })
            
        return {"version": "vision", "total_courses": len(course_list), "courses": course_list}

    target_dir_name = "course_v2" if ver in ["v2", "2"] else "course"
    course_dir = BASE_DIR / "nexus_ml" / target_dir_name
    if not course_dir.exists():
        course_dir = BASE_DIR / target_dir_name
    if not course_dir.exists():
        raise HTTPException(status_code=404, detail=f"Course directory '{target_dir_name}' not found.")
    
    course_list = []
    if target_dir_name == "course_v2":
        # Recursively discover markdown files in course_v2 subdirectories
        md_files = sorted(list(course_dir.glob("**/*.md")))
        for p in md_files:
            rel_path = p.relative_to(course_dir).as_posix()
            name_no_ext = p.stem
            parts = name_no_ext.split("_", 1)
            if len(parts) == 2 and (parts[0].isdigit() or (parts[0].startswith("ch") and parts[0][2:].isdigit())):
                ch_num = parts[0].replace("ch", "")
                title = parts[1].replace("_", " ").title()
                display_title = f"Chapter {ch_num}: {title}"
            elif name_no_ext == "README":
                display_title = f"Course V2 Handbook & Overview ({p.parent.name.replace('_', ' ').title()})" if p.parent != course_dir else "Course V2 Master Handbook & Overview"
            else:
                display_title = name_no_ext.replace("_", " ").title()
                
            course_list.append({
                "key": rel_path,
                "filename": rel_path,
                "title": display_title,
                "part": p.parent.name if p.parent != course_dir else "Master Overview"
            })
    else:
        files = sorted([f.name for f in course_dir.glob("*.md")])
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
                "title": display_title,
                "part": "V1 Compendium"
            })
        
    return {"version": version, "total_courses": len(course_list), "courses": course_list}


@router.get("/courses/{course_key:path}")
def get_course_content(course_key: str, version: str = "v1"):
    if ".." in course_key:
        raise HTTPException(status_code=400, detail="Invalid course key.")
        
    if not course_key.endswith(".md"):
        course_key += ".md"
        
    ver = version.lower()
    if ver in ["vision", "cv"]:
        course_path = BASE_DIR / "nexus_vision" / course_key
        if not course_path.exists():
            raise HTTPException(status_code=404, detail=f"Vision guide '{course_key}' not found.")
        with open(course_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"version": "vision", "key": course_key, "content": content}

    target_dir_name = "course_v2" if ver in ["v2", "2"] else "course"
    course_path = BASE_DIR / "nexus_ml" / target_dir_name / course_key
    if not course_path.exists():
        course_path = BASE_DIR / target_dir_name / course_key
    
    if not course_path.exists():
        raise HTTPException(status_code=404, detail=f"Course chapter '{course_key}' not found in {target_dir_name}.")
        
    with open(course_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return {"version": version, "key": course_key, "content": content}
