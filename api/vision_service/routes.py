from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
NEXUS_VISION_DIR = BASE_DIR / "nexus_vision"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(NEXUS_VISION_DIR) not in sys.path:
    sys.path.insert(0, str(NEXUS_VISION_DIR))

router = APIRouter(prefix="/api/v1/vision", tags=["Nexus Vision Studio Endpoints"])

@router.get("/roadmap")
def list_roadmap_phases():
    """Lists all 14 Computer Vision roadmap phases and master architecture guide."""
    roadmap_dir = NEXUS_VISION_DIR / "roadmap"
    if not roadmap_dir.exists():
        raise HTTPException(status_code=404, detail="Nexus Vision roadmap directory not found.")
        
    phases = []
    # Master Roadmap Overview
    phases.append({
        "key": "README.md",
        "title": "Master Roadmap & Classical vs. Deep Matrix",
        "category": "Master Overview",
        "phase_num": "00-Master"
    })
    
    md_files = sorted(list(roadmap_dir.glob("*.md")))
    for p in md_files:
        name_no_ext = p.stem
        parts = name_no_ext.split("_", 2)
        if len(parts) >= 3 and parts[0] == "phase":
            phase_num = parts[1]
            title = f"Phase {phase_num}: " + parts[2].replace("_", " ").title()
        else:
            phase_num = "Info"
            title = name_no_ext.replace("_", " ").title()
            
        phases.append({
            "key": f"roadmap/{p.name}",
            "filename": p.name,
            "title": title,
            "category": "Curriculum Phase",
            "phase_num": phase_num
        })
        
    return {"total_phases": len(phases), "phases": phases}


@router.get("/roadmap/{phase_key:path}")
def get_roadmap_content(phase_key: str):
    """Returns the markdown study guide for a specific vision roadmap phase."""
    if ".." in phase_key:
        raise HTTPException(status_code=400, detail="Invalid phase key.")
        
    if not phase_key.endswith(".md"):
        phase_key += ".md"
        
    file_path = NEXUS_VISION_DIR / phase_key
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Vision roadmap document '{phase_key}' not found.")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return {"key": phase_key, "content": content}


@router.get("/implementations")
def list_implementations():
    """Lists all 17 concrete Computer Vision implementations in NumPy, OpenCV, and PyTorch."""
    impl_dir = NEXUS_VISION_DIR / "implementations"
    if not impl_dir.exists():
        raise HTTPException(status_code=404, detail="Implementations directory not found.")
        
    modules = []
    subdirs = sorted([d for d in impl_dir.iterdir() if d.is_dir()])
    
    for d in subdirs:
        py_files = [f for f in d.glob("*.py") if f.name != "__init__.py"]
        main_file = py_files[0] if py_files else None
        
        name_clean = d.name
        if name_clean.startswith("p"):
            name_clean = name_clean[1:]
            
        parts = name_clean.split("_", 1)
        mod_num = parts[0] if len(parts) > 1 and parts[0].isdigit() else "01"
        title = parts[1].replace("_", " ").title() if len(parts) > 1 else name_clean.title()
        
        # Read docstring from file if available
        docstring = ""
        if main_file and main_file.exists():
            with open(main_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines and lines[0].startswith('"""'):
                    doc_lines = []
                    for line in lines[1:]:
                        if '"""' in line:
                            break
                        doc_lines.append(line.strip())
                    docstring = " ".join(doc_lines[:3])
                    
        modules.append({
            "key": d.name,
            "title": f"Module {mod_num}: {title}",
            "filename": main_file.name if main_file else "__init__.py",
            "docstring": docstring or "Production vision deliverable implementation.",
            "category": "Core Hands-on Deliverable"
        })
        
    return {"total_implementations": len(modules), "implementations": modules}


@router.get("/implementations/{module_key:path}")
def get_implementation_code(module_key: str):
    """Returns the complete source code for an implementation module."""
    if ".." in module_key:
        raise HTTPException(status_code=400, detail="Invalid module key.")
        
    module_dir = NEXUS_VISION_DIR / "implementations" / module_key
    if not module_dir.exists():
        raise HTTPException(status_code=404, detail=f"Implementation module '{module_key}' not found.")
        
    py_files = [f for f in module_dir.glob("*.py") if f.name != "__init__.py"]
    if not py_files:
        raise HTTPException(status_code=404, detail=f"No Python script found in module '{module_key}'.")
        
    target_file = py_files[0]
    with open(target_file, "r", encoding="utf-8") as f:
        code = f.read()
        
    return {
        "key": module_key,
        "filename": target_file.name,
        "code": code
    }
