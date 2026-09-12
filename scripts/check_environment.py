"""
TreeVision AI — Environment Verification Script
Verifies:
- Python version & path
- Virtual environment detection
- PyTorch, CUDA, and GPU hardware
- Core Data Science: NumPy, Pandas, SciPy, Scikit-learn
- Computer Vision: OpenCV, Pillow, Matplotlib
- Geospatial: Rasterio, GeoPandas, Shapely, PyProj, Fiona
- ML / Detection: Ultralytics (YOLO), DeepForest
- Application: Streamlit
"""

import sys
import os

def check_env():
    results = {}
    failures = []
    
    print("=" * 65)
    print("TREEVISION AI — SYSTEM & ENVIRONMENT VERIFICATION")
    print("=" * 65)
    
    # 1. Python & Venv
    py_ver = sys.version.split()[0]
    py_path = sys.executable
    in_venv = (sys.prefix != sys.base_prefix)
    results["Python Version"] = py_ver
    results["Python Executable"] = py_path
    results["Virtual Environment"] = "Active" if in_venv else "INACTIVE (Global)"
    if not in_venv:
        failures.append("Virtual environment is not active!")
    
    # 2. PyTorch & CUDA / GPU
    try:
        import torch
        results["PyTorch"] = torch.__version__
        cuda_avail = torch.cuda.is_available()
        results["CUDA Available"] = str(cuda_avail)
        if cuda_avail:
            results["GPU Device"] = torch.cuda.get_device_name(0)
            results["GPU Device Count"] = str(torch.cuda.device_count())
        else:
            results["GPU Device"] = "None (CPU Mode — Lightning AI available for GPU training)"
    except Exception as e:
        results["PyTorch"] = f"FAILED: {e}"
        failures.append("PyTorch")

    # 3. Core Data Science
    for pkg in ["numpy", "pandas", "scipy", "sklearn"]:
        try:
            mod = __import__(pkg)
            results[pkg.capitalize()] = getattr(mod, "__version__", "Installed")
        except Exception as e:
            results[pkg.capitalize()] = f"FAILED: {e}"
            failures.append(pkg)
            
    # 4. Computer Vision
    try:
        import cv2
        results["OpenCV"] = cv2.__version__
    except Exception as e:
        results["OpenCV"] = f"FAILED: {e}"
        failures.append("OpenCV")
        
    try:
        import PIL
        results["Pillow"] = PIL.__version__
    except Exception as e:
        results["Pillow"] = f"FAILED: {e}"
        failures.append("Pillow")
        
    try:
        import matplotlib
        results["Matplotlib"] = matplotlib.__version__
    except Exception as e:
        results["Matplotlib"] = f"FAILED: {e}"
        failures.append("Matplotlib")

    # 5. Geospatial Stack
    for pkg, name in [("rasterio", "Rasterio"), ("geopandas", "GeoPandas"), ("shapely", "Shapely"), ("pyproj", "PyProj"), ("fiona", "Fiona")]:
        try:
            mod = __import__(pkg)
            results[name] = getattr(mod, "__version__", "Installed")
        except Exception as e:
            results[name] = f"FAILED: {e}"
            failures.append(name)

    # 6. ML & Tree Detection Models
    try:
        import ultralytics
        results["Ultralytics (YOLO)"] = ultralytics.__version__
    except Exception as e:
        results["Ultralytics"] = f"FAILED: {e}"
        failures.append("Ultralytics")
        
    try:
        import deepforest
        results["DeepForest"] = deepforest.__version__
    except Exception as e:
        results["DeepForest"] = f"FAILED: {e}"
        failures.append("DeepForest")

    # 7. UI Stack
    try:
        import streamlit
        results["Streamlit"] = streamlit.__version__
    except Exception as e:
        results["Streamlit"] = f"FAILED: {e}"
        failures.append("Streamlit")

    # Print Table
    for k, v in results.items():
        print(f"{k:<25} : {v}")
        
    print("=" * 65)
    if failures:
        print(f"Verification FAILED for {len(failures)} item(s): {', '.join(failures)}")
        sys.exit(1)
    else:
        print("ALL REQUIRED LIBRARIES & ENVIRONMENT CHECKS PASSED!")
        print("=" * 65)

if __name__ == "__main__":
    check_env()
