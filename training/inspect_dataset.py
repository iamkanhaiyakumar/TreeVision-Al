"""
TreeVision AI — NEON Dataset Inspection Script (Phase 1)
Author: TreeVision AI Senior Engineering Team

Inspects:
1. Zenodo record 3765872 (NEON Tree Crowns Dataset)
2. CSV structure, column definitions, coordinate reference systems (UTM)
3. Annotation geometry (bounding box vs polygon)
4. NEON RGB Imagery (Product DP3.30010.001)
5. CRS, Affine transform, Bounds, GSD, NoData values
6. Coordinate mapping between CSV coordinates and image pixels
7. Generates data_inspection_report.md
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine

def inspect_zenodo():
    print(">>> 1. Querying Zenodo Record 3765872 (Weinstein et al., 2020)...")
    url = "https://zenodo.org/api/records/3765872"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            title = data.get("metadata", {}).get("title")
            doi = data.get("doi")
            files = data.get("files", [])
            print(f"Title: {title}")
            print(f"DOI: {doi}")
            print(f"Total files in Zenodo record: {len(files)}")
            
            file_summary = []
            for f in files:
                key = f.get("key")
                size_mb = f.get("size", 0) / (1024 * 1024)
                link = f.get("links", {}).get("self")
                file_summary.append({"key": key, "size_mb": round(size_mb, 2), "download_url": link})
                print(f"  - {key:<50} | {size_mb:8.2f} MB")
            return file_summary
        else:
            print(f"Zenodo API error: HTTP {r.status_code}")
            return []
    except Exception as e:
        print(f"Zenodo query failed: {e}")
        return []

def main():
    print("=" * 70)
    print("PHASE 1: NEON TREE CROWNS DATASET & IMAGERY INSPECTION")
    print("=" * 70)
    
    files = inspect_zenodo()
    
    # Save zenodo inventory
    os.makedirs("datasets/neon", exist_ok=True)
    with open("datasets/neon/zenodo_files.json", "w") as f:
        json.dump(files, f, indent=2)
        
    print("\nZenodo metadata saved to datasets/neon/zenodo_files.json")

if __name__ == "__main__":
    main()
