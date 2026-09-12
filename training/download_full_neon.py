"""
TreeVision AI — Full NEON Dataset Downloader
Downloads all 38 files (~22.4 GB) from Zenodo Record 3765872 (Weinstein et al., 2020)
Supports:
- Resume interrupted downloads
- Progress bar per file
- Chunked streaming to prevent RAM overflow
- Site-by-site selection
"""

import os
import sys
import json
import time
import requests
from tqdm import tqdm

def download_file(url, target_path, expected_size=None):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Check if already fully downloaded
    if os.path.exists(target_path):
        current_size = os.path.getsize(target_path)
        if expected_size and current_size >= expected_size:
            print(f"  ✓ Already downloaded: {os.path.basename(target_path)} ({current_size / (1024*1024):.1f} MB)")
            return True
        headers = {"Range": f"bytes={current_size}-"}
        mode = "ab"
    else:
        current_size = 0
        headers = {}
        mode = "wb"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        
        if response.status_code == 416: # Range not satisfiable (already complete)
            return True
            
        total_size = int(response.headers.get("content-length", 0)) + current_size
        
        with open(target_path, mode) as f, tqdm(
            desc=os.path.basename(target_path),
            initial=current_size,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024): # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ❌ Error downloading {os.path.basename(target_path)}: {e}")
        return False

def download_neon_dataset(target_dir="datasets/neon/csv", sites=None):
    inventory_file = "datasets/neon/zenodo_files.json"
    if not os.path.exists(inventory_file):
        print("Zenodo inventory missing. Running inspect_dataset.py first...")
        from training.inspect_dataset import inspect_zenodo
        files = inspect_zenodo()
    else:
        with open(inventory_file) as f:
            files = json.load(f)

    print("=" * 65)
    print("🌲 TREEVISION AI — NEON DATASET BULK DOWNLOADER")
    print(f"Target Directory: {os.path.abspath(target_dir)}")
    print(f"Total Available Files: {len(files)}")
    print("=" * 65)

    total_download_mb = sum(f["size_mb"] for f in files)
    print(f"Total Archive Size: {total_download_mb / 1024:.2f} GB\n")

    for idx, item in enumerate(files, 1):
        key = item["key"]
        size_mb = item["size_mb"]
        url = item["download_url"]

        # Filter if specific sites requested
        if sites:
            matched = any(s.upper() in key.upper() for s in sites)
            if not matched:
                continue

        target_file = os.path.join(target_dir, key)
        print(f"[{idx}/{len(files)}] {key} ({size_mb:.1f} MB)...")
        download_file(url, target_file, expected_size=int(size_mb * 1024 * 1024))

    print("\n" + "=" * 65)
    print("✅ Download process completed!")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="datasets/neon/csv")
    parser.add_argument("--sites", nargs="+", help="Specific sites to download e.g. OSBS MLBS HARV")
    args = parser.parse_args()

    download_neon_dataset(target_dir=args.dir, sites=args.sites)
