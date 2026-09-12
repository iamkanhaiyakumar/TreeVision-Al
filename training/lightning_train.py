"""
TreeVision AI — Complete Lightning AI Cloud GPU Training Script
Run this script inside a Lightning AI Studio (with T4 / A10G GPU).

This script:
1. Verifies GPU & CUDA acceleration
2. Downloads NEON Tree Crowns benchmark data (Weinstein et al., 2020)
3. Formats annotations into YOLO format (train, val, test with zero-leakage site split)
4. Trains YOLOv8s with GPU acceleration, fp16 mixed precision, and multi-worker loading
5. Evaluates on held-out test set
6. Saves best weights to models/custom/best.pt
"""

import os
import sys
import time
import json
import shutil
import urllib.request
import yaml
import torch
from PIL import Image

def check_gpu():
    print("=" * 65)
    print("⚡ LIGHTNING AI GPU VERIFICATION")
    print("=" * 65)
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"GPU Device:      {device_name} (Count: {device_count})")
        print(f"VRAM Allocated:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        return "0"
    else:
        print("⚠️ Warning: No GPU detected! Running on CPU fallback.")
        return "cpu"

def prepare_lightning_dataset(base_dir="dataset_lightning"):
    print("\n" + "=" * 65)
    print("📦 PREPARING NEON BENCHMARK DATASET")
    print("=" * 65)
    
    os.makedirs(base_dir, exist_ok=True)
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    # Use DeepForest's built-in curated NEON benchmark sites
    try:
        import deepforest
        from deepforest import get_data
        data_dir = os.path.dirname(get_data("OSBS_029.tif"))
    except ImportError:
        print("Installing deepforest in environment...")
        os.system("pip install deepforest ultralytics rasterio shapely")
        import deepforest
        from deepforest import get_data
        data_dir = os.path.dirname(get_data("OSBS_029.tif"))

    import pandas as pd
    import xml.etree.ElementTree as ET

    # Strict Zero-Leakage Site Split across ecological biomes
    site_mapping = {
        "train": [
            ("OSBS_029.png", "OSBS_029.csv")   # Florida Longleaf Pine
        ],
        "val": [
            ("SOAP_061.png", "SOAP_061.xml")   # California Mixed Conifer
        ],
        "test": [
            ("2019_YELL_2_541000_4977000_image_crop.png", "2019_YELL_2_541000_4977000_image_crop.xml")  # Rocky Mountain Subalpine
        ]
    }

    def convert_box(xmin, ymin, xmax, ymax, img_w, img_h):
        xmin = max(0.0, min(float(img_w), xmin))
        ymin = max(0.0, min(float(img_h), ymin))
        xmax = max(0.0, min(float(img_w), xmax))
        ymax = max(0.0, min(float(img_h), ymax))
        bw = xmax - xmin
        bh = ymax - ymin
        if bw < 2.0 or bh < 2.0:
            return None
        xc = (xmin + xmax) / (2.0 * img_w)
        yc = (ymin + ymax) / (2.0 * img_h)
        nw = bw / float(img_w)
        nh = bh / float(img_h)
        return 0, round(xc, 6), round(yc, 6), round(nw, 6), round(nh, 6)

    stats = {}
    for split, files in site_mapping.items():
        stats[split] = {"images": 0, "trees": 0}
        for img_name, annot_name in files:
            img_src = os.path.join(data_dir, img_name)
            annot_src = os.path.join(data_dir, annot_name)
            if not os.path.exists(img_src) or not os.path.exists(annot_src):
                continue

            img = Image.open(img_src).convert("RGB")
            w, h = img.size
            img_dst = os.path.join(images_dir, split, img_name)
            img.save(img_dst)

            boxes = []
            if annot_name.endswith(".csv"):
                df = pd.read_csv(annot_src)
                for _, r in df.iterrows():
                    boxes.append((float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])))
            elif annot_name.endswith(".xml"):
                tree = ET.parse(annot_src)
                for obj in tree.findall(".//object"):
                    bnd = obj.find("bndbox")
                    if bnd is not None:
                        boxes.append((
                            float(bnd.find("xmin").text),
                            float(bnd.find("ymin").text),
                            float(bnd.find("xmax").text),
                            float(bnd.find("ymax").text)
                        ))

            yolo_lines = []
            for b in boxes:
                res = convert_box(b[0], b[1], b[2], b[3], w, h)
                if res:
                    yolo_lines.append(f"{res[0]} {res[1]} {res[2]} {res[3]} {res[4]}\n")

            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_dst = os.path.join(labels_dir, split, label_name)
            with open(label_dst, "w") as f:
                f.writelines(yolo_lines)

            stats[split]["images"] += 1
            stats[split]["trees"] += len(yolo_lines)

    data_yaml_path = os.path.join(base_dir, "data.yaml")
    yaml_dict = {
        "path": os.path.abspath(base_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "Tree"},
        "nc": 1
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    print(f"Dataset prepared successfully:")
    for s, st in stats.items():
        print(f"  - {s.upper():<6}: {st['images']} images, {st['trees']} crown annotations")
    print(f"Config YAML: {data_yaml_path}")
    return data_yaml_path

def train_on_lightning(data_yaml, device, epochs=30, batch_size=8, imgsz=640, lr=0.005):
    print("\n" + "=" * 65)
    print("🚀 LAUNCHING CLOUD GPU MODEL TRAINING")
    print(f"Epochs: {epochs} | Batch: {batch_size} | Imgsz: {imgsz} | Device: {device}")
    print("=" * 65)

    from ultralytics import YOLO

    # Start from pretrained YOLOv8s (anchor-free detection head)
    model = YOLO("yolov8s.pt")

    start_time = time.time()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr,
        device=device,
        project="lightning_runs",
        name="treevision_yolo",
        exist_ok=True,
        workers=4 if device != "cpu" else 0,
        amp=(device != "cpu"),  # Automatic Mixed Precision for 2x faster GPU training
        verbose=True
    )
    total_time = round(time.time() - start_time, 2)

    # Validate on held-out test set
    print("\n" + "=" * 65)
    print("🔬 EVALUATING ON ZERO-LEAKAGE HELD-OUT TEST SET (Yellowstone Biome)")
    print("=" * 65)
    val_res = model.val(data=data_yaml, split="test", imgsz=imgsz, device=device)

    metrics = {
        "model": "TreeVision_YOLOv8s_LightningAI",
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "device": device,
        "total_train_time_sec": total_time,
        "precision": round(float(val_res.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_res.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(val_res.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    }
    p = metrics["precision"]
    r = metrics["recall"]
    metrics["f1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

    print("\nFinal Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")

    # Save export weights
    os.makedirs("models/custom", exist_ok=True)
    best_src = os.path.join("lightning_runs", "treevision_yolo", "weights", "best.pt")
    best_dst = os.path.join("models", "custom", "best.pt")
    if os.path.exists(best_src):
        shutil.copyfile(best_src, best_dst)
        print(f"\n🎉 Best trained model saved to: {best_dst}")
    
    with open("lightning_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 65)
    print("✅ Training complete! Download 'models/custom/best.pt' for deployment.")
    print("=" * 65)

def main():
    device = check_gpu()
    data_yaml = prepare_lightning_dataset()
    # 30 epochs on GPU takes ~1-2 minutes on T4/A10G
    epochs = 30 if device != "cpu" else 5
    batch = 8 if device != "cpu" else 2
    train_on_lightning(data_yaml=data_yaml, device=device, epochs=epochs, batch_size=batch)

if __name__ == "__main__":
    main()
