"""
TreeVision AI — Custom YOLO Training Pipeline (Phase 4)
Trains YOLOv8s on geographically split NEON airborne dataset.
Saves checkpoints, logs experiment metrics, and stores best weights in models/custom/best.pt.
Configurable for local execution and cloud GPU clusters (Lightning AI).
"""

import os
import sys
import json
import time
import argparse
import yaml
from ultralytics import YOLO

def train_model(
    data_yaml: str = "datasets/neon/processed/data.yaml",
    base_model: str = "yolov8s.pt",
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 4,
    lr: float = 0.005,
    device: str = "cpu",
    output_dir: str = "models/custom",
    experiment_dir: str = "experiments"
):
    print("=" * 65)
    print("PHASE 4: TRAINING CUSTOM TREE CROWN DETECTION MODEL")
    print(f"Base Model: {base_model} | Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz}")
    print(f"Device: {device} | Data: {data_yaml}")
    print("=" * 65)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    
    start_time = time.time()
    model = YOLO(base_model)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        device=device,
        project="runs/detect",
        name="neon_crowns",
        exist_ok=True,
        verbose=True,
        workers=0
    )
    
    train_duration = round(time.time() - start_time, 2)
    
    # Save best weights
    best_weights_src = os.path.join("runs", "detect", "neon_crowns", "weights", "best.pt")
    best_weights_dst = os.path.join(output_dir, "best.pt")
    
    if os.path.exists(best_weights_src):
        import shutil
        shutil.copyfile(best_weights_src, best_weights_dst)
        print(f"Best model weights saved to: {best_weights_dst}")
    else:
        # Fallback to last.pt or base if train was 1 epoch
        last_src = os.path.join("runs", "detect", "neon_crowns", "weights", "last.pt")
        if os.path.exists(last_src):
            import shutil
            shutil.copyfile(last_src, best_weights_dst)
            print(f"Saved last weights to: {best_weights_dst}")

    # Validate on val set
    print("\nRunning Validation on Held-Out Validation Set...")
    val_results = model.val(data=data_yaml, split="val", imgsz=imgsz, device=device)
    
    metrics = {
        "model": "YOLOv8s_Custom",
        "base_model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch,
        "learning_rate": lr,
        "device": device,
        "train_time_sec": train_duration,
        "precision": round(float(val_results.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_results.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    }
    
    # Calculate F1
    p = metrics["precision"]
    r = metrics["recall"]
    metrics["f1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

    exp_json = os.path.join(experiment_dir, f"exp_{int(time.time())}.json")
    with open(exp_json, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("\nTraining & Validation Completed Programmatically:")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v}")
    print(f"Experiment log saved to: {exp_json}")
    print("=" * 65)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
