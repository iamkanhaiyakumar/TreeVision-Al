import os
import sys
import json
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import run_treevision_pipeline
from src.postprocess import compute_box_iou
from training.evaluate import load_ground_truth_boxes, match_predictions_to_ground_truth

def optimize():
    test_img_path = "datasets/neon/processed/images/test/2019_YELL_2_541000_4977000_image_crop.png"
    label_path = "datasets/neon/processed/labels/test/2019_YELL_2_541000_4977000_image_crop.txt"

    img_pil = Image.open(test_img_path).convert("RGB")
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    gt_boxes = load_ground_truth_boxes(label_path, w, h)
    ref_count = len(gt_boxes)

    print(f"Ground Truth Reference Crowns: {ref_count}")
    print("=" * 70)
    print("SEARCHING OPTIMAL CONFIDENCE THRESHOLDS (DeepForest)")
    print("=" * 70)

    deepforest_results = []
    for conf in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        res = run_treevision_pipeline(
            image_input=img_np,
            model_type="deepforest",
            confidence_threshold=conf,
            manual_gsd=0.10
        )
        boxes = [(d["xmin"], d["ymin"], d["xmax"], d["ymax"]) for d in res["detections"]]
        tp, fp, fn = match_predictions_to_ground_truth(boxes, gt_boxes, iou_thresh=0.35)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        count_err = abs(len(boxes) - ref_count) / ref_count * 100
        
        deepforest_results.append({
            "confidence": conf,
            "predictions": len(boxes),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "count_error": round(count_err, 2)
        })
        print(f"Conf: {conf:.2f} | Dets: {len(boxes):3d} | TP: {tp:3d} | FP: {fp:2d} | Precision: {prec*100:5.2f}% | Recall: {rec*100:5.2f}% | F1: {f1*100:5.2f}%")

    print("\n" + "=" * 70)
    print("SEARCHING OPTIMAL CONFIDENCE THRESHOLDS (Custom YOLO)")
    print("=" * 70)
    yolo_results = []
    for conf in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        res = run_treevision_pipeline(
            image_input=img_np,
            model_type="custom",
            confidence_threshold=conf,
            manual_gsd=0.10
        )
        boxes = [(d["xmin"], d["ymin"], d["xmax"], d["ymax"]) for d in res["detections"]]
        tp, fp, fn = match_predictions_to_ground_truth(boxes, gt_boxes, iou_thresh=0.35)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        count_err = abs(len(boxes) - ref_count) / ref_count * 100

        yolo_results.append({
            "confidence": conf,
            "predictions": len(boxes),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "count_error": round(count_err, 2)
        })
        print(f"Conf: {conf:.2f} | Dets: {len(boxes):3d} | TP: {tp:3d} | FP: {fp:2d} | Precision: {prec*100:5.2f}% | Recall: {rec*100:5.2f}% | F1: {f1*100:5.2f}%")

    with open("evaluation/threshold_optimization.json", "w") as f:
        json.dump({"deepforest": deepforest_results, "yolo": yolo_results}, f, indent=2)

if __name__ == "__main__":
    optimize()
