"""
TreeVision AI — Unified Tree Crown Detector Interface
Provides standardized access to both Custom YOLO models and DeepForest baseline.
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
from ultralytics import YOLO
from src.deepforest_baseline import DeepForestBaseline

class CustomYOLODetector:
    def __init__(self, weights_path: str = "models/custom/best.pt", confidence_threshold: float = 0.35):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        # Fallback to pretrained base if custom weights not yet generated
        target_path = self.weights_path
        if not os.path.exists(target_path):
            print(f"Custom model weights not found at '{target_path}', falling back to 'yolov8s.pt'")
            target_path = "yolov8s.pt"
            
        self.model = YOLO(target_path)

    def predict_tile(self, tile_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLO inference on a single RGB tile.
        """
        results = self.model.predict(
            source=tile_rgb,
            conf=self.confidence_threshold,
            verbose=False,
            device="cpu"
        )
        
        detections = []
        if not results:
            return detections
            
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections
            
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        
        for i in range(len(xyxy)):
            detections.append({
                "xmin": float(xyxy[i][0]),
                "ymin": float(xyxy[i][1]),
                "xmax": float(xyxy[i][2]),
                "ymax": float(xyxy[i][3]),
                "confidence": round(float(confs[i]), 4),
                "label": "Tree",
                "model_source": "Custom_YOLO"
            })
            
        return detections

def get_tree_detector(
    model_type: str = "custom",
    confidence_threshold: float = 0.35,
    custom_weights: str = "models/custom/best.pt"
):
    """
    Factory function to instantiate selected detector.
    """
    if model_type.lower() in ["deepforest", "baseline", "deepforest_baseline"]:
        return DeepForestBaseline(confidence_threshold=confidence_threshold)
    else:
        return CustomYOLODetector(weights_path=custom_weights, confidence_threshold=confidence_threshold)
