"""
TreeVision AI — DeepForest Baseline Model Wrapper
Integrates pretrained DeepForest (Weinstein et al., 2020) RetinaNet model
for establishing scientific benchmark performance.
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
import pandas as pd
from deepforest import main as df_main

class DeepForestBaseline:
    def __init__(self, confidence_threshold: float = 0.30):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Initializes pretrained DeepForest release weights.
        """
        self.model = df_main.deepforest()
        self.model.load_model()
        # Set config score threshold
        self.model.config["score_thresh"] = self.confidence_threshold

    def predict_tile(self, tile_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs tree crown detection on a single RGB tile (H, W, 3) uint8.
        Returns standardized list of detections.
        """
        # DeepForest expects float/uint8 RGB image
        try:
            # DeepForest 2.x accepts float32 RGB array
            img_float = tile_rgb.astype("float32")
            df_preds = self.model.predict_image(image=img_float)
        except Exception as e:
            print(f"DeepForest predict_tile error: {e}")
            return []

        if df_preds is None or df_preds.empty:
            return []

        detections = []
        for _, row in df_preds.iterrows():
            score = float(row.get("score", 0.0))
            if score >= self.confidence_threshold:
                detections.append({
                    "xmin": float(row["xmin"]),
                    "ymin": float(row["ymin"]),
                    "xmax": float(row["xmax"]),
                    "ymax": float(row["ymax"]),
                    "confidence": round(score, 4),
                    "label": str(row.get("label", "Tree")),
                    "model_source": "DeepForest_Baseline"
                })

        return detections
