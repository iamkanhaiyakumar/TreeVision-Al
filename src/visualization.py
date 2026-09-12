"""
TreeVision AI — Computer Vision Visualization Module
Renders bounding boxes, crown overlay masks, confidence labels,
and side-by-side comparison graphics.
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

def draw_detections(
    image_rgb: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_confidence: bool = True,
    box_color: Tuple[int, int, int] = (0, 230, 115),  # High-visibility vibrant green
    overlay_alpha: float = 0.25
) -> np.ndarray:
    """
    Overlays detected tree crown bounding boxes and translucent canopy fills on an RGB image.
    """
    vis_img = image_rgb.copy()
    overlay = image_rgb.copy()
    
    h, w = vis_img.shape[:2]

    # 1. First draw translucent canopy fill for all trees
    for det in detections:
        xmin = max(0, int(round(det["xmin"])))
        ymin = max(0, int(round(det["ymin"])))
        xmax = min(w - 1, int(round(det["xmax"])))
        ymax = min(h - 1, int(round(det["ymax"])))
        
        # Draw filled ellipse or rectangle representing crown canopy
        center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        axes = (max(1, (xmax - xmin) // 2), max(1, (ymax - ymin) // 2))
        cv2.ellipse(overlay, center, axes, 0, 0, 360, box_color, -1)

    # Blend overlay
    cv2.addWeighted(overlay, overlay_alpha, vis_img, 1.0 - overlay_alpha, 0, vis_img)

    # 2. Draw crisp bounding boxes and concise labels
    for det in detections:
        xmin = max(0, int(round(det["xmin"])))
        ymin = max(0, int(round(det["ymin"])))
        xmax = min(w - 1, int(round(det["xmax"])))
        ymax = min(h - 1, int(round(det["ymax"])))
        
        # Bounding box
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), box_color, 2)

        # Label text
        label_parts = []
        if show_labels and "tree_id" in det:
            label_parts.append(det["tree_id"])
        if show_confidence and "confidence" in det:
            label_parts.append(f"{det['confidence']*100:.0f}%")
        if "area_m2" in det and det["area_m2"] is not None:
            label_parts.append(f"{det['area_m2']:.0f}m²")

        if label_parts:
            label = " | ".join(label_parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position above box or inside if near top border
            text_y = max(ymin - 4, text_h + 4)
            text_x = max(0, xmin)
            
            # Background badge
            cv2.rectangle(
                vis_img,
                (text_x, text_y - text_h - 2),
                (text_x + text_w + 4, text_y + baseline - 2),
                (20, 20, 20),
                -1
            )
            # Text
            cv2.putText(
                vis_img,
                label,
                (text_x + 2, text_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    return vis_img

def create_side_by_side_comparison(
    original_rgb: np.ndarray,
    annotated_rgb: np.ndarray
) -> np.ndarray:
    """
    Creates a side-by-side comparison array (Original vs Detected Trees) with header labels.
    """
    h, w = original_rgb.shape[:2]
    
    # Header height
    header_h = 40
    combined = np.zeros((h + header_h, w * 2, 3), dtype=np.uint8)
    
    # Headers
    cv2.rectangle(combined, (0, 0), (w * 2, header_h), (30, 30, 30), -1)
    cv2.putText(
        combined, "ORIGINAL AIRBORNE / SATELLITE IMAGERY",
        (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA
    )
    cv2.putText(
        combined, "TREEVISION AI — DETECTED CROWNS & CANOPY",
        (w + 20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 115), 2, cv2.LINE_AA
    )
    
    # Place images
    combined[header_h : header_h + h, 0 : w] = original_rgb
    combined[header_h : header_h + h, w : w * 2] = annotated_rgb
    
    return combined
