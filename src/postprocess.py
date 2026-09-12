"""
TreeVision AI — Postprocessing & Cross-Tile Deduplication
Handles projection of tile detections into global image coordinates,
cross-tile duplicate removal via Non-Maximum Suppression (IoU + Center Distance),
and unique Tree ID assignment.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def compute_box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes:
    (xmin, ymin, xmax, ymax).
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def compute_center_distance(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Computes Euclidean distance between the centers of two bounding boxes.
    """
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

def map_tile_detection_to_global(
    detection: Dict[str, Any],
    col_offset: int,
    row_offset: int
) -> Dict[str, Any]:
    """
    Shifts a tile-local bounding box by (col_offset, row_offset)
    into global image pixel coordinates.
    """
    det = dict(detection)
    det["xmin"] = det["xmin"] + col_offset
    det["ymin"] = det["ymin"] + row_offset
    det["xmax"] = det["xmax"] + col_offset
    det["ymax"] = det["ymax"] + row_offset
    det["centroid_x"] = (det["xmin"] + det["xmax"]) / 2.0
    det["centroid_y"] = (det["ymin"] + det["ymax"]) / 2.0
    return det

def deduplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.40,
    center_dist_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Removes duplicate detections appearing across overlapping tile boundaries.
    Iterates in order of descending confidence.
    """
    if not detections:
        return []

    # Sort descending by confidence score
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    
    keep = []
    
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        box_best = (best["xmin"], best["ymin"], best["xmax"], best["ymax"])
        
        remaining = []
        for candidate in sorted_dets:
            box_cand = (candidate["xmin"], candidate["ymin"], candidate["xmax"], candidate["ymax"])
            iou = compute_box_iou(box_best, box_cand)
            
            # Check center distance condition
            is_duplicate = False
            if iou >= iou_threshold:
                is_duplicate = True
            elif center_dist_threshold is not None:
                dist = compute_center_distance(box_best, box_cand)
                # If centers are extremely close and non-trivial IoU
                if dist <= center_dist_threshold and iou > 0.20:
                    is_duplicate = True

            if not is_duplicate:
                remaining.append(candidate)
                
        sorted_dets = remaining

    # Assign sequential Tree IDs
    for idx, det in enumerate(keep, start=1):
        det["tree_id"] = f"tree_{idx:04d}"

    return keep
