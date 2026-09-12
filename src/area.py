"""
TreeVision AI — Canopy Area & Geospatial Coverage Module
Calculates individual crown area estimates, resolves canopy overlap via
Shapely geometry union, and computes canopy coverage percentage.
Strictly distinguishes between rectangular bounding-box area and true canopy geometry.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from rasterio.transform import Affine

def compute_crown_area(
    bbox_pixel: Tuple[float, float, float, float],
    gsd_x: float,
    gsd_y: float,
    use_elliptical_factor: bool = False
) -> Dict[str, float]:
    """
    Estimates single tree crown area from pixel dimensions and Ground Sample Distance (GSD).
    If use_elliptical_factor is True, applies π/4 (~0.785) elliptical crown approximation.
    """
    xmin, ymin, xmax, ymax = bbox_pixel
    w_px = max(0.0, xmax - xmin)
    h_px = max(0.0, ymax - ymin)
    
    w_m = w_px * gsd_x
    h_m = h_px * gsd_y
    
    bbox_area_m2 = w_m * h_m
    fill_factor = (np.pi / 4.0) if use_elliptical_factor else 1.0
    estimated_area_m2 = bbox_area_m2 * fill_factor
    
    return {
        "width_m": round(w_m, 2),
        "height_m": round(h_m, 2),
        "bbox_area_m2": round(bbox_area_m2, 2),
        "estimated_crown_area_m2": round(estimated_area_m2, 2)
    }

def calculate_canopy_statistics(
    detections: List[Dict[str, Any]],
    img_width: int,
    img_height: int,
    gsd_x: Optional[float] = None,
    gsd_y: Optional[float] = None,
    transform: Optional[Affine] = None,
    aoi_area_m2: Optional[float] = None,
    use_elliptical_factor: bool = False
) -> Dict[str, Any]:
    """
    Aggregates tree count, individual crown areas, total non-overlapping canopy area,
    and canopy coverage percentage across the image or AOI.
    """
    tree_count = len(detections)
    
    stats = {
        "tree_count": tree_count,
        "is_georeferenced": False,
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "total_canopy_area_m2": None,
        "sum_individual_areas_m2": None,
        "aoi_area_m2": None,
        "canopy_coverage_pct": None,
        "mean_crown_area_m2": None,
        "median_crown_area_m2": None,
        "area_status_message": ""
    }

    if tree_count == 0:
        stats["area_status_message"] = "No trees detected in the specified area."
        return stats

    # Check if spatial resolution is available
    if gsd_x is None or gsd_y is None:
        stats["area_status_message"] = (
            "Absolute canopy area cannot be reliably calculated because spatial "
            "resolution metadata (GSD) is unavailable for this image format."
        )
        return stats

    stats["is_georeferenced"] = True
    
    # 1. Calculate individual crown areas
    individual_areas = []
    projected_boxes = []

    for det in detections:
        p_box = (det["xmin"], det["ymin"], det["xmax"], det["ymax"])
        area_info = compute_crown_area(p_box, gsd_x, gsd_y, use_elliptical_factor)
        
        det["width_m"] = area_info["width_m"]
        det["height_m"] = area_info["height_m"]
        det["area_m2"] = area_info["estimated_crown_area_m2"]
        individual_areas.append(det["area_m2"])

        # Map to projected coordinates if transform is present
        if transform is not None:
            left, top = transform * (det["xmin"], det["ymin"])
            right, bottom = transform * (det["xmax"], det["ymax"])
            
            proj_left = min(left, right)
            proj_bottom = min(bottom, top)
            proj_right = max(left, right)
            proj_top = max(bottom, top)
            
            det["proj_left"] = proj_left
            det["proj_bottom"] = proj_bottom
            det["proj_right"] = proj_right
            det["proj_top"] = proj_top
            det["proj_centroid_x"] = (proj_left + proj_right) / 2.0
            det["proj_centroid_y"] = (proj_bottom + proj_top) / 2.0
            
            projected_boxes.append(shapely_box(proj_left, proj_bottom, proj_right, proj_top))
        else:
            # Fallback to local metric boxes (meter-based from pixel 0,0)
            m_left = det["xmin"] * gsd_x
            m_top = det["ymin"] * gsd_y
            m_right = det["xmax"] * gsd_x
            m_bottom = det["ymax"] * gsd_y
            projected_boxes.append(shapely_box(m_left, m_top, m_right, m_bottom))

    # 2. Compute non-overlapping total canopy area via geometric union
    # Prevents double-counting overlapping crowns (Rule 35)
    union_geometry = unary_union(projected_boxes)
    total_non_overlapping_area_m2 = union_geometry.area
    
    if use_elliptical_factor:
        total_non_overlapping_area_m2 *= (np.pi / 4.0)

    stats["total_canopy_area_m2"] = round(total_non_overlapping_area_m2, 2)
    stats["sum_individual_areas_m2"] = round(float(np.sum(individual_areas)), 2)
    stats["mean_crown_area_m2"] = round(float(np.mean(individual_areas)), 2)
    stats["median_crown_area_m2"] = round(float(np.median(individual_areas)), 2)

    # 3. Compute AOI area and coverage %
    if aoi_area_m2 is not None and aoi_area_m2 > 0:
        total_aoi = aoi_area_m2
    else:
        # Total image footprint area in m²
        total_aoi = (img_width * gsd_x) * (img_height * gsd_y)
        
    stats["aoi_area_m2"] = round(total_aoi, 2)
    
    if total_aoi > 0:
        coverage_pct = min(100.0, (total_non_overlapping_area_m2 / total_aoi) * 100.0)
        stats["canopy_coverage_pct"] = round(coverage_pct, 2)

    stats["area_status_message"] = (
        f"Computed over {tree_count} detected crowns using {gsd_x*100:.1f} cm/px resolution. "
        f"Overlapping crowns resolved via geometric union."
    )

    return stats
