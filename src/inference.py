"""
TreeVision AI — End-to-End Production Inference Pipeline
Orchestrates:
Image Ingestion -> Geospatial Validation -> KML AOI Processing ->
Image Tiling -> Tree Crown Detection -> Global Mapping ->
Cross-Tile Deduplication -> Crown Geometry & Area Calculation ->
Canopy Coverage Aggregation -> Visualization -> CSV / GeoJSON Export
"""

import os
import json
import time
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from PIL import Image
import rasterio

from src.geospatial import read_geospatial_metadata, pixel_to_projected_box
from src.kml import process_kml_for_imagery, filter_detections_by_aoi
from src.tiling import tile_image_array
from src.postprocess import map_tile_detection_to_global, deduplicate_detections
from src.area import calculate_canopy_statistics
from src.detector import get_tree_detector
from src.visualization import draw_detections, create_side_by_side_comparison

def run_treevision_pipeline(
    image_input: Union[str, np.ndarray],
    kml_path: Optional[str] = None,
    model_type: str = "custom",
    confidence_threshold: float = 0.35,
    custom_weights_path: str = "models/custom/best.pt",
    tile_size: int = 640,
    overlap: float = 0.15,
    cross_tile_iou_threshold: float = 0.40,
    manual_gsd: Optional[float] = None,
    use_elliptical_factor: bool = False,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Executes the complete TreeVision AI analytical pipeline.
    """
    start_time = time.time()
    
    def log_progress(step_name: str, pct: int):
        if progress_callback:
            progress_callback(step_name, pct)
        else:
            print(f"[{pct}%] {step_name}")

    # 1. Image Ingestion & Geospatial Metadata
    log_progress("Loading image and reading geospatial metadata...", 10)
    
    image_path = None
    if isinstance(image_input, str):
        image_path = image_input
        geo_meta = read_geospatial_metadata(image_path)
        
        # Load image array for inference
        try:
            with rasterio.open(image_path) as src:
                # Read 3 bands RGB
                if src.count >= 3:
                    img_data = src.read([1, 2, 3])
                else:
                    img_data = np.repeat(src.read([1]), 3, axis=0)
                image_rgb = np.transpose(img_data, (1, 2, 0))
        except Exception:
            # Fallback to standard PIL
            pil_img = Image.open(image_path).convert("RGB")
            image_rgb = np.array(pil_img)
    else:
        # Array passed directly
        image_rgb = image_input
        geo_meta = {
            "is_georeferenced": False,
            "crs": None,
            "transform": None,
            "gsd_x": None,
            "gsd_y": None,
            "bounds": None
        }

    img_h, img_w = image_rgb.shape[:2]

    # Handle manual GSD override if metadata missing
    gsd_x = geo_meta.get("gsd_x")
    gsd_y = geo_meta.get("gsd_y")
    if manual_gsd is not None and manual_gsd > 0:
        gsd_x = manual_gsd
        gsd_y = manual_gsd

    # 2. Optional KML AOI Processing
    log_progress("Validating AOI geometry and checking bounds...", 25)
    aoi_data = None
    aoi_area_m2 = None
    if kml_path and os.path.exists(kml_path):
        if geo_meta.get("is_georeferenced"):
            aoi_data = process_kml_for_imagery(
                kml_path,
                image_crs_str=geo_meta["crs"],
                image_bounds=geo_meta["bounds"]
            )
            aoi_area_m2 = aoi_data["aoi_area_m2"]
        else:
            raise ValueError(
                "KML AOI boundary cannot be applied to an unreferenced image lacking CRS metadata."
            )

    # 3. Image Tiling
    log_progress("Generating sliding-window tiles...", 40)
    tiles = tile_image_array(
        image_rgb=image_rgb,
        tile_size=tile_size,
        overlap=overlap,
        base_transform=geo_meta.get("transform"),
        crs=geo_meta.get("crs")
    )

    # 4. Model Inference across Tiles
    log_progress(f"Running tree crown detection ({model_type})...", 55)
    detector = get_tree_detector(
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        custom_weights=custom_weights_path
    )

    all_raw_detections = []
    for tile in tiles:
        tile_dets = detector.predict_tile(tile.image)
        for d in tile_dets:
            # Map local tile coordinates to global image coordinates
            global_d = map_tile_detection_to_global(d, tile.col_offset, tile.row_offset)
            # Clip bounds to image borders
            global_d["xmin"] = max(0.0, min(float(img_w - 1), global_d["xmin"]))
            global_d["ymin"] = max(0.0, min(float(img_h - 1), global_d["ymin"]))
            global_d["xmax"] = max(0.0, min(float(img_w - 1), global_d["xmax"]))
            global_d["ymax"] = max(0.0, min(float(img_h - 1), global_d["ymax"]))
            
            # Reject zero-area or invalid boxes
            if global_d["xmax"] > global_d["xmin"] and global_d["ymax"] > global_d["ymin"]:
                all_raw_detections.append(global_d)

    # 5. Cross-Tile Duplicate Removal
    log_progress("Merging tile detections and eliminating duplicates...", 70)
    deduped_detections = deduplicate_detections(
        all_raw_detections,
        iou_threshold=cross_tile_iou_threshold,
        center_dist_threshold=15.0
    )

    # 6. Apply KML AOI Filtering if present
    if aoi_data is not None:
        log_progress("Filtering crowns within KML Area of Interest...", 80)
        # Compute projected coordinates before filtering
        transform = geo_meta.get("transform")
        if transform:
            for det in deduped_detections:
                l, t = transform * (det["xmin"], det["ymin"])
                r, b = transform * (det["xmax"], det["ymax"])
                det["proj_left"] = min(l, r)
                det["proj_bottom"] = min(b, t)
                det["proj_right"] = max(l, r)
                det["proj_top"] = max(b, t)
                det["proj_centroid_x"] = (det["proj_left"] + det["proj_right"]) / 2.0
                det["proj_centroid_y"] = (det["proj_bottom"] + det["proj_top"]) / 2.0
                
        deduped_detections = filter_detections_by_aoi(
            deduped_detections,
            aoi_data["intersection_geometry"]
        )

    # 7. Canopy Area & Statistics Calculation
    log_progress("Calculating canopy area and forest metrics...", 88)
    statistics = calculate_canopy_statistics(
        detections=deduped_detections,
        img_width=img_w,
        img_height=img_h,
        gsd_x=gsd_x,
        gsd_y=gsd_y,
        transform=geo_meta.get("transform"),
        aoi_area_m2=aoi_area_m2,
        use_elliptical_factor=use_elliptical_factor
    )

    # 8. Visualizations
    log_progress("Generating visual overlays and reports...", 95)
    annotated_image = draw_detections(image_rgb, deduped_detections)
    comparison_image = create_side_by_side_comparison(image_rgb, annotated_image)

    # Calculate average confidence
    confs = [d["confidence"] for d in deduped_detections]
    statistics["avg_confidence_pct"] = round(float(np.mean(confs) * 100.0), 1) if confs else 0.0
    statistics["inference_time_sec"] = round(time.time() - start_time, 2)
    statistics["model_used"] = model_type

    log_progress("Analysis complete!", 100)

    return {
        "detections": deduped_detections,
        "statistics": statistics,
        "geospatial_metadata": geo_meta,
        "annotated_image": annotated_image,
        "comparison_image": comparison_image,
        "original_image": image_rgb
    }

def export_results_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Exports tree-level detections to a clean Pandas DataFrame.
    """
    rows = []
    for d in detections:
        row = {
            "tree_id": d.get("tree_id"),
            "confidence": d.get("confidence"),
            "pixel_xmin": round(d.get("xmin", 0), 1),
            "pixel_ymin": round(d.get("ymin", 0), 1),
            "pixel_xmax": round(d.get("xmax", 0), 1),
            "pixel_ymax": round(d.get("ymax", 0), 1),
            "centroid_pixel_x": round(d.get("centroid_x", 0), 1),
            "centroid_pixel_y": round(d.get("centroid_y", 0), 1),
            "estimated_crown_area_m2": d.get("area_m2"),
            "crown_width_m": d.get("width_m"),
            "crown_height_m": d.get("height_m"),
            "projected_easting_x": round(d.get("proj_centroid_x", 0), 2) if d.get("proj_centroid_x") else None,
            "projected_northing_y": round(d.get("proj_centroid_y", 0), 2) if d.get("proj_centroid_y") else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def export_results_to_geojson(
    detections: List[Dict[str, Any]], 
    crs_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exports georeferenced crown footprints as a GeoJSON FeatureCollection.
    """
    features = []
    for d in detections:
        # Check if projected geometry is available
        if "proj_left" in d:
            minx, miny, maxx, maxy = d["proj_left"], d["proj_bottom"], d["proj_right"], d["proj_top"]
            coordinates = [[
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny]
            ]]
            feat = {
                "type": "Feature",
                "properties": {
                    "tree_id": d.get("tree_id"),
                    "confidence": d.get("confidence"),
                    "estimated_area_m2": d.get("area_m2"),
                    "crown_width_m": d.get("width_m"),
                    "crown_height_m": d.get("height_m")
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                }
            }
            features.append(feat)

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": crs_str if crs_str else "EPSG:4326"}
        },
        "features": features
    }
    return geojson
