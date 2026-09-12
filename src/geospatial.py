"""
TreeVision AI — Geospatial Processing Module
Handles GeoTIFF metadata extraction, affine coordinate transformations,
projected CRS validation, and physical area calculations.
"""

import os
from typing import Dict, Any, Tuple, Optional
import rasterio
from rasterio.transform import Affine
from shapely.geometry import box as shapely_box, Polygon
import pyproj
from pyproj import CRS, Transformer

def read_geospatial_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts spatial reference, affine transformation, and resolution from an image.
    Supports GeoTIFF and standard image formats (PNG, JPG).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    meta = {
        "is_georeferenced": False,
        "crs": None,
        "crs_epsg": None,
        "transform": None,
        "gsd_x": None,
        "gsd_y": None,
        "pixel_area_m2": None,
        "bounds": None,
        "width": 0,
        "height": 0,
        "count": 0,
        "nodata": None,
        "driver": "Unknown"
    }

    try:
        with rasterio.open(image_path) as src:
            meta["width"] = src.width
            meta["height"] = src.height
            meta["count"] = src.count
            meta["nodata"] = src.nodata
            meta["driver"] = src.driver

            if src.crs is not None:
                meta["is_georeferenced"] = True
                meta["crs"] = str(src.crs)
                meta["crs_epsg"] = src.crs.to_epsg()
                meta["transform"] = src.transform
                meta["bounds"] = src.bounds
                
                # Ground sample distance (meters per pixel for projected CRS)
                res_x, res_y = src.res
                meta["gsd_x"] = abs(res_x)
                meta["gsd_y"] = abs(res_y)
                
                # Check if projected (in meters) or geographic (degrees)
                is_projected = src.crs.is_projected
                meta["is_projected"] = is_projected
                
                if is_projected:
                    meta["pixel_area_m2"] = meta["gsd_x"] * meta["gsd_y"]
                else:
                    meta["pixel_area_m2"] = None  # Geographic degrees cannot be directly multiplied
    except Exception as e:
        meta["error"] = str(e)

    return meta

def pixel_to_projected_box(
    pixel_box: Tuple[float, float, float, float], 
    transform: Affine
) -> Tuple[float, float, float, float]:
    """
    Converts pixel bounding box (xmin, ymin, xmax, ymax) into
    projected coordinates (left, bottom, right, top).
    """
    px_min, py_min, px_max, py_max = pixel_box
    
    # Top-left and bottom-right in projected space
    left, top = transform * (px_min, py_min)
    right, bottom = transform * (px_max, py_max)
    
    # Ensure correct min/max ordering
    return (min(left, right), min(bottom, top), max(left, right), max(bottom, top))

def projected_to_pixel_box(
    proj_box: Tuple[float, float, float, float],
    transform: Affine
) -> Tuple[float, float, float, float]:
    """
    Converts projected bounding box (left, bottom, right, top) into
    pixel bounding box (xmin, ymin, xmax, ymax).
    """
    left, bottom, right, top = proj_box
    inv_transform = ~transform
    
    px_left, py_top = inv_transform * (left, top)
    px_right, py_bottom = inv_transform * (right, bottom)
    
    return (
        min(px_left, px_right),
        min(py_top, py_bottom),
        max(px_left, px_right),
        max(py_top, py_bottom)
    )

def create_box_polygon(coords: Tuple[float, float, float, float]) -> Polygon:
    """
    Creates a Shapely Polygon from (minx, miny, maxx, maxy).
    """
    minx, miny, maxx, maxy = coords
    return shapely_box(minx, miny, maxx, maxy)

def reproject_geometry(geom: Polygon, src_crs_str: str, dst_crs_str: str) -> Polygon:
    """
    Reprojects a Shapely geometry from source CRS to destination CRS.
    """
    src_crs = CRS.from_user_input(src_crs_str)
    dst_crs = CRS.from_user_input(dst_crs_str)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    from shapely.ops import transform
    return transform(transformer.transform, geom)
