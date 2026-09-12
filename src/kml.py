"""
TreeVision AI — KML / AOI Parser & Spatial Clipper
Handles KML ingestion, polygon validation, reprojection to imagery CRS,
intersection checks, and AOI-based spatial filtering.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple, Dict, Any
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import pyproj
from src.geospatial import reproject_geometry

def parse_kml_coordinates(coord_text: str) -> List[Tuple[float, float]]:
    """
    Parses KML coordinate strings: 'lon,lat,alt lon,lat,alt ...'
    Returns list of (lon, lat) tuples.
    """
    coords = []
    tokens = coord_text.strip().split()
    for token in tokens:
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            except ValueError:
                continue
    return coords

def load_kml_polygon(kml_path: str) -> Polygon:
    """
    Loads and parses KML/KMZ files into a unified Shapely Polygon in EPSG:4326.
    Handles multiple polygons via unary union.
    """
    if not os.path.exists(kml_path):
        raise FileNotFoundError(f"KML file not found: {kml_path}")
        
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Invalid KML format: {e}")

    # Namespaces commonly used in KML
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    
    polygons = []
    
    # Search for coordinates under LinearRing / outerBoundaryIs
    for poly_elem in root.iter():
        tag = poly_elem.tag.split("}")[-1] if "}" in poly_elem.tag else poly_elem.tag
        if tag == "Polygon":
            outer_coords = None
            inner_coords_list = []
            
            for child in poly_elem.iter():
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if ctag == "outerBoundaryIs":
                    coord_elem = child.find(".//coordinates") or child.find(".//{http://www.opengis.net/kml/2.2}coordinates")
                    if coord_elem is not None and coord_elem.text:
                        outer_coords = parse_kml_coordinates(coord_elem.text)
                elif ctag == "innerBoundaryIs":
                    coord_elem = child.find(".//coordinates") or child.find(".//{http://www.opengis.net/kml/2.2}coordinates")
                    if coord_elem is not None and coord_elem.text:
                        in_coords = parse_kml_coordinates(coord_elem.text)
                        if len(in_coords) >= 3:
                            inner_coords_list.append(in_coords)
                            
            if outer_coords and len(outer_coords) >= 3:
                poly = Polygon(outer_coords, inner_coords_list)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    polygons.append(poly)
                    
    if not polygons:
        raise ValueError("No valid Polygon geometries found in KML file.")
        
    unified = unary_union(polygons)
    if isinstance(unified, MultiPolygon):
        # Pick largest polygon if multipolygon or return union
        return unified
    return unified

def process_kml_for_imagery(
    kml_path: str,
    image_crs_str: str,
    image_bounds: Any
) -> Dict[str, Any]:
    """
    Validates KML, reprojects to imagery CRS, and computes AOI intersection.
    """
    # 1. Load KML in EPSG:4326 (WGS84 lon/lat)
    kml_geom_wgs84 = load_kml_polygon(kml_path)
    
    # 2. Reproject to image CRS if image is georeferenced
    if image_crs_str:
        try:
            aoi_projected = reproject_geometry(kml_geom_wgs84, "EPSG:4326", image_crs_str)
        except Exception as e:
            raise ValueError(f"Failed to reproject KML from EPSG:4326 to {image_crs_str}: {e}")
    else:
        # Non-georeferenced imagery cannot be spatially matched to real-world KML
        raise ValueError("Cannot apply real-world KML to an unreferenced image lacking CRS metadata.")

    # 3. Create bounding box geometry for imagery
    from shapely.geometry import box as s_box
    img_box = s_box(image_bounds.left, image_bounds.bottom, image_bounds.right, image_bounds.top)
    
    # 4. Check intersection
    if not aoi_projected.intersects(img_box):
        raise ValueError(
            f"KML AOI does not intersect the imagery bounds!\n"
            f"Image Bounds: {image_bounds}\n"
            f"KML AOI Bounds: {aoi_projected.bounds}"
        )
        
    intersection_geom = aoi_projected.intersection(img_box)
    if intersection_geom.is_empty:
        raise ValueError("Intersection between KML AOI and imagery is empty.")
        
    return {
        "aoi_geometry_projected": aoi_projected,
        "intersection_geometry": intersection_geom,
        "aoi_area_m2": intersection_geom.area,
        "kml_original_wgs84": kml_geom_wgs84
    }

def filter_detections_by_aoi(
    detections: List[Dict[str, Any]],
    aoi_geom_projected: Polygon
) -> List[Dict[str, Any]]:
    """
    Filters detections whose projected centroid or bounding box falls inside the AOI polygon.
    """
    filtered = []
    for det in detections:
        # Check if centroid is provided in projected space
        cx = det.get("proj_centroid_x")
        cy = det.get("proj_centroid_y")
        
        if cx is not None and cy is not None:
            pt = Point(cx, cy)
            if aoi_geom_projected.contains(pt):
                filtered.append(det)
        else:
            # Fallback to projected box intersection
            p_box = (
                det.get("proj_left", 0),
                det.get("proj_bottom", 0),
                det.get("proj_right", 0),
                det.get("proj_top", 0)
            )
            from shapely.geometry import box as s_box
            b_geom = s_box(*p_box)
            if aoi_geom_projected.intersects(b_geom):
                filtered.append(det)
                
    return filtered
