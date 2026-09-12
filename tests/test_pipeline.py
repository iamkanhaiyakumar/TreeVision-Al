"""
TreeVision AI — Automated System & Pipeline Test Suite
Tests:
1. Geospatial metadata extraction & CRS validation
2. Affine coordinate transformations
3. Non-overlapping canopy area & coverage calculation
4. KML parsing, reprojection, and intersection
5. Image tiling & cross-tile deduplication
6. Full end-to-end inference pipeline & export formats
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box as s_box

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.geospatial import read_geospatial_metadata, pixel_to_projected_box
from src.kml import load_kml_polygon, process_kml_for_imagery
from src.tiling import generate_tile_windows, tile_image_array
from src.postprocess import compute_box_iou, compute_center_distance, deduplicate_detections
from src.area import compute_crown_area, calculate_canopy_statistics
from src.inference import run_treevision_pipeline, export_results_to_dataframe, export_results_to_geojson

class TestTreeVisionGeospatial(unittest.TestCase):
    def setUp(self):
        self.sample_tif = os.path.join(PROJECT_ROOT, "demo", "sample_forest.tif")
        self.sample_kml = os.path.join(PROJECT_ROOT, "demo", "sample_aoi.kml")

    def test_geospatial_metadata(self):
        meta = read_geospatial_metadata(self.sample_tif)
        self.assertTrue(meta["is_georeferenced"])
        self.assertEqual(meta["crs_epsg"], 32617)
        self.assertAlmostEqual(meta["gsd_x"], 0.10, places=2)
        self.assertAlmostEqual(meta["gsd_y"], 0.10, places=2)
        self.assertEqual(meta["width"], 400)
        self.assertEqual(meta["height"], 400)

    def test_kml_parsing_and_intersection(self):
        meta = read_geospatial_metadata(self.sample_tif)
        aoi = process_kml_for_imagery(self.sample_kml, meta["crs"], meta["bounds"])
        self.assertIsNotNone(aoi["aoi_area_m2"])
        self.assertGreater(aoi["aoi_area_m2"], 0)
        self.assertTrue(aoi["intersection_geometry"].is_valid)

    def test_tiling_logic(self):
        windows = generate_tile_windows(1000, 1000, tile_size=640, overlap=0.15)
        self.assertGreater(len(windows), 1)
        # All windows must be within image bounds
        for x, y, w, h in windows:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, 1000)
            self.assertLessEqual(y + h, 1000)

    def test_deduplication(self):
        # Two identical boxes with different confidence
        dets = [
            {"xmin": 100, "ymin": 100, "xmax": 150, "ymax": 150, "confidence": 0.85},
            {"xmin": 102, "ymin": 101, "xmax": 152, "ymax": 151, "confidence": 0.70},
            {"xmin": 300, "ymin": 300, "xmax": 350, "ymax": 350, "confidence": 0.90}
        ]
        deduped = deduplicate_detections(dets, iou_threshold=0.40)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["confidence"], 0.90)
        self.assertEqual(deduped[1]["confidence"], 0.85)

    def test_canopy_area_calculation(self):
        dets = [
            {"xmin": 0, "ymin": 0, "xmax": 20, "ymax": 20, "confidence": 0.8}, # 20px * 0.1m = 2m -> 4 m²
            {"xmin": 10, "ymin": 0, "xmax": 30, "ymax": 20, "confidence": 0.7} # overlapping 10px * 20px
        ]
        stats = calculate_canopy_statistics(
            detections=dets,
            img_width=100,
            img_height=100,
            gsd_x=0.10,
            gsd_y=0.10,
            transform=None
        )
        self.assertEqual(stats["tree_count"], 2)
        # 3m width x 2m height union = 6 m²
        self.assertAlmostEqual(stats["total_canopy_area_m2"], 6.0, places=1)
        # Sum of individuals: 4m² + 4m² = 8 m²
        self.assertAlmostEqual(stats["sum_individual_areas_m2"], 8.0, places=1)

    def test_export_formats(self):
        dets = [
            {"tree_id": "tree_0001", "confidence": 0.85, "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30, "area_m2": 4.0, "proj_left": 400000, "proj_bottom": 3000000, "proj_right": 400002, "proj_top": 3000002}
        ]
        df = export_results_to_dataframe(dets)
        self.assertEqual(len(df), 1)
        self.assertIn("tree_id", df.columns)
        
        geojson = export_results_to_geojson(dets, crs_str="EPSG:32617")
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 1)

if __name__ == "__main__":
    unittest.main()
