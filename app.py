"""
TreeVision AI — Streamlit Web Application
Production interface for automated tree crown detection, counting, and canopy area estimation.
Designed for intuitive non-technical review, zero-setup demonstration, and defensible geospatial analysis.
"""

import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import (
    run_treevision_pipeline,
    export_results_to_dataframe,
    export_results_to_geojson
)
from src.geospatial import read_geospatial_metadata

# Page configuration
st.set_page_config(
    page_title="TreeVision AI — Tree Crown Detection & Canopy Analysis",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1b4d3e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: #276749;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .limitations-box {
        background-color: #fffaf0;
        border-left: 4px solid #dd6b20;
        padding: 14px;
        border-radius: 4px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">🌲 TreeVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Individual Tree Crown Detection & Canopy Area Analysis from High-Resolution Imagery</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("📁 Input Imagery & Area")

# Preset Sample Data Toggle
use_sample_data = st.sidebar.checkbox("Load Sample NEON Forest GeoTIFF", value=False, help="Loads verified sample GeoTIFF from Ordway-Swisher Biological Station (OSBS_029.tif).")

uploaded_image = None
image_path_to_process = None
temp_image_file = None

if use_sample_data:
    sample_path = os.path.join(PROJECT_ROOT, "demo", "sample_forest.tif")
    if os.path.exists(sample_path):
        image_path_to_process = sample_path
        st.sidebar.success(" Loaded demo/sample_forest.tif (10 cm/px GeoTIFF)")
    else:
        st.sidebar.warning("Sample file not found, please upload an image.")
else:
    uploaded_image = st.sidebar.file_uploader(
        "Upload Forest Image (GeoTIFF, PNG, JPG)",
        type=["tif", "tiff", "png", "jpg", "jpeg"]
    )
    if uploaded_image:
        suffix = os.path.splitext(uploaded_image.name)[1]
        temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_image_file.write(uploaded_image.read())
        temp_image_file.flush()
        image_path_to_process = temp_image_file.name

# KML AOI Upload
st.sidebar.subheader("🗺️ Area of Interest (Optional)")
use_sample_kml = False
if use_sample_data:
    use_sample_kml = st.sidebar.checkbox("Apply Sample KML AOI Boundary", value=False, help="Clips analysis to a sub-region polygon inside the sample GeoTIFF.")

uploaded_kml = None
kml_path_to_process = None
temp_kml_file = None

if use_sample_kml:
    sample_kml = os.path.join(PROJECT_ROOT, "demo", "sample_aoi.kml")
    if os.path.exists(sample_kml):
        kml_path_to_process = sample_kml
        st.sidebar.info("📌 Sample KML boundary active")
else:
    uploaded_kml = st.sidebar.file_uploader("Upload KML AOI File (.kml)", type=["kml"])
    if uploaded_kml:
        temp_kml_file = tempfile.NamedTemporaryFile(delete=False, suffix=".kml")
        temp_kml_file.write(uploaded_kml.read())
        temp_kml_file.flush()
        kml_path_to_process = temp_kml_file.name

# Model & Parameter Configuration
st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.radio(
    "Detection Architecture",
    ["DeepForest Baseline (Recommended)", "Custom YOLOv8s Model"],
    index=0,
    help="DeepForest is pretrained across 21 NEON sites. Custom YOLO demonstrates lightweight architecture trained on NEON."
)
model_key = "deepforest" if "DeepForest" in model_choice else "custom"

confidence_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.15,
    max_value=0.90,
    value=0.40,
    step=0.05,
    help="Minimum model confidence score required to accept a tree crown detection. Recommended: 0.40 (77.2% precision) or 0.55 (85.3% precision)."
)

with st.sidebar.expander("⚙️ Advanced Geospatial Parameters"):
    tile_size = st.select_slider("Tile Processing Window Size", options=[320, 512, 640, 1024], value=640)
    overlap = st.slider("Tile Boundary Overlap", min_value=0.05, max_value=0.30, value=0.15, step=0.05)
    manual_gsd = st.number_input("Manual GSD (meters/pixel) [For non-GeoTIFF]", min_value=0.01, max_value=5.0, value=0.10, step=0.01)
    use_elliptical = st.checkbox("Apply π/4 Elliptical Crown Correction", value=False, help="Applies 0.785 fill factor to approximate elliptical crowns instead of full rectangles.")

analyze_button = st.sidebar.button("🌲 Analyze Forest Imagery", type="primary", use_container_width=True)

# Main Dashboard Execution
if image_path_to_process and analyze_button:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(msg, pct):
        status_text.markdown(f"**Status:** {msg}")
        progress_bar.progress(pct)

    try:
        results = run_treevision_pipeline(
            image_input=image_path_to_process,
            kml_path=kml_path_to_process,
            model_type=model_key,
            confidence_threshold=confidence_thresh,
            tile_size=tile_size,
            overlap=overlap,
            manual_gsd=manual_gsd,
            use_elliptical_factor=use_elliptical,
            progress_callback=update_progress
        )

        status_text.empty()
        progress_bar.empty()

        stats = results["statistics"]
        geo_meta = results["geospatial_metadata"]
        detections = results["detections"]

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Trees Detected</div>
                <div class="metric-value">🌳 {stats['tree_count']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            area_str = f"{stats['total_canopy_area_m2']:,.1f} m²" if stats['total_canopy_area_m2'] is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Canopy Area (Estimated)</div>
                <div class="metric-value">📐 {area_str}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            cov_str = f"{stats['canopy_coverage_pct']:.1f}%" if stats['canopy_coverage_pct'] is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Canopy Coverage</div>
                <div class="metric-value">📊 {cov_str}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">🎯 {stats['avg_confidence_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Image Visualizations
        tab_comp, tab_annotated, tab_original = st.tabs(["Side-by-Side Comparison", "Detected Tree Overlay", "Original Imagery"])
        with tab_comp:
            st.image(results["comparison_image"], use_container_width=True, caption="Visual validation: Original Airborne Raster vs Detected Tree Crowns")
        with tab_annotated:
            st.image(results["annotated_image"], use_container_width=True, caption=f"Identified {stats['tree_count']} individual crowns with bounding boxes and estimated areas.")
        with tab_original:
            st.image(results["original_image"], use_container_width=True, caption="Original input raster prior to computer vision inference.")

        # Geospatial Details
        if geo_meta.get("is_georeferenced"):
            st.info(
                f"🛰️ **Geospatial Reference:** CRS `{geo_meta.get('crs')}` | "
                f"Ground Sample Distance (GSD): `{geo_meta.get('gsd_x'):.2f}m x {geo_meta.get('gsd_y'):.2f}m` per pixel | "
                f"Total Analyzed Footprint: `{stats.get('aoi_area_m2', 0):,.1f} m²` "
                f"({stats.get('aoi_area_m2', 0) / 10000.0:.3f} hectares)"
            )

        # Export Data & Table
        st.subheader("📋 Tree Inventory & Geospatial Exports")
        df_results = export_results_to_dataframe(detections)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Inventory (CSV)",
                data=csv_data,
                file_name="treevision_inventory.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            geojson_dict = export_results_to_geojson(detections, crs_str=geo_meta.get("crs"))
            geojson_data = json.dumps(geojson_dict, indent=2).encode("utf-8")
            st.download_button(
                label="🗺️ Download Footprints (GeoJSON for GIS)",
                data=geojson_data,
                file_name="treevision_crowns.geojson",
                mime="application/geo+json",
                use_container_width=True
            )

        st.dataframe(
            df_results[[c for c in ["tree_id", "confidence", "estimated_crown_area_m2", "crown_width_m", "crown_height_m", "centroid_pixel_x", "centroid_pixel_y"] if c in df_results.columns]],
            use_container_width=True,
            height=250
        )

    except Exception as err:
        st.error(f"Analysis encountered an error: {err}")

elif analyze_button and not image_path_to_process:
    st.warning("⚠️ **No image selected!** Please upload a forest image (GeoTIFF, PNG, JPG) in the sidebar or check **'Load Sample NEON Forest GeoTIFF'**.")

elif not analyze_button:
    if use_sample_data or uploaded_image:
        st.info("👈 Click **'🌲 Analyze Forest Imagery'** in the sidebar to run tree crown detection.")
    else:
        st.info("👈 **Get Started:** Upload your forest image (GeoTIFF, PNG, JPG) in the sidebar, or check **'Load Sample NEON Forest GeoTIFF'** for a quick demonstration, then click **'🌲 Analyze Forest Imagery'**.")
    
    st.markdown("""
    ### How TreeVision AI Works
    1. **High-Resolution Airborne & Satellite Ingestion:** Automatically reads GeoTIFF coordinate reference systems (CRS) and Ground Sample Distance (GSD).
    2. **Boundary / KML Clipping:** Optionally restricts counting to an Area of Interest (AOI) polygon without manual GIS pre-processing.
    3. **Sliding-Window Image Tiling:** Handles gigapixel rasters via overlap tiling without memory exhaustion.
    4. **AI Crown Detection:** Employs peer-reviewed DeepForest baseline and custom YOLO architectures.
    5. **Cross-Tile Deduplication:** Removes split detections across tile seams using geometric IoU and centroid distance.
    6. **Canopy Area & Overlap Resolution:** Calculates canopy footprint in projected meters while resolving overlapping branches via geometric union.
    """)

# Known Limitations Section (Rule 47)
st.markdown("""
<div class="limitations-box">
    <h4>⚠️ Known Limitations & Scientific Disclosures</h4>
    <ul>
        <li><strong>Bounding-Box Area Approximation:</strong> In the absence of polygon segmentation masks, crown areas are estimated from bounding rectangles, which overestimate organic circular crowns by ~20–25%.</li>
        <li><strong>Resolution Dependency:</strong> The detection heads are calibrated for high-resolution imagery (&le; 30 cm/pixel). Results on coarse satellite data (e.g. 10m Sentinel-2 or 3m PlanetScope) cannot resolve discrete stems.</li>
        <li><strong>Dense Closed Canopy:</strong> Interlocking branches in closed deciduous canopies can merge into unified green masses, leading to under-counting.</li>
        <li><strong>Shadow Occlusion:</strong> Deep topographic or cloud shadows can suppress contrast and reduce detection confidence.</li>
        <li><strong>Automated Estimate Disclosure:</strong> These outputs constitute automated remote-sensing estimates and do not replace physical ground-truth forest inventory plots.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
