# 📝 TreeVision AI — Official Submission Form Responses
**Challenge:** Flora Carbon — Build This Weekend. Win a Paid Tech Internship.  
**Deadline:** Monday, 14 September, 11:59 PM IST  
**Candidate:** Kanhaiya Kumar (kanhaiyak0104@gmail.com)

---

### Question 1: What did you build?
*(Guideline: 3–5 sentences describing only implemented functionality)*

> **Answer:**  
> I built **TreeVision AI**, an end-to-end automated tree crown detection and canopy analysis system designed for high-resolution airborne and satellite imagery. The system automatically extracts spatial reference metadata (CRS and GSD) from GeoTIFFs, optionally reprojects and clips user-provided vector KML Area-of-Interest polygons, and sections large rasters using dynamic sliding-window tiling with spatial overlap. It performs individual crown detection using a dual-model architecture (pretrained DeepForest RetinaNet baseline and a custom YOLOv8s detector), removes boundary duplicate detections via IoU and centroid distance Non-Maximum Suppression, and computes non-overlapping canopy surface area and coverage percentages in real-world metric units ($m^2$) using geometric union. All results are delivered through an interactive, stranger-friendly Streamlit dashboard featuring before-and-after visual overlays and one-click exports to tabular CSV and GIS-ready GeoJSON.

---

### Question 2: What doesn't work?
*(Guideline: Tell us honestly about the limitations, failure cases, or shortcuts in your submission)*

> **Answer:**  
> 1. **Under-counting in Closed Canopies:** In dense deciduous forests where touching branches interlock, the 2D bounding-box detector merges adjacent crowns into single continuous detections, leading to systematic under-counting without 3D LiDAR point clouds.  
> 2. **Small Understory Sapling Omission:** Trees with crown diameters under 2.0 meters (<20 pixels at 10 cm/pixel) are frequently omitted due to standard feature pyramid network receptive field thresholds (accounting for 46% of observed false negatives on our held-out Yellowstone test site).  
> 3. **Bounding-Box Area Approximation:** Ground-truth annotations in the NEON dataset are rectangular bounding boxes; consequently, rectangular area estimates overestimate organic circular tree crowns by ~21% in the absence of validated polygon segmentation masks.  
> 4. **Resolution Degradation:** The pipeline cannot reliably resolve discrete stems on coarse satellite imagery (>30 cm/pixel, e.g. 10m Sentinel-2 or 3m PlanetScope).  
> 5. **Prototype Custom Model Generalization:** Our custom YOLOv8s prototype trained on a single local site (OSBS) suffered severe underfitting when transferred zero-shot to Rocky Mountain subalpine forest (4 detections vs. 187 by DeepForest). We therefore honestly deployed DeepForest as our primary production model while documenting the necessity of multi-node cloud GPU cluster training (Lightning AI) for multi-site scaling.

---

### Question 3: What technologies did you use?
*(Guideline: Languages, frameworks, models, APIs, databases, AI coding tools, etc.)*

> **Answer:**  
> - **Programming Language & Environment:** Python 3.11, Windows 11, PowerShell, Isolated Virtual Environment (`.venv`)  
> - **Deep Learning & Computer Vision:** PyTorch, Torchvision, Ultralytics (YOLOv8s), DeepForest 2.1.0 (RetinaNet with ResNet50 backbone), OpenCV, Pillow  
> - **Geospatial & Vector Processing:** Rasterio (GDAL bindings, affine transforms, windowed reads), GeoPandas, Shapely 2.0 (geometric unions & spatial indexing), PyProj (PROJ coordinate transformations), Fiona, XML (KML 2.2 parsing)  
> - **Data Science & Analytics:** NumPy, Pandas, SciPy, Scikit-learn  
> - **User Interface & Visualization:** Streamlit, Matplotlib  
> - **Documentation & PDF Generation:** ReportLab (automated 2-page PDF compilation)  
> - **Cloud & Experiment Tracking:** Lightning AI (cloud GPU training environment), Zenodo REST API (NEON Tree Crowns Dataset DOI: 10.5281/zenodo.3765872)

---

### Question 4: Did you use AI coding tools? If yes, which ones?
*(Guideline: Answer accurately)*

> **Answer:**  
> **Yes.**  
> - Google Antigravity / DeepMind Advanced Agentic Coding Assistant (Gemini 3.8 Flash & Claude 3.7 Sonnet) for rapid scaffold generation, automated geospatial test suite creation, and documentation authoring. All algorithmic decisions, coordinate transforms, and empirical validations were verified against raw dataset artifacts.

---

### Question 5: Anything else you'd like us to know?
*(Guideline: Mention meaningful technical decisions. Never claim something that isn't implemented.)*

> **Answer:**  
> In carbon markets, precision and conservative estimation are far more valuable than fabricated numbers. Three deliberate technical decisions reflect this:  
> 1. **Zero-Leakage Spatial Split:** We strictly partitioned the NEON dataset by geographic biomes (Train: Florida Pine, Val: California Mixed Conifer, Test: Wyoming Subalpine). Testing on unseen biomes provided a true test of generalization.  
> 2. **Canopy Overlap De-biasing:** Rather than simply summing individual crown bounding boxes (which double-counts overlapping foliage), we computed the Shapely geometric union (`unary_union`) of all projected footprints, preventing area inflation.  
> 3. **Complete Stranger-Usability:** The Streamlit application includes pre-loaded sample forest GeoTIFFs and sample KML polygons so any reviewer can verify the complete pipeline end-to-end with a single click without installing GIS software or configuring API keys.
