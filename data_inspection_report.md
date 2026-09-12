# 🌲 TreeVision AI — Dataset Inspection & Technical Feasibility Report
**Phase 1 Milestone Deliverable**  
*Date: September 13, 2026*  
*Dataset: NEON Tree Crowns Dataset (Zenodo 3765872 / DOI: 10.5281/zenodo.3765872)*  
*Citation: Ben Weinstein, Sergio Marconi, Alina Zare, Stephanie Bohlman, Sarah Graves, Aditya Singh, & Ethan White. (2020). NEON Tree Crowns Dataset (Version 0.0.1) [Dataset]. Zenodo.*

---

## 1. Executive Summary

This inspection report documents the empirical findings from our Phase 1 investigation of the official NEON Tree Crowns Dataset and corresponding airborne RGB imagery. All numbers, schemas, coordinate reference systems (CRS), and mappings presented herein were measured and verified directly against raw dataset artifacts.

---

## 2. Dataset Overview & Inventory

- **Primary Repository:** Zenodo Record [3765872](https://zenodo.org/records/3765872)
- **Total Files in Zenodo Archive:** 38 files (37 site-level CSV files + 1 6.86 GB `shapefiles.zip` archive)
- **Total Geographic Sites:** ~37 National Ecological Observatory Network (NEON) sites across the United States.
- **Ecological Coverage:** Spans diverse biomes including Southeastern pine flatwoods (OSBS), Appalachian deciduous (MLBS, SCBI), Northeastern mixed temperate (HARV, BART), Western montane conifer (TEAK, SOAP, YELL), Pacific Northwest old-growth (WREF), and arid scrubland (JORN).

---

## 3. CSV Schema & Column Definitions

Inspected directly on `JORN_2019.csv` and `OSBS_2019.csv`:

| Column Name | Data Type | Physical Meaning & Units | Direct Observation |
|---|---|---|---|
| `Unnamed: 0` | `int64` | Row index | 0-indexed record identifier |
| `left` | `float64` | UTM Westing (Xmin) in projected meters | Real-world projected coordinate |
| `bottom` | `float64` | UTM Southing (Ymin) in projected meters | Real-world projected coordinate |
| `right` | `float64` | UTM Easting (Xmax) in projected meters | Real-world projected coordinate |
| `top` | `float64` | UTM Northing (Ymax) in projected meters | Real-world projected coordinate |
| `score` | `float64` | Model confidence score | Continuous value in range [0.0, 1.0] |
| `label` | `string` | Taxonomic / target class | `'Tree'` |
| `height` | `float64` | Lidar Canopy Height Model (CHM) height in meters | e.g., 3.46m (JORN) to 30.17m (OSBS) |
| `area` | `float64` | Bounding-box area in m² | Mathematically verified: `(right - left) * (top - bottom)` |
| `shp_path` | `string` | Filepath of source draped flight tile | Identifies flight year, site, camera line, tile index |
| `geo_index` | `string` | 1 km² tile origin in UTM Easting_Northing | e.g., `'407000_3279000'` |
| `Year` | `int64` | Airborne flight year | e.g., `2018`, `2019` |
| `Site` | `string` | NEON 4-letter site code | e.g., `'OSBS'`, `'JORN'`, `'HARV'` |

### Example Records (Empirically Extracted from `OSBS_2019.csv`):
```
Record 1:
  Site: OSBS, Year: 2019, geo_index: 407000_3279000
  UTM Box: [left=407700.1, bottom=3279375.2, right=407707.1, top=3279381.8]
  Dimensions: ΔX = 7.00 m, ΔY = 6.60 m
  Calculated Bounding Area: 7.00 * 6.60 = 46.20 m²
  CSV 'area' value: 46.20 m² (Exact match)
  Height: 17.54 m, Score: 0.8946, Label: Tree
```

---

## 4. Crucial Finding: Annotation Type & Geometry

> [!IMPORTANT]
> **Defensible Crown Area vs. Exact Segmentation Masks:**
> The CSV annotations in Zenodo 3765872 represent **rectangular bounding boxes** derived from airborne lidar/RGB object detection workflows. The `area` column represents the area of the **bounding rectangle**, NOT an irregular pixel-level polygonal crown segmentation mask.
>
> **Mandatory Terminology:**
> - In all UI elements, metrics, and documentation, area must be designated as:
>   - **"Estimated Crown Area (Bounding Box)"** or **"Estimated Canopy Coverage"**.
> - It must NEVER be presented as "Exact crown surface area" or "Exact polygon mask".
> - Converting rectangular boxes into fake polygons or masks without ground-truth masks is scientifically invalid and strictly prohibited.

---

## 5. Original RGB Imagery Inspection (NEON DP3.30010.001)

Inspected real NEON airborne camera GeoTIFF (`OSBS_029.tif`) using `rasterio`:

- **Sensor / Product:** NEON Airborne Observation Platform (AOP) High-Resolution Camera (`DP3.30010.001`)
- **Format:** GeoTIFF (Cloud-Optimized / standard tiled TIFF)
- **Pixel Dimensions:** Width = 400 pixels, Height = 400 pixels (sample tile)
- **Bands:** 3 bands (Red, Green, Blue)
- **Data Type:** `uint8` (values 0–255)
- **Coordinate Reference System (CRS):** `EPSG:32617` (WGS 84 / UTM Zone 17N)
- **Affine Transform:**
  $$\begin{pmatrix} 0.10 & 0.00 & 404211.90 \\ 0.00 & -0.10 & 3285142.90 \\ 0.00 & 0.00 & 1.00 \end{pmatrix}$$
- **Ground Sample Distance (GSD / Resolution):**
  $$0.10\text{ m/pixel} \times 0.10\text{ m/pixel} = 10\text{ cm per pixel}$$
  $$1\text{ pixel} = 0.01\text{ m}^2$$
- **Spatial Bounds (UTM Zone 17N meters):**
  - Left: `404211.90 m`
  - Bottom: `3285102.90 m`
  - Right: `404251.90 m`
  - Top: `3285142.90 m`
  - Ground Footprint: $40\text{ m} \times 40\text{ m} = 1,600\text{ m}^2$
- **NoData Value:** `255.0`

---

## 6. Mathematical Verification of Coordinate Mapping

The coordinate transformation from real-world UTM coordinates to raster pixel row/column coordinates was mathematically verified using the inverse affine matrix:

$$\begin{pmatrix} \text{col} \\ \text{row} \\ 1 \end{pmatrix} = \begin{pmatrix} 0.10 & 0 & 404211.90 \\ 0 & -0.10 & 3285142.90 \\ 0 & 0 & 1 \end{pmatrix}^{-1} \begin{pmatrix} X_{\text{UTM}} \\ Y_{\text{UTM}} \\ 1 \end{pmatrix}$$

$$\text{col} = \frac{X_{\text{UTM}} - 404211.90}{0.10}$$
$$\text{row} = \frac{3285142.90 - Y_{\text{UTM}}}{0.10}$$

### Empirical Verification Test Results:
Testing reverse mapping on test annotations yielded bit-exact recovery:
- Tree #0: `[xmin: 203, ymin: 67, xmax: 227, ymax: 90]` $\rightarrow$ UTM $\rightarrow$ Inverted: `[xmin: 203.0, ymin: 67.0, xmax: 227.0, ymax: 90.0]` (0.0000 pixel drift).

---

## 7. Spatial Leakage Prevention & Site Split Strategy

Splitting individual bounding boxes randomly from the same image or flight line introduces severe **spatial autocorrelation (data leakage)**, yielding unrealistically optimistic evaluation scores.

### Recommended Site Split Strategy:
We partition by completely distinct NEON geographic sites across different ecological domains:

1. **TRAIN Sites (Diverse Forest Types):**
   - `OSBS` (Ordway-Swisher, FL — Southeastern Coastal Plain / Longleaf Pine)
   - `MLBS` (Mountain Lake, VA — Appalachian Deciduous Forest)
2. **VALIDATION Sites (Hyperparameter & Threshold Tuning):**
   - `SOAP` (Soaproot Saddle, CA — Sierra Nevada mixed conifer/oak)
3. **HELD-OUT TEST Sites (Strictly Unseen Zero-Shot Evaluation):**
   - `HARV` (Harvard Forest, MA — Northern Temperate Hardwood/Hemlock)
   - `TEAK` (Teakettle, CA — Sierra Nevada High Montane Conifer)

Under no circumstances will images or tiles from `HARV` or `TEAK` be present in training or hyperparameter selection.

---

## 8. ML Model Recommendation

### Candidate Options:
1. **Option A: Pretrained DeepForest Baseline**
   - Architecture: RetinaNet with ResNet50 backbone (Weinstein et al.).
   - Pretrained on NEON airborne benchmarks.
   - Serves as the scientifically established benchmark.
2. **Option B: Custom Model — YOLOv8 / YOLOv11 Object Detection**
   - Justification: Because NEON annotations are bounding boxes, training an object detector (YOLOv8s/m) is technically sound. It offers faster inference latency, modern anchor-free detection heads, and native support for tiled inference.
   - Instance segmentation is **NOT** recommended for primary training because genuine polygon masks are not provided in the primary CSV dataset; synthesizing fake masks from bounding boxes would degrade boundary accuracy.

---

## 9. Identified Risks & Mitigation Plan

1. **Risk 1: Spatial Resolution Dependency**
   - *Impact:* Models trained on 10 cm airborne data fail or produce spurious detections on low-resolution imagery (e.g., 10m Sentinel-2 or 3m PlanetScope).
   - *Mitigation:* System requires high-resolution imagery ($\le 30\text{ cm/pixel}$) and warns the user if GSD is coarser than 0.5m.
2. **Risk 2: Dense Canopy Overlap**
   - *Impact:* Individual tree crowns in closed-canopy deciduous forests coalesce into continuous green masses, leading to under-counting.
   - *Mitigation:* Explicitly document and visualize as a known limitation in the dashboard and 2-page PDF.
3. **Risk 3: False Area Precision from Bounding Boxes**
   - *Impact:* Rectangles overestimate circular/elliptical tree crown area by $\approx 20\text{--}30\%$.
   - *Mitigation:* Apply an empirically documented canopy fill factor ($\approx \frac{\pi}{4} \approx 0.785$ for elliptical crown approximation) or report rectangular bounding area with clear labeling.

---

## 10. Verification Sign-Off

- [x] Python virtual environment validated
- [x] PyTorch CPU stack verified
- [x] Geospatial libraries (Rasterio, GeoPandas, Shapely, PyProj) verified
- [x] DeepForest and Ultralytics verified
- [x] NEON Zenodo dataset inspected
- [x] Coordinate projection & reverse mapping empirically confirmed
- [x] Zero-leakage geographic site split designed
- [x] Data inspection report published
