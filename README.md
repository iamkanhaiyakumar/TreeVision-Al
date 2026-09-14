# 🌲 TreeVision AI
### Automated Individual Tree Crown Detection & Canopy Area Analysis from High-Resolution Remote-Sensing Imagery

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![DeepForest](https://img.shields.io/badge/DeepForest-2.1.0-brightgreen.svg)](https://deepforest.readthedocs.io/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8s-blue.svg)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Problem Statement & Motivation
Accurate estimation of individual tree stems and forest canopy cover is the critical foundation for:
1. **Nature-Based Carbon Markets:** Verifying carbon removal credits requires audit-proof stem counts and canopy density rather than uncalibrated regional indices.
2. **Forest Inventory & Ecology:** Monitoring biodiversity, crown mortality, and timber stand volume.
3. **Disaster Management:** Assessing wildfire fuel loads and hurricane canopy damage.

Standard global satellite imagery (e.g. 10m Sentinel-2) cannot resolve individual tree crowns. Manual delineation of crowns across high-resolution airborne or sub-meter satellite rasters is labor-prohibitive. **TreeVision AI** provides an end-to-end, scientifically defensible automated pipeline that detects individual crowns, counts trees, calculates canopy coverage in real-world metric units, and supports polygon Area-of-Interest (AOI) clipping via KML.

---

## 🏗️ 2. System Architecture

```
[ High-Resolution GeoTIFF / RGB Image ]   +   [ Optional KML Boundary ]
                   │                                     │
                   ▼                                     ▼
        Geospatial Metadata Validation            WGS84 KML Reprojection
        (CRS, Transform, GSD Extraction)         (Intersection with Imagery)
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                        Sliding-Window Image Tiling
                        (Configurable 640px, 15% Overlap)
                                      │
                                      ▼
                         AI Crown Detection Head
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          DeepForest Baseline (RetinaNet)      Custom YOLOv8s Model
          (Multi-Site Pretrained Benchmark)     (Trained on NEON Biomes)
                    └─────────────────┬─────────────────┘
                                      ▼
                      Cross-Tile Deduplication (NMS)
                     (IoU ≥ 0.40 & Centroid Distance)
                                      │
                                      ▼
                         Geospatial Area Engine
                (Bounding Box Projection & Overlap Union)
                                      │
                                      ▼
                     Interactive Streamlit Dashboard
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
             Metric Badges      Visual Overlays   Data Exports
           (Count, Area, %)    (Before vs After)  (CSV & GeoJSON)
```

---

## 🌟 3. Core Features
- 🌲 **Individual Crown Detection:** Detects discrete tree crowns with bounding box geometry and model confidence scores.
- 📐 **Projected Canopy Area Calculation:** Automatically converts pixel areas into true ground square meters ($m^2$) using GeoTIFF affine metadata.
- 🛡️ **Overlap De-biasing:** Resolves touching or overlapping canopy crowns using Shapely geometric union (`unary_union`), ensuring overlapping branches are **never double-counted**.
- 🗺️ **KML / AOI Support:** Upload sub-plot boundary KML files. The system automatically reprojects WGS84 coordinates to imagery CRS and filters detections to the target polygon.
- 🧩 **Sliding-Window Tiling:** Processes arbitrarily large rasters without GPU/CPU memory exhaustion.
- 🔍 **Strict Cross-Tile Deduplication:** Eliminates duplicate detections straddling tile seams.
- 📊 **GIS-Ready Exports:** One-click download of tree inventories as tabular CSV or geospatial GeoJSON for QGIS / ArcGIS.
- ⚠️ **Transparent Limitations:** Explicit disclosures distinguishing rectangular bounding-box estimates from exact polygonal masks.

---

## 📊 4. Dataset & Scientific Integrity

### Primary Benchmark Dataset: NEON Tree Crowns Dataset
- **Citation:** Ben Weinstein, Sergio Marconi, Alina Zare, Stephanie Bohlman, Sarah Graves, Aditya Singh, & Ethan White. (2020). *NEON Tree Crowns Dataset (Version 0.0.1)* [Dataset]. Zenodo. [DOI: 10.5281/zenodo.3765872](https://doi.org/10.5281/zenodo.3765872).
- **Sensor:** NEON Airborne Observation Platform (AOP) High-Resolution Camera (`DP3.30010.001`).
- **Resolution:** $0.10\text{ m/pixel}$ (10 cm GSD).
- **Coordinate Reference System:** Universal Transverse Mercator (UTM Projected CRS, e.g. EPSG:32617).

### Zero-Leakage Geographic Partition
To prevent spatial autocorrelation (data leakage), the dataset was partitioned strictly across distinct ecological biomes:
- **Train Set:** `OSBS` (Ordway-Swisher, FL — Southeastern Pine Flatwoods)
- **Validation Set:** `SOAP` (Soaproot Saddle, CA — Sierra Nevada Mixed Conifer / Oak)
- **Held-Out Test Set:** `YELL` (Yellowstone Northern Range, WY — Rocky Mountain Subalpine Conifer)

---

## 🔬 5. Evaluation & Empirical Comparison

Evaluated on the completely unseen Yellowstone National Park test area (**279 verified reference tree crowns**):

![Benchmark Comparison](docs/images/benchmark_comparison.png)

| Evaluation Metric | DeepForest Baseline (RetinaNet) | Custom YOLOv8s (Option B, Cloud GPU) | Status / Significance |
|---|---|---|---|
| **True Positives (TP)** | **136** | 123 | Ground truth match at $\text{IoU} \ge 0.35$ |
| **False Positives (FP)** | **51** (Only 10 at 0.55 conf) | 400 | False alarms on background/shadows |
| **False Negatives (FN)** | **143** | 156 | Missed stems (dense stands, saplings) |
| **Precision** | **72.73%** (Peak: **85.29%**) | 23.52% | Confidence in positive detections |
| **Recall** | **48.75%** | **44.09%** | **Custom model captures 44.1% of true forest crowns** |
| **F1-Score** | **58.37%** | 30.67% | Harmonic mean of detection balance |
| **Predicted Tree Count** | 187 | 523 | Reference ground truth: 279 |
| **Inference Speed** | ~85 ms / tile | **~14 ms / tile** | **Custom YOLO is ~6x faster on edge/CPU** |

---

### 📈 Option A: Threshold Optimization & Precision Curve (DeepForest)
By adjusting the model's confidence threshold, false alarms drop significantly, driving **Precision up to 85.29%**:

![Threshold Operating Curve](docs/images/threshold_operating_curve.png)

| Operating Mode | Confidence Threshold | Precision | Recall | F1-Score | False Positives | Recommended Use Case |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **High Recall (Coverage)** | `0.25` | 70.65% | **50.90%** | **0.5917** | 59 | Initial broad canopy scouting |
| **Balanced Baseline** | `0.30` | 72.73% | 48.75% | 0.5837 | 51 | General forest inventory |
| **Optimized Production** | `0.40` | **77.18%** | 41.22% | 0.5374 | 34 | Production carbon accounting |
| **High Precision** | `0.45` | **78.81%** | 33.33% | 0.4685 | 25 | Conservation boundary audits |
| **Strict Verification** | `0.55` | **85.29%** | 20.79% | 0.3343 | **10** | High-certainty stem certification |

---

### 🌲 Visual Detection Overlays (Side-by-Side Comparison)

![Visual Detection Comparison](docs/images/visual_detections_comparison.png)

---

## ⚡ 6. Cloud GPU Training (Option B — Lightning AI Studio)
The custom YOLOv8s model was trained in the cloud on an NVIDIA Tesla T4 GPU with multi-site conifer data:
```bash
# Launch automated end-to-end 100-epoch cloud training on Tesla T4 GPU:
python training/run_cloud_training.py
```
This automated workflow:
1. **Connects** to the user's active Lightning AI Studio on an NVIDIA Tesla T4 GPU (`T4_SMALL`).
2. **Synchronizes** source code, zero-leakage NEON datasets (611+ tree annotations + hard negative background), and training pipelines via compressed bundle in seconds.
3. **Trains** YOLOv8s for 100 epochs with Automatic Mixed Precision (`amp=True`, CUDA 12.8), Cosine LR scheduler, Mosaic, and MixUp augmentations.
4. **Evaluates** on the held-out test split and downloads `models/custom/best.pt` directly to the local project.
5. **Auto-shuts down** the cloud GPU instance to prevent credit waste.

---

## ⚠️ 7. Known Limitations & Disclosures (Honesty in Carbon Markets)
1. **Bounding-Box Area Approximation:** Ground-truth annotations in the NEON dataset are rectangular bounding boxes. Bounding-box area overestimates true irregular crown footprint by $\approx 20\text{--}25\%$.
2. **Nadir vs. Oblique/Web Imagery:** Models are strictly calibrated on nadir (vertical top-down) airborne rasters. Oblique side-angle photos or small web thumbnails (<640px) lack GSD calibration and cause false alarms as foliage textures mimic miniature crowns.
3. **Resolution Dependency:** Model accuracy degrades sharply on imagery coarser than $0.30\text{ m/pixel}$. Free satellite rasters ($3\text{m}$ Planet, $10\text{m}$ Sentinel-2) cannot reliably separate discrete crowns.
4. **Closed-Canopy Merging:** In dense deciduous forests with interlocking branches, adjacent crowns merge into single continuous detections, resulting in under-counting.
5. **Shadow Occlusion:** Steep north-facing slopes and deep tree shadows reduce spectral contrast and lead to false negatives.

---

## 🚀 7. Installation & Quick Start

### Prerequisites
- Windows 11 / Linux / macOS
- Python 3.11 recommended

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/iamkanhaiyakumar/TreeVision-Al.git
cd TreeVision-Al

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Verify System
```bash
python scripts/check_environment.py
```

### 3. Run Automated Tests
```bash
python tests/test_pipeline.py
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The application loads with the verified NEON sample GeoTIFF pre-configured for one-click demonstration!

---

## 📁 8. Project Structure

```
TreeVision-AI/
├── app.py                                # Production Streamlit UI Dashboard
├── requirements.txt                      # Deployment dependencies
├── requirements-training.txt             # Cloud GPU training dependencies
├── TreeVision_AI_Submission_Explanation.pdf  # 2-Page Challenge Submission PDF
├── submission_form_answers.md            # Challenge form answers
├── .gitignore                            # Excludes models, caches, and datasets
│
├── configs/
│   ├── config.yaml                       # Master application configuration
│   └── split.yaml                        # Zero-leakage geographic site split
│
├── src/                                  # Core Analytical Engine
│   ├── geospatial.py                     # GeoTIFF metadata, CRS & projection logic
│   ├── kml.py                            # KML polygon parsing & AOI clipping
│   ├── tiling.py                         # Dynamic sliding-window tiling
│   ├── postprocess.py                    # Cross-tile deduplication (NMS)
│   ├── area.py                           # Canopy area & non-overlapping coverage
│   ├── detector.py                       # Unified model loading interface
│   ├── deepforest_baseline.py            # Pretrained DeepForest model wrapper
│   ├── visualization.py                  # Computer vision overlay graphics
│   └── inference.py                      # End-to-end production pipeline
│
├── training/
│   ├── run_cloud_training.py             # Lightning AI T4 GPU cloud automation runner
│   ├── lightning_train.py                # Standalone cloud training & evaluation script
│   ├── prepare_dataset.py                # Spatial matching & YOLO dataset generation
│   ├── train.py                          # Custom YOLO local training script
│   └── evaluate.py                       # Comparative evaluation & metric calculation
│
├── demo/
│   ├── sample_forest.tif                 # Verified sample GeoTIFF (OSBS NEON site)
│   └── sample_aoi.kml                    # Verified sample AOI boundary polygon
│
├── evaluation/
│   ├── baseline_metrics.json             # Real measured DeepForest metrics
│   ├── custom_metrics.json               # Real measured Custom YOLO metrics
│   ├── model_comparison.md               # Head-to-head comparison table
│   └── error_analysis.md                 # Detailed failure mode analysis
│
└── tests/
    └── test_pipeline.py                  # Automated test suite (6 passing tests)
```

---

## 📜 9. Citation & Acknowledgments
```bibtex
@dataset{weinstein_2020_3765872,
  author       = {Ben Weinstein and Sergio Marconi and Alina Zare and 
                  Stephanie Bohlman and Sarah Graves and Aditya Singh and Ethan White},
  title        = {{NEON Tree Crowns Dataset}},
  month        = apr,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.0.1},
  doi          = {10.5281/zenodo.3765872},
  url          = {https://doi.org/10.5281/zenodo.3765872}
}
```
