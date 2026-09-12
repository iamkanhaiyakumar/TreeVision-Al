"""
TreeVision AI — Dataset Preparation & YOLO Dataset Generation (Phase 2)
Converts NEON geospatial annotations and imagery tiles into normalized YOLO format:
<class_id> <x_center> <y_center> <width> <height>

Strictly enforces geographic site split to prevent spatial autocorrelation.
Validates bounding box geometry, clips to boundary, and logs dropped/invalid records.
"""

import os
import sys
import yaml
import shutil
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine
from PIL import Image

def convert_box_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_width: int, img_height: int
):
    """
    Converts pixel (xmin, ymin, xmax, ymax) into normalized YOLO coordinates:
    (x_center, y_center, width, height) in range [0, 1].
    """
    # Clip to image boundaries
    cl_xmin = max(0.0, min(float(img_width), xmin))
    cl_ymin = max(0.0, min(float(img_height), ymin))
    cl_xmax = max(0.0, min(float(img_width), xmax))
    cl_ymax = max(0.0, min(float(img_height), ymax))
    
    bw = cl_xmax - cl_xmin
    bh = cl_ymax - cl_ymin
    
    # Reject degenerate boxes (e.g. area < 1 pixel or width/height <= 0)
    if bw < 2.0 or bh < 2.0:
        return None
        
    x_center = (cl_xmin + cl_xmax) / (2.0 * img_width)
    y_center = (cl_ymin + cl_ymax) / (2.0 * img_height)
    norm_w = bw / float(img_width)
    norm_h = bh / float(img_height)
    
    # Assert within valid range
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < norm_w <= 1.0 and 0.0 < norm_h <= 1.0):
        return None
        
    return 0, round(x_center, 6), round(y_center, 6), round(norm_w, 6), round(norm_h, 6)

def prepare_yolo_dataset(
    output_base: str = "datasets/neon/processed",
    tile_size: int = 640
):
    """
    Assembles train, validation, and held-out test sets using NEON sample tiles and annotations.
    """
    print("=" * 65)
    print("PHASE 2: PREPARING YOLO DATASET WITH GEOGRAPHIC SPLIT")
    print("=" * 65)
    
    import deepforest
    data_dir = os.path.dirname(deepforest.get_data("OSBS_029.tif"))
    
    # Create directory structure
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(output_base, "images", s), exist_ok=True)
        os.makedirs(os.path.join(output_base, "labels", s), exist_ok=True)

    # We map NEON samples across distinct sites:
    # Train: OSBS_029 (Florida Pine)
    # Val: SOAP samples (California Conifer/Oak)
    # Test: YELL / TEAK (Rocky Mountain Conifer / High Sierra)
    site_mapping = {
        "train": [
            ("OSBS_029.png", "OSBS_029.csv")
        ],
        "val": [
            ("SOAP_061.png", "SOAP_061.xml")
        ],
        "test": [
            ("2019_YELL_2_541000_4977000_image_crop.png", "2019_YELL_2_541000_4977000_image_crop.xml")
        ]
    }

    report = {
        "total_images": 0,
        "total_annotations": 0,
        "valid_annotations": 0,
        "dropped_annotations": 0,
        "split_stats": {}
    }

    for split, files in site_mapping.items():
        report["split_stats"][split] = {"images": 0, "annotations": 0}
        for img_name, annot_name in files:
            img_src = os.path.join(data_dir, img_name)
            annot_src = os.path.join(data_dir, annot_name)
            
            if not os.path.exists(img_src) or not os.path.exists(annot_src):
                print(f"Skipping {img_name}, file not found")
                continue
                
            pil_img = Image.open(img_src).convert("RGB")
            w, h = pil_img.size
            
            # Destination image
            dst_img_path = os.path.join(output_base, "images", split, img_name)
            pil_img.save(dst_img_path)
            
            # Parse annotations
            boxes = []
            if annot_name.endswith(".csv"):
                df = pd.read_csv(annot_src)
                for _, row in df.iterrows():
                    boxes.append((float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])))
            elif annot_name.endswith(".xml"):
                import xml.etree.ElementTree as ET
                tree = ET.parse(annot_src)
                for obj in tree.findall(".//object"):
                    bnd = obj.find("bndbox")
                    if bnd is not None:
                        boxes.append((
                            float(bnd.find("xmin").text),
                            float(bnd.find("ymin").text),
                            float(bnd.find("xmax").text),
                            float(bnd.find("ymax").text)
                        ))

            # Convert to YOLO format
            yolo_lines = []
            for b in boxes:
                report["total_annotations"] += 1
                yolo_box = convert_box_to_yolo(b[0], b[1], b[2], b[3], w, h)
                if yolo_box is not None:
                    cls_id, xc, yc, nw, nh = yolo_box
                    yolo_lines.append(f"{cls_id} {xc} {yc} {nw} {nh}\n")
                    report["valid_annotations"] += 1
                else:
                    report["dropped_annotations"] += 1

            # Write label txt
            base_txt_name = os.path.splitext(img_name)[0] + ".txt"
            dst_txt_path = os.path.join(output_base, "labels", split, base_txt_name)
            with open(dst_txt_path, "w") as f:
                f.writelines(yolo_lines)
                
            report["split_stats"][split]["images"] += 1
            report["split_stats"][split]["annotations"] += len(yolo_lines)
            report["total_images"] += 1

    # Create data.yaml
    # Use relative paths from dataset base
    data_yaml_content = {
        "path": os.path.abspath(output_base).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "Tree"},
        "nc": 1
    }
    
    yaml_path = os.path.join(output_base, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)

    print("\nYOLO Dataset Preparation Report:")
    print(f"Total Images: {report['total_images']}")
    print(f"Total Annotations: {report['total_annotations']}")
    print(f"Valid YOLO Annotations: {report['valid_annotations']}")
    print(f"Dropped Degenerate Annotations: {report['dropped_annotations']}")
    for s, st in report["split_stats"].items():
        print(f"  - {s.upper():<6}: {st['images']} images, {st['annotations']} crown annotations")
    print(f"Dataset config written to: {yaml_path}")
    print("=" * 65)

    return yaml_path

if __name__ == "__main__":
    prepare_yolo_dataset()
