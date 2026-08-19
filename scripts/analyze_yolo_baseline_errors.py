#!/usr/bin/env python3
"""Run read-only predictions of best.pt on V2 dataset test split to analyze failures."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/detection/detector/weights/best.pt")
DATA_YAML = Path("data/detection/data.yaml")
OUT_DIR = Path("reports/yolo/error_analysis")


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[0])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[0])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def analyze_errors():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]

    model = YOLO(MODEL_PATH)
    test_img_dir = Path("data/detection/test/images")
    test_lbl_dir = Path("data/detection/test/labels")

    images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    logger.info("Analyzing %d test images...", len(images))

    stats = {cls_name: {"gt": 0, "tp": 0, "fn": 0, "fp": 0, "wrong_cls": 0, "small_fn": 0, "medium_fn": 0, "large_fn": 0} for cls_name in names}
    viz_counts = {cls_name: 0 for cls_name in names}

    for img_path in images:
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # Parse GT
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    c = int(p[0])
                    cx, cy, w, h = (float(x) for x in p[1:])
                    x1 = (cx - w / 2) * W
                    y1 = (cy - h / 2) * H
                    x2 = (cx + w / 2) * W
                    y2 = (cy + h / 2) * H
                    gt_boxes.append({"cls": c, "cls_name": names[c], "box": [x1, y1, x2, y2], "rel_area": w * h, "matched": False})

        for g in gt_boxes:
            stats[g["cls_name"]]["gt"] += 1

        # Run inference
        results = model.predict(source=str(img_path), imgsz=640, conf=0.25, verbose=False)[0]
        pred_boxes = []
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy().tolist()
            c = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            pred_boxes.append({"cls": c, "cls_name": names[c], "box": b, "conf": conf, "matched": False})

        # Match preds to GT
        for p in pred_boxes:
            best_iou = 0.0
            best_gt = None
            for g in gt_boxes:
                iou = box_iou(p["box"], g["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g

            if best_iou >= 0.5:
                if best_gt["cls"] == p["cls"]:
                    p["matched"] = True
                    best_gt["matched"] = True
                    stats[p["cls_name"]]["tp"] += 1
                else:
                    p["matched"] = True
                    best_gt["matched"] = True
                    stats[p["cls_name"]]["wrong_cls"] += 1
                    stats[best_gt["cls_name"]]["wrong_cls"] += 1
            else:
                stats[p["cls_name"]]["fp"] += 1

        for g in gt_boxes:
            if not g["matched"]:
                stats[g["cls_name"]]["fn"] += 1
                if g["rel_area"] < 0.01:
                    stats[g["cls_name"]]["small_fn"] += 1
                elif g["rel_area"] < 0.25:
                    stats[g["cls_name"]]["medium_fn"] += 1
                else:
                    stats[g["cls_name"]]["large_fn"] += 1

        # Save visualization if interesting error
        has_grape_error = any(g["cls_name"] == "Grape" and not g["matched"] for g in gt_boxes)
        has_cherry_error = any(g["cls_name"] == "cherry" and not g["matched"] for g in gt_boxes)
        has_apple_error = any(g["cls_name"] == "Apple" and not g["matched"] for g in gt_boxes)
        has_guava_error = any(g["cls_name"] == "guava" and not g["matched"] for g in gt_boxes)

        if (has_grape_error or has_cherry_error or has_apple_error or has_guava_error) and sum(viz_counts.values()) < 20:
            viz_img = img.copy()
            # Draw GT in RED
            for g in gt_boxes:
                b = [int(v) for v in g["box"]]
                cv2.rectangle(viz_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.putText(viz_img, f"GT: {g['cls_name']}", (b[0], max(0, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw Preds in CYAN
            for p in pred_boxes:
                b = [int(v) for v in p["box"]]
                cv2.rectangle(viz_img, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 2)
                cv2.putText(viz_img, f"PRED: {p['cls_name']} {p['conf']:.2f}", (b[0], b[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            out_path = OUT_DIR / f"error_{img_path.name}"
            cv2.imwrite(str(out_path), viz_img)
            viz_counts["Grape"] += 1

    summary_path = OUT_DIR / "error_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Saved error summary to %s", summary_path)


if __name__ == "__main__":
    analyze_errors()
