#!/usr/bin/env python3
"""Run Controlled Experiment 1 (exp1_imgsz960), evaluate baseline vs exp1, and run error analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import yaml
from ultralytics import YOLO

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluate_detector import evaluate_detector
from scripts.train_detector import train_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASELINE_WEIGHTS = Path("models/detection/detector/weights/best.pt")
EXPECTED_BASELINE_HASH = "0dcfd7a50c4a50a86252a746d31967c154ae61f8c2f369d8b67bf47cc0c8184b"
DATA_YAML = Path("data/detection/data.yaml")

EXP_DIR = Path("runs/detect/exp1_imgsz960")
REPORTS_DIR = Path("reports/yolo/experiments")
ERR_ANALYSIS_DIR = Path("reports/yolo/error_analysis/exp1_imgsz960")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def box_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[0])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[0])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main():
    # 1. Pre-training Safety Check
    logger.info("=== Pre-training Safety Check ===")
    assert BASELINE_WEIGHTS.exists(), "best.pt missing"
    current_hash = sha256_file(BASELINE_WEIGHTS)
    logger.info("best.pt SHA256: %s", current_hash)
    if current_hash != EXPECTED_BASELINE_HASH:
        logger.error("BASELINE WEIGHT HASH MISMATCH! Expected %s, got %s", EXPECTED_BASELINE_HASH, current_hash)
        sys.exit(1)
    logger.info("Baseline weight hash verified OK.")

    # 2. Run Training
    logger.info("=== Starting Experiment 1: imgsz=960 ===")
    t0 = time.time()
    best_exp_weights = train_detector(
        data_yaml=DATA_YAML,
        model_name="yolo11n.pt",
        epochs=50,
        batch=16,
        imgsz=960,
        output_dir=EXP_DIR,
        device="auto",
        workers=4,
        patience=10,
        seed=42,
    )
    duration = time.time() - t0
    logger.info("Training finished in %.2f seconds. Weights: %s", duration, best_exp_weights)

    exp_weight_hash = sha256_file(best_exp_weights)

    # 3. Evaluate Experiment 1 on Val & Test splits
    logger.info("=== Evaluating Experiment 1 on Validation Split ===")
    val_metrics = evaluate_detector(
        model_path=best_exp_weights,
        data_yaml=DATA_YAML,
        split="val",
        output_report=REPORTS_DIR / "exp1_imgsz960_val.json",
    )

    logger.info("=== Evaluating Experiment 1 on Test Split ===")
    test_metrics = evaluate_detector(
        model_path=best_exp_weights,
        data_yaml=DATA_YAML,
        split="test",
        output_report=REPORTS_DIR / "exp1_imgsz960_test.json",
    )

    # Save exp1_imgsz960.json
    exp_report = {
        "experiment_id": "exp1_imgsz960",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "architecture": "YOLO11n",
        "pretrained_model": "yolo11n.pt",
        "dataset": "data/detection (V2 FROZEN)",
        "imgsz": 960,
        "epochs": 50,
        "batch": 16,
        "optimizer": "AdamW (Auto)",
        "learning_rate": 0.000714,
        "weight_decay": 0.0005,
        "patience": 10,
        "seed": 42,
        "device": "cpu",
        "workers": 4,
        "ultralytics_version": "8.4.120",
        "output_directory": str(EXP_DIR),
        "checkpoint_path": str(best_exp_weights),
        "checkpoint_sha256": exp_weight_hash,
        "training_duration_seconds": round(duration, 2),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "exp1_imgsz960.json", "w", encoding="utf-8") as f:
        json.dump(exp_report, f, indent=2)

    # 4. Error Analysis & Visual Comparisons for Exp1
    logger.info("=== Running Error Analysis for Exp 1 ===")
    ERR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = ERR_ANALYSIS_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]

    model_exp1 = YOLO(best_exp_weights)
    model_base = YOLO(BASELINE_WEIGHTS)

    test_img_dir = Path("data/detection/test/images")
    test_lbl_dir = Path("data/detection/test/labels")
    images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))

    stats = {cls_name: {"gt": 0, "tp": 0, "fn": 0, "fp": 0, "wrong_cls": 0, "small_fn": 0, "medium_fn": 0, "large_fn": 0} for cls_name in names}
    viz_count = 0

    for img_path in images:
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

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

        res_exp1 = model_exp1.predict(source=str(img_path), imgsz=960, conf=0.25, verbose=False)[0]
        preds_exp1 = []
        for box in res_exp1.boxes:
            b = box.xyxy[0].cpu().numpy().tolist()
            c = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            preds_exp1.append({"cls": c, "cls_name": names[c], "box": b, "conf": conf, "matched": False})

        for p in preds_exp1:
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

        # Side-by-side comparison for problem classes
        has_grape = any(g["cls_name"] in ["Grape", "cherry", "guava", "Apple"] for g in gt_boxes)
        if has_grape and viz_count < 10:
            res_base = model_base.predict(source=str(img_path), imgsz=640, conf=0.25, verbose=False)[0]
            preds_base = []
            for box in res_base.boxes:
                b = box.xyxy[0].cpu().numpy().tolist()
                c = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                preds_base.append({"cls": c, "cls_name": names[c], "box": b, "conf": conf})

            canvas_base = img.copy()
            canvas_exp1 = img.copy()

            for g in gt_boxes:
                b = [int(v) for v in g["box"]]
                cv2.rectangle(canvas_base, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.putText(canvas_base, f"GT:{g['cls_name']}", (b[0], max(0, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.rectangle(canvas_exp1, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.putText(canvas_exp1, f"GT:{g['cls_name']}", (b[0], max(0, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            for pb in preds_base:
                b = [int(v) for v in pb["box"]]
                cv2.rectangle(canvas_base, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 2)
                cv2.putText(canvas_base, f"B640:{pb['cls_name']}", (b[0], b[3] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            for pe in preds_exp1:
                b = [int(v) for v in pe["box"]]
                cv2.rectangle(canvas_exp1, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(canvas_exp1, f"E960:{pe['cls_name']}", (b[0], b[3] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            stacked = cv2.hstack([canvas_base, canvas_exp1])
            cv2.imwrite(str(viz_dir / f"comp_{img_path.name}"), stacked)
            viz_count += 1

    (ERR_ANALYSIS_DIR / "error_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Saved Exp 1 error summary & visualizations.")

    # 5. Post-training Integrity Check
    logger.info("=== Post-training Safety Check ===")
    post_hash = sha256_file(BASELINE_WEIGHTS)
    if post_hash != EXPECTED_BASELINE_HASH:
        logger.error("POST-TRAINING BASELINE WEIGHT MUTATION! Expected %s, got %s", EXPECTED_BASELINE_HASH, post_hash)
        sys.exit(1)
    logger.info("Baseline weight hash unchanged & verified: %s", post_hash)


if __name__ == "__main__":
    main()
