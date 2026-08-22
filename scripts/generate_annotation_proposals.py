#!/usr/bin/env python3
"""Generate AI-assisted annotation proposals for unresolved review cases.

This script runs the V2 YOLO detector over images flagged for review that lack
a recorded human decision. It generates bounding box proposals and comparison
visualizations without modifying the existing dataset.

Usage:
    python scripts/generate_annotation_proposals.py
"""
import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Ensure repository root is in Python path
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.review_detection_annotations import _find_dataset_root, _read_boxes, load_data_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Output paths
PROPOSAL_DIR = Path("reports/audit_review/ai_annotation_proposals")
VISUALIZATION_DIR = PROPOSAL_DIR / "visualizations"
PROPOSAL_FILE = PROPOSAL_DIR / "proposals.json"


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_boxes(image: np.ndarray, boxes, class_names, color=(0, 255, 0), is_yolo_norm=False) -> np.ndarray:
    """Draw bounding boxes on an image.
    If is_yolo_norm is True, boxes are [cls_id, cx, cy, w, h] normalized.
    If False, boxes are Ultralytics box objects.
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    if is_yolo_norm:
        for box in boxes:
            cls_id = int(box[0])
            cx, cy, bw, bh = box[1:5]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            label = f"{name} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    return img


def generate_proposals() -> None:
    parser = argparse.ArgumentParser(description="Generate AI-assisted annotation proposals.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/detection"))
    parser.add_argument("--weights", type=Path, default=Path("models/detection/detector/weights/best.pt"))
    args = parser.parse_args()

    PROPOSAL_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    data_root = _find_dataset_root(args.data_dir)
    if data_root is None:
        logger.error("Dataset root not found.")
        sys.exit(1)

    _, _, class_names = load_data_config(data_root)

    if not args.weights.exists():
        logger.error(f"Model weights not found at {args.weights}")
        sys.exit(1)

    model_hash = compute_file_hash(args.weights)
    logger.info("Loading YOLO model: %s", args.weights)
    model = YOLO(str(args.weights))

    # Read human decisions to find which images are already resolved.
    # NOTE: human_decisions.json contains two record shapes:
    #   1. legacy records with an explicit "image_filename" key, and
    #   2. AI-proposal review records that only carry "image" (full path) and/or
    #      an "ai_proposal" nested dict.
    # Only *real* human decisions (accepted / corrected / kept / rejected /
    # excluded) remove an image from the unresolved set. "uncertain" and
    # "pending" keep the image pending for review, matching the Phase 3.6
    # decision semantics (AI proposals are NOT approved annotations).
    _REAL_DECISIONS = {"accepted", "corrected", "kept", "rejected", "excluded"}

    def _rec_filename(rec: dict) -> str | None:
        if rec.get("image_filename"):
            return rec["image_filename"]
        if rec.get("ai_proposal") and rec["ai_proposal"].get("image"):
            return Path(rec["ai_proposal"]["image"]).name
        if rec.get("image"):
            return Path(rec["image"]).name
        return None

    human_decisions_path = Path("reports/audit_review/human_decisions.json")
    decided_images = set()
    if human_decisions_path.exists():
        with open(human_decisions_path) as f:
            hd = json.load(f)
            for rec in hd.get("records", []):
                # Skip pending / uncertain records: those images stay unresolved.
                if rec.get("human_decision") not in _REAL_DECISIONS:
                    continue
                fname = _rec_filename(rec)
                if fname:
                    decided_images.add(fname)

    # Load review categories
    review_dir = Path("reports/audit_review")
    categories_to_check = ["ambiguous_classes", "tiny_boxes", "many_objects", "empty_labels", "huge_box", "huge_boxes"]
    unresolved_items = []
    
    for p in review_dir.glob("*_review.json"):
        cat_name = p.stem.replace("_review", "")
        if cat_name not in categories_to_check:
            continue
            
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            # huge_box_review.json is a dict with records
            data = data.get("records", [])
            
        for item in data:
            img_filename = item.get("image_filename")
            if not img_filename or img_filename in decided_images:
                continue
            
            # Use 'category' field from item or fallback to file stem
            item_cat = item.get("category", cat_name)
            unresolved_items.append({
                "image_path": Path(item.get("image")),
                "image_filename": img_filename,
                "split": item.get("split"),
                "category": item_cat
            })

    logger.info("Found %d unresolved items to process.", len(unresolved_items))

    proposals_data = []

    for item in unresolved_items:
        img_path = Path(_REPO_ROOT) / item["image_path"]
        if not img_path.exists():
            logger.warning("Image missing: %s", img_path)
            continue
            
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            logger.warning("Could not read image: %s", img_path)
            continue
            
        # Get ground truth labels
        label_path = data_root / item["split"] / "labels" / (img_path.stem + ".txt")
        gt_boxes = []
        if label_path.exists():
            gt_boxes, _ = _read_boxes(label_path, len(class_names))

        # Run model inference
        results = model.predict(cv_img, verbose=False)
        ai_boxes = results[0].boxes

        # Create proposal entries
        image_proposals = []
        if ai_boxes is not None and len(ai_boxes) > 0:
            for box in ai_boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                proposal = {
                    "proposal_id": str(uuid.uuid4()),
                    "image": str(item["image_path"]),
                    "split": item["split"],
                    "review_category": item["category"],
                    "model_path": str(args.weights),
                    "model_hash": model_hash,
                    "class_id": cls_id,
                    "class_name": class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "proposal_status": "pending_human_review",
                    "created_at": datetime.now().isoformat()
                }
                image_proposals.append(proposal)
                proposals_data.append(proposal)

        # Create visualization montage
        img_h, img_w = cv_img.shape[:2]
        vis_img = np.zeros((img_h, img_w * 2, 3), dtype=np.uint8)
        
        # Left side: Ground Truth
        gt_img = draw_boxes(cv_img, gt_boxes, class_names, color=(0, 0, 255), is_yolo_norm=True)
        cv2.putText(gt_img, "GROUND TRUTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(gt_img, f"Category: {item['category']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        vis_img[:, :img_w] = gt_img
        
        # Right side: AI Proposals
        ai_img = draw_boxes(cv_img, ai_boxes, class_names, color=(0, 255, 0), is_yolo_norm=False)
        cv2.putText(ai_img, "AI PROPOSAL - NOT GROUND TRUTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        vis_img[:, img_w:] = ai_img

        vis_path = VISUALIZATION_DIR / f"{img_path.stem}_proposal.jpg"
        cv2.imwrite(str(vis_path), vis_img)

    # Save proposals JSON
    with open(PROPOSAL_FILE, "w", encoding="utf-8") as f:
        json.dump(proposals_data, f, indent=2)

    logger.info("Generated %d proposals.", len(proposals_data))
    logger.info("Proposals saved to %s", PROPOSAL_FILE)


if __name__ == "__main__":
    generate_proposals()
