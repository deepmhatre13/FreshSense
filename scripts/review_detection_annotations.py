#!/usr/bin/env python3
"""Review suspicious annotations for the SmartFreshAI detection dataset.

This is the **Phase 1** annotation-review system for the Dataset-V3 preparation
pipeline. It is intentionally *read-only*: it never writes to, moves, renames,
or deletes any dataset file. It only:

1. Flags suspicious images (empty labels, huge boxes, tiny boxes, many objects,
   ambiguous/similar-class images).
2. Renders annotated visualizations of each flagged image, overlaying the
   bounding boxes, class names, class IDs, the image filename and the reason
   the image was flagged.
3. Writes review JSON records that humans review before any V3 correction.

Conventions follow the rest of the SmartFreshAI scripts (module docstring,
``from __future__ import annotations``, argparse CLI, standard logging, JSON
reports under ``reports/``).

Usage:
    python scripts/review_detection_annotations.py
    python scripts/review_detection_annotations.py --data-dir data/detection
        --outdir reports/audit_review
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allow running directly from any CWD.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.audit_detection_dataset import (  # noqa: E402
    SPLIT_NAMES,
    _find_dataset_root,
    _list_images,
    _list_labels,
    _read_boxes,
    load_data_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Classes frequently confused by the current model (webcam + audit evidence).
AMBIGUOUS_CLASSES = {"Apple", "Mango", "Cherry", "Chickoo", "Guava", "Grape"}

# Suspicion thresholds.
_HUGE_AREA = 0.95 * 0.95     # bbox covers ~the whole image
_TINY_AREA = 0.005           # bbox smaller than 0.5% of image area
# "many objects": a per-image count is flagged if it exceeds this absolute cap.
_MANY_OBJECTS_CAP = 60


def collect_suspicious_images(
    data_root: Path,
    names: List[str],
    max_per_category: int = 200,
) -> Dict[str, List[dict]]:
    """Walk every split and flag suspicious images by category.

    Returns a dict: category name -> list of review records. The dataset is
    never modified. Category keys:
    - empty_labels (image exists but its label is missing OR an empty file)
    - huge_boxes (any bbox area >= ~whole image)
    - tiny_boxes (any bbox area < _TINY_AREA)
    - many_objects (objects per image > _MANY_OBJECTS_CAP)
    - ambiguous_classes (image contains >=1 of the confusion-prone classes)
    """
    empty: List[dict] = []
    huge: List[dict] = []
    tiny: List[dict] = []
    many: List[dict] = []
    ambiguous: List[dict] = []

    for split in SPLIT_NAMES:
        im_dir = data_root / split / "images"
        lb_dir = data_root / split / "labels"
        if not im_dir.is_dir() or not lb_dir.is_dir():
            continue
        images = _list_images(im_dir)
        label_stems = {p.stem for p in _list_labels(lb_dir)}
        for img in images:
            stem = img.stem
            lbl = lb_dir / (stem + ".txt")
            is_empty = (not lbl.exists()) or (lbl.stat().st_size == 0)
            if is_empty:
                if len(empty) < max_per_category:
                    empty.append(_record(
                        img, lb_dir, names, split, "empty_label",
                        extra={"empty_label_path": str(lbl) if lbl.exists() else None}))
                continue
            boxes, issues = _read_boxes(lbl, len(names))
            areas = [(b[3] * b[4]) for b in boxes if 0 <= b[0] < len(names)]
            if any(a >= _HUGE_AREA for a in areas):
                huge.append(_record(
                    img, lb_dir, names, split, "huge_box",
                    extra={"max_area_ratio": round(max(areas, default=0), 6)}))
            if any(a < _TINY_AREA for a in areas):
                if len(tiny) < max_per_category:
                    tiny.append(_record(
                        img, lb_dir, names, split, "tiny_box",
                        extra={"min_area_ratio": round(min(areas), 6)}))
            if len(boxes) > _MANY_OBJECTS_CAP:
                if len(many) < max_per_category:
                    many.append(_record(
                        img, lb_dir, names, split, "many_objects",
                        extra={"object_count": len(boxes)}))
            present = {names[b[0]] for b in boxes if 0 <= b[0] < len(names)}
            if present & AMBIGUOUS_CLASSES:
                if len(ambiguous) < max_per_category:
                    ambiguous.append(_record(
                        img, lb_dir, names, split, "ambiguous_classes",
                        extra={"classes_present": sorted(present)}))

    logger.info("Suspicion counts: empty=%d, huge=%d, tiny=%d, many=%d, ambiguous=%d",
                len(empty), len(huge), len(tiny), len(many), len(ambiguous))
    return {
        "empty_labels": empty,
        "huge_boxes": huge,
        "tiny_boxes": tiny,
        "many_objects": many,
        "ambiguous_classes": ambiguous,
    }


def _label_path(labels_dir: Path, stem: str) -> Optional[Path]:
    """Return the ``.txt`` label path for an image stem if it exists."""
    for ext in (".txt",):
        cand = labels_dir / (stem + ext)
        if cand.exists():
            return cand
    return None


def render_review_image(
    image_path: Path,
    label_path: Optional[Path],
    names: List[str],
    reasons: List[str],
    out_path: Path,
) -> None:
    """Write an annotated review image with boxes, labels, filename and reasons."""
    img = cv2.imread(str(image_path))
    if img is None:
        img = np.full((480, 640, 3), 40, dtype=np.uint8)
    canvas = img.copy()
    h, w = canvas.shape[:2]
    if label_path is not None and label_path.exists():
        text = label_path.read_text(encoding="utf-8").strip()
        if text:
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = (float(p) for p in parts[1:])
                except ValueError:
                    continue
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = (0, 220, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                label = f"id={cls_id} {names[cls_id] if 0 <= cls_id < len(names) else '?'}"
                cv2.putText(canvas, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Header: filename
    cv2.putText(canvas, image_path.name, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # Footer: reasons (each line)
    for i, r in enumerate(reasons):
        cv2.putText(canvas, f"- {r}", (8, h - 8 - (len(reasons) - 1 - i) * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def compute_image_md5(path: Path) -> str:
    """Return the MD5 hex digest of the raw file bytes of *path*."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_md5_index(data_root: Path) -> Dict[str, List[dict]]:
    """Index every image by its MD5 across all splits.

    Each index entry carries the split, image stem, image path, whether a
    non-empty label exists, and (if so) the label text. Used to prove whether
    an empty-label image is an exact byte-duplicate of a labeled image.
    """
    index: Dict[str, List[dict]] = {}
    for split in SPLIT_NAMES:
        im_dir = data_root / split / "images"
        lb_dir = data_root / split / "labels"
        if not im_dir.is_dir():
            continue
        for img in _list_images(im_dir):
            stem = img.stem
            lbl = lb_dir / (stem + ".txt")
            has_label = lbl.exists() and lbl.stat().st_size > 0
            entry = {
                "split": split,
                "stem": stem,
                "image": str(img),
                "has_label": has_label,
                "label_text": lbl.read_text(encoding="utf-8") if has_label else "",
                "label_path": str(lbl),
            }
            index.setdefault(compute_image_md5(img), []).append(entry)
    return index


def find_v3_corrections(
    empty_records: List[dict], md5_index: Dict[str, List[dict]]
) -> List[dict]:
    """Classify each empty-label image for V3 handling.

    Returns records with decision:
    - ``apply_to_v3``  : byte-exact duplicate of at least one *labeled* image
      (hard evidence the empty label is a labeling error) -> safe to copy.
    - ``needs_manual_review`` : no labeled twin found; resolution needs visual
      judgment (whether fruit is present) -> do NOT auto-fix.
    Each record also reports cross-split leakage if its MD5 appears in >1 split.
    """
    corrections: List[dict] = []
    for rec in empty_records:
        img = Path(rec["image"])
        md5 = compute_image_md5(img)
        twins = md5_index.get(md5, [])
        labeled_twins = [t for t in twins if t["has_label"]]
        splits = sorted({t["split"] for t in twins})
        leak = len(splits) > 1
        if labeled_twins:
            twins_out = [
                {"split": t["split"], "stem": t["stem"],
                 "label_path": t["label_path"], "label_text": t["label_text"]}
                for t in labeled_twins
            ]
            corrections.append({
                **rec,
                "decision": "apply_to_v3",
                "evidence": "exact_duplicate_of_labeled_image",
                "confidence": "high",
                "matched_labeled_copies": twins_out,
                "cross_split_leakage": leak,
            })
        else:
            corrections.append({
                **rec,
                "decision": "needs_manual_review",
                "evidence": "no_exact_duplicate_of_labeled_image",
                "confidence": "none",
                "matched_labeled_copies": [],
                "cross_split_leakage": leak,
            })
    return corrections


def _record(image_path, labels_dir, names, split, reason, decision="needs_manual_review",
            extra=None):
    """Build a single review record; does NOT read the label (caller decides)."""
    rec = {
        "image": str(image_path),
        "image_filename": image_path.name,
        "split": split,
        "category": reason,
        "decision": decision,
        "reason": reason,
        }
    if extra:
        rec.update(extra)
    return rec


def review_dataset(
    data_dir: str | Path,
    out_dir: Path,
    max_per_category: int = 200,
) -> Dict[str, Dict]:
    """Run the full read-only review: JSON records + visualizations per category.

    ``out_dir`` receives ``<category>/`` folders of images and a
    ``<category>_review.json`` record file for every category. Returns a summary.
    """
    data_dir = Path(data_dir)
    data_root = _find_dataset_root(data_dir)
    if data_root is None:
        raise FileNotFoundError(f"No dataset root with data.yaml under: {data_dir}")
    _, nc, names = load_data_config(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    findings = collect_suspicious_images(data_root, names, max_per_category)
    summary: Dict[str, Dict] = {}

    for category, records in findings.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        rendered = 0
        for rec in records[:max_per_category]:
            img = Path(rec["image"])
            lbl = _label_path(data_root / rec["split"] / "labels", img.stem)
            render_review_image(img, lbl, names, [rec["reason"]], cat_dir / img.name)
            rendered += 1
        json_path = out_dir / f"{category}_review.json"
        json_path.write_text(
            json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8")
        summary[category] = {
            "count": len(records),
            "rendered": rendered,
            "review_json": str(json_path),
            "visual_dir": str(cat_dir),
        }
        logger.info("%s: %d flagged images (rendered %d)",
                    category, summary[category]["count"], rendered)

    index_path = out_dir / "review_index.json"
    index_path.write_text(
        json.dumps({k: v["review_json"] for k, v in summary.items()}, indent=2) + "\n",
        encoding="utf-8",
    )

    # V3 correction manifest: evidence-based decisions for empty-label images.
    md5_index = build_md5_index(data_root)
    corrections = find_v3_corrections(findings["empty_labels"], md5_index)
    corr_path = out_dir / "v3_corrections.json"
    corr_path.write_text(
        json.dumps(corrections, indent=2, default=str) + "\n", encoding="utf-8")
    apply_count = sum(1 for c in corrections if c["decision"] == "apply_to_v3")
    review_count = sum(1 for c in corrections if c["decision"] == "needs_manual_review")
    summary["v3_corrections"] = {
        "total": len(corrections),
        "apply_to_v3": apply_count,
        "needs_manual_review": review_count,
        "manifest": str(corr_path),
    }
    logger.info("V3 corrections: %d apply_to_v3, %d needs_manual_review (manifest -> %s)",
                apply_count, review_count, corr_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review suspicious annotations (read-only) for V3 prep")
    parser.add_argument("--data-dir", type=Path, default=Path("data/detection"),
                        help="Detection dataset root (default: data/detection)")
    parser.add_argument("--outdir", type=Path,
                        default=Path("reports/audit_review"),
                        help="Where to write review visualizations and JSON records")
    parser.add_argument("--max-per-category", type=int, default=200,
                        help="Cap of flagged images rendered per category")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SmartFreshAI - Annotation Review (Phase 1, read-only)")
    logger.info("=" * 70)
    logger.info("Data dir: %s  ->  %s", args.data_dir, args.outdir)

    summary = review_dataset(args.data_dir, args.outdir, args.max_per_category)

    logger.info("=" * 70)
    logger.info("Review complete. Outputs under: %s", args.outdir)
    for cat, s in summary.items():
        logger.info("  %s: %d flagged, %d rendered -> %s",
                    cat, s["count"], s["rendered"], s["visual_dir"])
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())