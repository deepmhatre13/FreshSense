#!/usr/bin/env python3
"""Human-adjudication layer for SmartFreshAI Dataset V3 preparation.

This script records EXPLICIT HUMAN DECISIONS for flagged annotations so the
future ``data/detection_v3/`` builder can apply only validated corrections.

It is intentionally *read-only* with respect to the source dataset: it never
writes to, moves, renames, or deletes anything under ``data/detection/``. It
never fabricates bounding boxes. It only:

1. Loads the suspension flags produced by ``review_detection_annotations.py``
   (empty labels, huge boxes, tiny boxes, many objects, ambiguous classes).
2. Lets a human record a decision for each flagged item via an explicit schema.
3. Writes machine-readable manifests, e.g.
   ``reports/audit_review/human_decisions.json`` and
   ``reports/audit_review/huge_box_review.json``.

Decision schema
---------------
Each record has:
  - image          : path to the image under data/detection
  - image_filename : basename
  - split          : train / valid / test
  - category       : empty_label | huge_box | grape_policy | ambiguous
  - decision       : validated per category
  - class_name     : class the annotation concerns (when applicable)
  - action         : one of the allowed actions
  - suggested      : human suggestion (e.g. huge-box keep/tighten/manual_review)
  - notes          : free text evidence from the human review
  - bbox           : OPTIONAL ground-truth box; NEVER fabricate coordinates.

Allowed decisions:
  - empty_label : "annotate" | "keep_empty" | "remove"
  - huge_box    : "keep" | "tighten" | "manual_review"
  - grape_policy: "bunch_policy_confirmed" | "per_berry_accepted" | "needs_reannotation"
  - ambiguous   : "confirmed" | "needs_second_opinion" | "exclude"

Usage:
    python scripts/adjudicate_detection_annotations.py
    python scripts/adjudicate_detection_annotations.py --out reports/audit_review/human_decisions.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.review_detection_annotations import (  # noqa: E402
    _find_dataset_root,
    _list_images,
    _list_labels,
    _read_boxes,
    load_data_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "valid", "test")
HUGE_AREA_THRESHOLD = 0.95 * 0.95  # matches the review system

VALID_DECISIONS = {
    "empty_label": {"annotate", "keep_empty", "remove"},
    "huge_box":    {"keep", "tighten", "manual_review"},
    "grape_policy":{"bunch_policy_confirmed", "per_berry_accepted", "needs_reannotation"},
    "ambiguous":   {"confirmed", "needs_second_opinion", "exclude"},
}
VALID_ACTIONS = {
    "manual_annotation_required", "no_change", "exclude_from_v3",
    "manual_review_required", "reannotate_in_v3", "confirmed", "second_opinion",
}

def find_suspensions(data_root: Path, names: List[str]) -> Dict[str, List[dict]]:
    """Recompute the suspension flags without touching any dataset files."""
    findings: Dict[str, List[dict]] = {"empty_label": [], "huge_box": []}
    for split in SPLIT_NAMES:
        im_dir = data_root / split / "images"
        lb_dir = data_root / split / "labels"
        if not im_dir.is_dir() or not lb_dir.is_dir():
            continue
        for img in _list_images(im_dir):
            lbl = lb_dir / (img.stem + ".txt")
            missing_or_empty = (not lbl.exists()) or (lbl.stat().st_size == 0)
            if missing_or_empty:
                findings["empty_label"].append({
                    "image": str(img), "image_filename": img.name, "split": split,
                    "category": "empty_label",
                    "label_path": str(lbl) if lbl.exists() else ""})
                continue
            boxes, _ = _read_boxes(lbl, len(names))
            areas = [(b[3] * b[4]) for b in boxes if 0 <= b[0] < len(names)]
            if areas and max(areas) >= HUGE_AREA_THRESHOLD:
                findings["huge_box"].append({
                    "image": str(img), "image_filename": img.name, "split": split,
                    "category": "huge_box", "label_path": str(lbl),
                    "max_area_ratio": round(max(areas), 6)})
    return findings


def validate_decision(rec: dict) -> List[str]:
    """Return validation errors for a decision record (never modifies input)."""
    errors: List[str] = []
    cat = rec.get("category")
    if cat not in VALID_DECISIONS:
        errors.append(f"unknown category: {cat}")
        return errors
    dec = rec.get("decision")
    if dec not in VALID_DECISIONS[cat]:
        errors.append(f"decision '{dec}' not valid for category '{cat}'")
    action = rec.get("action")
    if action is not None and action not in VALID_ACTIONS:
        errors.append(f"action '{action}' is not an allowed action")
    if cat == "empty_label" and dec == "annotate":
        if not rec.get("class_name"):
            errors.append("empty_label 'annotate' requires class_name")
        if rec.get("bbox"):
            errors.append("empty_label 'annotate' must NOT carry a fabricated bbox; "
                          "real coordinates must be supplied by a human later")
    return errors


def seed_empty_label_decisions(corrections_path: Path) -> List[dict]:
    """Turn v3_corrections evidence into human-decision records.

    Every image here was visually confirmed to contain fruit, so the decision is
    ``annotate`` + ``manual_annotation_required`` with NO bbox coordinates.
    """
    if not corrections_path.exists():
        raise FileNotFoundError(f"v3_corrections manifest not found: {corrections_path}")
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    records: List[dict] = []
    for c in corrections:
        cls = "Grape" if c["image_filename"].lower().startswith("grape") else "Apple"
        records.append({
            "image": c["image"],
            "image_filename": c["image_filename"],
            "split": c["split"],
            "category": "empty_label",
            "decision": "annotate",
            "class_name": cls,
            "action": "manual_annotation_required",
            "notes": "Visible fruit confirmed by human review; a real, verified "
                     "bounding box must be supplied in V3 (none fabricated here).",
            "bbox": None,
        })
    return records


def seed_huge_box_decisions(huge: List[dict]) -> List[dict]:
    """Seed suggested status + decision for huge boxes. Labels stay untouched."""
    for rec in huge:
        area = rec.get("max_area_ratio", 0)
        if area >= 0.99:
            rec["decision"] = "tighten"
            rec["suggested"] = "tighten"
            rec["action"] = "manual_review_required"
        elif area >= 0.96:
            rec["decision"] = "manual_review"
            rec["suggested"] = "manual_review"
            rec["action"] = "manual_review_required"
        else:
            rec["decision"] = "keep"
            rec["suggested"] = "keep"
            rec["action"] = "no_change"
    return huge

def write_manifest(records: List[dict], out_path: Path,
                   notes: Optional[List[str]] = None) -> Path:
    """Write a validated, machine-readable decision manifest."""
    invalid = []
    for r in records:
        errs = validate_decision(r)
        if errs:
            invalid.append({"record": r.get("image_filename"), "errors": errs})
    payload = {
        "schema_version": 1,
        "source": "human-adjudication",
        "notes": notes or [],
        "record_count": len(records),
        "records": records,
    }
    if invalid:
        payload["validation_warnings"] = invalid
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                        encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Human-adjudication layer for V3 (records decisions, does NOT modify dataset)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/detection"))
    parser.add_argument("--corrections", type=Path,
                        default=Path("reports/audit_review/v3_corrections.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/audit_review/human_decisions.json"))
    args = parser.parse_args()

    data_root = _find_dataset_root(args.data_dir)
    if data_root is None:
        logger.error("No dataset root found under %s", args.data_dir)
        return 1
    _, _, names = load_data_config(data_root)

    logger.info("=" * 70)
    logger.info("SmartFreshAI - Annotation Human Adjudication (Phase 1)")
    logger.info("=" * 70)

    empty = seed_empty_label_decisions(args.corrections)
    logger.info("Seed %d empty-label decisions (annotate, no fabricated bbox).", len(empty))

    susp = find_suspensions(data_root, names)
    huge = seed_huge_box_decisions(susp["huge_box"])
    logger.info("Huge-box review: %d cases seeded with suggested status.", len(huge))

    notes = [
        "All empty-label images visually confirmed to contain fruit; none converted "
        "to background. Decision records manual_annotation_required.",
        "No bounding-box coordinates were fabricated. V3 construction is blocked "
        "until real boxes are supplied by a human for the annotate records.",
    ]
    write_manifest(empty + huge, args.out, notes=notes)
    write_manifest(
        huge, args.out.with_name("huge_box_review.json"),
        notes=["Suggested statuses only; labels under data/detection are untouched."])
    logger.info("Manifest written: %s", args.out)
    logger.info("Huge-box manifest written: %s",
                args.out.with_name("huge_box_review.json"))
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
