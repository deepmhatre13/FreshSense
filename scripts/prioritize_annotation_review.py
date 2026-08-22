#!/usr/bin/env python3
"""Phase 3.6 — Review prioritization and human-adjudication UX.

Analyse each unresolved review image and compute an evidence-based **review
priority** (review *order*, NOT a correctness score). Every signal below is
derived from measurable, reproducible comparisons between ground-truth labels
and AI proposals.

Review-order score semantics
----------------------------
- HIGH  : strong evidence of a discrepancy that a human must inspect first
          (class disagreement, missing/extra detection, very poor box overlap,
          multiple conflicting classes, unusual object count, Grape policy
          conflict, zero AI proposal for a labelled object, etc.).
- MEDIUM: moderate disagreement, moderate IoU mismatch, moderate confidence.
- LOW   : strong agreement between GT and AI, high confidence, good box
          overlap, same class.

No image is automatically approved. HIGH agreement between GT and AI proposals
never implies the annotation is correct — it only lowers review priority.

Outputs
-------
- reports/audit_review/review_priority.json  (machine readable)
- reports/audit_review/review_priority.md    (human readable)

Usage:
    python scripts/prioritize_annotation_review.py [--data-dir data/detection]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repository root is in the Python path (allows `python scripts/...py`
# and `python -m scripts.prioritize_annotation_review` and imports from tests).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.audit_detection_dataset import (  # noqa: E402
    _find_dataset_root,
    _read_boxes,
    load_data_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REVIEW_DIR = Path("reports/audit_review")
PROPOSAL_FILE = REVIEW_DIR / "ai_annotation_proposals" / "proposals.json"
HUMAN_DECISIONS_FILE = REVIEW_DIR / "human_decisions.json"
REVIEW_CATEGORIES = ["ambiguous_classes", "tiny_boxes", "many_objects", "empty_labels", "huge_box", "huge_boxes"]

PRIORITY_JSON = REVIEW_DIR / "review_priority.json"
PRIORITY_MD = REVIEW_DIR / "review_priority.md"

# --- Grape policy -----------------------------------------------------------
GRAPE_CLASS_NAME = "Grape"
# If an image's AI proposals are mostly Grape and there are many small boxes
# that look like individual berries rather than bunches, flag a policy conflict.
GRAPE_TINY_BOX_AREA = 0.005          # berry-sized box (< 0.5% of image area)
GRAPE_MIN_BOXES_FOR_CONFLICT = 8     # lots of small grape boxes -> likely berries


# --- Box helpers ------------------------------------------------------------
def box_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """Area of an axis-aligned box. Non-negative."""
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = box_area(ix1, iy1, ix2, iy2)
    union = box_area(ax1, ay1, ax2, ay2) + box_area(bx1, by1, bx2, by2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def yolo_to_xyxy(cls_id: int, cx: float, cy: float, w: float, h: float, img_w: float, img_h: float):
    """Convert a normalized YOLO box to pixel (x1, y1, x2, y2)."""
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return x1, y1, x2, y2


# --- matching --------------------------------------------------------------

def match_boxes(
    gt_boxes: List[dict],
    ai_boxes: List[dict],
    iou_threshold: float = 0.5,
) -> Dict[str, object]:
    """Greedy match ground-truth boxes to AI proposal boxes by IoU.

    ``gt_boxes``/``ai_boxes`` are dicts with pixel ``x1/y1/x2/y2`` and
    ``class_name``. Returns a dict with matched pairs, IoU stats, unmatched
    GT and unmatched AI boxes.
    """
    gt_matched = [False] * len(gt_boxes)
    ai_matched = [False] * len(ai_boxes)
    pairs: List[dict] = []
    # Greedy: iterate GT boxes, pick the best unmatched AI box with IoU >= threshold.
    for g_idx, g in enumerate(gt_boxes):
        best_i = -1
        best_iou = 0.0
        g_box = (g["x1"], g["y1"], g["x2"], g["y2"])
        for a_idx, a in enumerate(ai_boxes):
            if ai_matched[a_idx]:
                continue
            ov = iou(g_box, (a["x1"], a["y1"], a["x2"], a["y2"]))
            if ov > best_iou:
                best_iou = ov
                best_i = a_idx
        if best_i >= 0 and best_iou >= iou_threshold:
            gt_matched[g["_idx"]] = True
            ai_matched[best_i] = True
            pairs.append({
                "gt_box": g,
                "ai_box": ai_boxes[best_i],
                "iou": best_iou,
                "class_disagreement": g["class_name"] != ai_boxes[best_i]["class_name"],
            })
    unmatched_gt = [g for g, m in zip(gt_boxes, gt_matched) if not m]
    unmatched_ai = [a for a, m in zip(ai_boxes, ai_matched) if not m]
    return {
        "pairs": pairs,
        "unmatched_gt": unmatched_gt,
        "unmatched_ai": unmatched_ai,
    }


def _decorate_gt_with_index(gt_boxes: List[dict]) -> None:
    for i, g in enumerate(gt_boxes):
        g["_idx"] = i
# --- analysis -----------------------------------------------------------------

def load_review_categories() -> Dict[str, List[str]]:
    """Map image filename -> sorted list of review categories it appears in."""
    result: Dict[str, List[str]] = defaultdict(list)
    for cat in REVIEW_CATEGORIES:
        p = REVIEW_DIR / f"{cat}_review.json"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("records", [])
        for item in data:
            fname = item.get("image_filename") or (Path(item.get("image", "")).name or None)
            if not fname:
                continue
            item_cat = item.get("category", cat)
            result[fname].append(item_cat)
    return {k: sorted(set(v)) for k, v in result.items()}


def load_proposals_by_image() -> Dict[str, List[dict]]:
    """Map image filename -> list of AI proposal dicts (pixel coords already)."""
    if not PROPOSAL_FILE.exists():
        return {}
    with open(PROPOSAL_FILE, "r", encoding="utf-8") as f:
        proposals = json.load(f)
    by_img: Dict[str, List[dict]] = defaultdict(list)
    for p in proposals:
        by_img[Path(p["image"]).name].append(p)
    return dict(by_img)


def load_decided_images() -> set:
    """Images removed from the unresolved set by a *real* human decision.

    AI-proposal records marked ``uncertain`` or ``pending`` do NOT resolve an
    image (they remain unresolved). Only accepted / corrected / kept /
    rejected / excluded records count as resolved.
    """
    _REAL = {"accepted", "corrected", "kept", "rejected", "excluded"}
    if not HUMAN_DECISIONS_FILE.exists():
        return set()
    with open(HUMAN_DECISIONS_FILE, "r", encoding="utf-8") as f:
        hd = json.load(f)
    decided = set()
    for rec in hd.get("records", []):
        if rec.get("human_decision") not in _REAL:
            continue
        fname = None
        if rec.get("image_filename"):
            fname = rec["image_filename"]
        elif rec.get("ai_proposal") and rec["ai_proposal"].get("image"):
            fname = Path(rec["ai_proposal"]["image"]).name
        elif rec.get("image"):
            fname = Path(rec["image"]).name
        if fname:
            decided.add(fname)
    return decided


def read_gt_boxes(label_path: Path, class_names: List[str]) -> List[dict]:
    """Read ground-truth boxes (normalized YOLO) for one label file."""
    if not label_path.exists():
        return []
    boxes, _issues = _read_boxes(label_path, len(class_names))
    return [{
        "class_id": b[0],
        "class_name": class_names[b[0]] if b[0] < len(class_names) else str(b[0]),
        "cx": b[1], "cy": b[2], "w": b[3], "h": b[4],
    } for b in boxes if b[3] > 0 and b[4] > 0]
# --- per-image evidence -------------------------------------------------------

def analyse_image(
    fname: str,
    split: str,
    categories: List[str],
    gt_boxes_norm: List[dict],
    ai_proposals: List[dict],
    img_w: float,
    img_h: float,
    class_names: List[str],
) -> dict:
    """Compute the full evidence bundle for one image.

    ``gt_boxes_norm`` are normalized YOLO boxes (dicts with class_id/cx/cy/w/h);
    ``ai_proposals`` are the raw proposal dicts (pixel x1/y1/x2/y2, class_name).
    """
    # Ground-truth boxes -> pixel coords
    gt_boxes: List[dict] = []
    for g in gt_boxes_norm:
        x1, y1, x2, y2 = yolo_to_xyxy(g["class_id"], g["cx"], g["cy"], g["w"], g["h"], img_w, img_h)
        gt_boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class_id": g["class_id"],
            "class_name": class_names[g["class_id"]] if g["class_id"] < len(class_names) else str(g["class_id"]),
        })
    _decorate_gt_with_index(gt_boxes)

    # AI proposal boxes (already pixel coords)
    ai_boxes: List[dict] = []
    for p in ai_proposals:
        ai_boxes.append({
            "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"],
            "class_id": p.get("class_id"),
            "class_name": p.get("class_name"),
            "confidence": p.get("confidence", 0.0),
            "proposal_id": p.get("proposal_id"),
        })

    matched = match_boxes(gt_boxes, ai_boxes)

    gt_classes = sorted({g["class_name"] for g in gt_boxes})
    ai_classes = sorted({a["class_name"] for a in ai_boxes})

    confidences = [a.get("confidence", 0.0) for a in ai_boxes]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    min_conf = min(confidences) if confidences else 0.0
    max_conf = max(confidences) if confidences else 0.0

    pair_ious = [pair["iou"] for pair in matched["pairs"]]
    mean_iou = sum(pair_ious) / len(pair_ious) if pair_ious else 0.0

    class_disagreements = [p for p in matched["pairs"] if p["class_disagreement"]]
    conflict_classes = sorted(
        {p["gt_box"]["class_name"] for p in class_disagreements}
        | {p["ai_box"]["class_name"] for p in class_disagreements}
    )

    # Grape policy conflict: image is Grape-heavy and AI proposes many tiny boxes.
    area = img_w * img_h
    grape_ai = [a for a in ai_boxes if a["class_name"] == GRAPE_CLASS_NAME]
    tiny_grape = [
        a for a in grape_ai
        if area > 0 and box_area(a["x1"], a["y1"], a["x2"], a["y2"]) / area < GRAPE_TINY_BOX_AREA
    ]
    grape_policy_conflict = bool(grape_ai and len(tiny_grape) >= GRAPE_MIN_BOXES_FOR_CONFLICT)

    return {
        "image_filename": fname,
        "image": ai_proposals[0]["image"] if ai_proposals else f"data/detection/{split}/images/{fname}",
        "split": split,
        "review_categories": categories,
        "priority_category": categories[0] if categories else "none",
        "gt_object_count": len(gt_boxes),
        "ai_proposal_count": len(ai_boxes),
        "gt_classes": gt_classes,
        "proposed_classes": ai_classes,
        "class_disagreements": len(class_disagreements),
        "conflict_classes": conflict_classes,
        "missing_detections": len(matched["unmatched_gt"]),
        "extra_detections": len(matched["unmatched_ai"]),
        "confidence_mean": round(avg_conf, 4),
        "confidence_min": round(min_conf, 4),
        "confidence_max": round(max_conf, 4),
        "mean_iou": round(mean_iou, 4),
        "matched_pairs": len(matched["pairs"]),
        "unmatched_gt_boxes": [g["class_name"] for g in matched["unmatched_gt"]],
        "unmatched_ai_boxes": [a["class_name"] for a in matched["unmatched_ai"]],
        "grape_policy_conflict": grape_policy_conflict,
        "flags": [],
    }
# --- review-order score ------------------------------------------------------
#
# The score is a REVIEW-ORDER score, NOT a correctness score. A higher score
# means "inspect this image sooner". It is built from the measurable evidence
# bundle in ``analyse_image``. Each component adds points; the weighted total is
# clamped to [0, 100] and mapped to HIGH / MEDIUM / LOW.
#
# Weights (documented formula):
#   base                      : from review category presence
#   class_disagreements       : +18 per disagreed matched pair (cap 3) -> 54
#   missing_detections        : +15 per unmatched GT object (cap 4)   -> 60
#   extra_detections          : +8  per unmatched AI box    (cap 6)   -> 48
#   low overlap (mean_iou)    : +25 when < 0.30, +12 when < 0.50
#   conflicting classes       : +20 when GT and AI class sets differ
#   unusual object count      : +20 when GT count >= many-objects cap
#                              or AI count >= 2 * GT count (+ GT > 0)
#   low confidence            : +15 when mean confidence < 0.50
#   grape policy conflict     : +30 (highest single flag)
#
# Level mapping:
#   HIGH   : score >= 40
#   MEDIUM : 15 <= score < 40
#   LOW    : score < 15

_MANY_OBJECTS_CAP = 60
_IOU_MATCH_THRESHOLD = 0.5


def _category_base(categories: List[str]) -> float:
    """Baseline priority derived from the review categories an image belongs to."""
    if "ambiguous_classes" in categories:
        return 25.0
    if "empty_labels" in categories:
        return 20.0
    if "tiny_boxes" in categories or "many_objects" in categories:
        return 15.0
    if "huge_box" in categories or "huge_boxes" in categories:
        return 10.0
    return 0.0


def compute_review_score(ev: dict) -> dict:
    """Compute the review-order score for one evidence bundle.

    Returns ``{"score": float, "level": str, "reasons": [str]}``. The reasons
    list records *which* measurable signals pushed the score up so the output is
    auditable.
    """
    score = 0.0
    reasons: List[str] = []

    base = _category_base(ev.get("review_categories", []))
    score += base
    if base:
        reasons.append(f"category:{ev.get('priority_category')}")

    n_disc = min(ev["class_disagreements"], 3)
    if n_disc:
        score += 18 * n_disc
        reasons.append(f"class_disagreements={ev['class_disagreements']}")

    n_miss = min(ev["missing_detections"], 4)
    if n_miss:
        score += 15 * n_miss
        reasons.append(f"missing_detections={ev['missing_detections']}")

    n_extra = min(ev["extra_detections"], 6)
    if n_extra:
        score += 8 * n_extra
        reasons.append(f"extra_detections={ev['extra_detections']}")

    mean_iou = ev["mean_iou"]
    if ev["matched_pairs"] > 0 and mean_iou < 0.30:
        score += 25
        reasons.append(f"very_poor_overlap(iou={mean_iou})")
    elif ev["matched_pairs"] > 0 and mean_iou < 0.50:
        score += 12
        reasons.append(f"poor_overlap(iou={mean_iou})")

    # Conflicting class sets (GT classes vs proposed classes differ)
    if set(ev["gt_classes"]) != set(ev["proposed_classes"]):
        score += 20
        reasons.append("conflicting_classes")

    # Unusual object count
    gt = ev["gt_object_count"]
    ai = ev["ai_proposal_count"]
    if gt >= _MANY_OBJECTS_CAP or (gt > 0 and ai >= 2 * gt):
        score += 10
        reasons.append(f"unusual_object_count(gt={gt},ai={ai})")

    # Low confidence
    if ev["ai_proposal_count"] > 0 and ev["confidence_mean"] < 0.50:
        score += 15
        reasons.append(f"low_confidence(mean={ev['confidence_mean']})")

    if ev["grape_policy_conflict"]:
        score += 30
        reasons.append("grape_policy_conflict")

    score = max(0.0, min(100.0, score))
    if score >= 40:
        level = "HIGH"
    elif score >= 15:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {"score": round(score, 2), "level": level, "reasons": reasons}
# --- orchestration ------------------------------------------------------------

def image_dimensions(image_path: Path) -> Tuple[float, float]:
    """Return (width, height) of an image, or (1, 1) if unreadable."""
    import cv2  # local import keeps module importable without opencv in tests
    img = cv2.imread(str(image_path))
    if img is None:
        return 1.0, 1.0
    h, w = img.shape[:2]
    return float(w), float(h)


def collect_evidence(
    data_root: Path,
    class_names: List[str],
) -> List[dict]:
    """Build the evidence bundle for every unresolved image.

    Unresolved = an image that appears in a review-category file, has not been
    resolved by a real human decision, and has at least one AI proposal.
    """
    rev = load_review_categories()
    proposals_by_img = load_proposals_by_image()
    decided = load_decided_images()

    evidence: List[dict] = []
    for fname, categories in rev.items():
        if fname in decided:
            continue
        ai_proposals = proposals_by_img.get(fname)
        if not ai_proposals:
            continue  # no AI proposals -> cannot compare; leave pending
        split = ai_proposals[0].get("split", "train")
        img_path = Path(_REPO_ROOT) / ai_proposals[0]["image"]
        label_path = data_root / split / "labels" / (Path(fname).stem + ".txt")
        gt_norm = read_gt_boxes(label_path, class_names)
        img_w, img_h = image_dimensions(img_path)
        ev = analyse_image(
            fname, split, rev[fname], gt_norm, ai_proposals,
            img_w, img_h, class_names,
        )
        scoring = compute_review_score(ev)
        ev["review_score"] = scoring["score"]
        ev["review_priority"] = scoring["level"]
        ev["reasons"] = scoring["reasons"]
        ev["flags"] = _compile_flags(ev)
        evidence.append(ev)
    return evidence


def _compile_flags(ev: dict) -> List[str]:
    flags: List[str] = []
    if ev["class_disagreements"] > 0:
        flags.append("class_disagreement")
    if ev["missing_detections"] > 0:
        flags.append("missing_detection")
    if ev["extra_detections"] > 0:
        flags.append("extra_detection")
    if ev["grape_policy_conflict"]:
        flags.append("grape_policy_conflict")
    if set(ev["gt_classes"]) != set(ev["proposed_classes"]):
        flags.append("conflicting_classes")
    if ev["ai_proposal_count"] > 0 and ev["confidence_mean"] < 0.5:
        flags.append("low_confidence")
    if ev["gt_object_count"] >= _MANY_OBJECTS_CAP or (
        ev["gt_object_count"] > 0 and ev["ai_proposal_count"] >= 2 * ev["gt_object_count"]
    ):
        flags.append("unusual_object_count")
    if ev["matched_pairs"] > 0 and ev["mean_iou"] < 0.5:
        flags.append("poor_box_overlap")
    if ev["ai_proposal_count"] == 0:
        flags.append("no_ai_proposal")
    if ev["gt_object_count"] == 0 and ev["ai_proposal_count"] > 0:
        flags.append("ai_detects_no_gt_object")
    return flags
# --- reports ------------------------------------------------------------------

def _class_confusion_counts(evidence: List[dict]) -> Counter:
    """Count (gt_class -> ai_class) disagreements across matched pairs."""
    counter: Counter = Counter()
    for ev in evidence:
        # Recompute conflicts from reason / conflict_classes evidence:
        if ev["class_disagreements"] and ev["conflict_classes"]:
            counter[tuple(ev["conflict_classes"])] += ev["class_disagreements"]
    return counter


def generate_reports(evidence: List[dict], data_root: Path) -> dict:
    """Persist review_priority.json and review_priority.md; return summary."""
    total = len(evidence)
    high = [e for e in evidence if e["review_priority"] == "HIGH"]
    med = [e for e in evidence if e["review_priority"] == "MEDIUM"]
    low = [e for e in evidence if e["review_priority"] == "LOW"]

    high_sorted = sorted(high, key=lambda e: e["review_score"], reverse=True)
    by_score = sorted(evidence, key=lambda e: e["review_score"], reverse=True)
    top20 = by_score[:20]

    cat_counter = Counter()
    for e in evidence:
        for c in e["review_categories"]:
            cat_counter[c] += 1

    flag_counter = Counter()
    for e in evidence:
        for fl in e["flags"]:
            flag_counter[fl] += 1

    confusions = _class_confusion_counts(evidence)

    avg_conf = (
        sum(e["confidence_mean"] for e in evidence if e["ai_proposal_count"] > 0)
        / max(1, sum(1 for e in evidence if e["ai_proposal_count"] > 0))
    )
    # Agreement stat: share of matched GT boxes that agree on class + IoU.
    agree_gt = sum(e["matched_pairs"] for e in evidence)
    agree_class = sum(e["matched_pairs"] - e["class_disagreements"] for e in evidence)
    agree_ratio = agree_class / max(1, agree_gt)

    needs_drawing = [e for e in evidence if e["gt_object_count"] == 0 or e["missing_detections"] > 0]
    # "highly consistent": LOW priority (strong agreement, high confidence, good overlap).
    consistent = low

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_images": total,
        "high_count": len(high),
        "medium_count": len(med),
        "low_count": len(low),
        "category_counts": dict(cat_counter),
        "flag_counts": dict(flag_counter),
        "top20": [{
            "image_filename": e["image_filename"],
            "score": e["review_score"],
            "priority": e["review_priority"],
            "category": e["priority_category"],
            "gt_count": e["gt_object_count"],
            "ai_count": e["ai_proposal_count"],
            "class_disagreements": e["class_disagreements"],
            "missing": e["missing_detections"],
            "extra": e["extra_detections"],
            "mean_iou": e["mean_iou"],
            "reasons": e["reasons"],
        } for e in top20],
        "class_confusion_counts": [{"gt_vs_ai": list(k), "count": v} for k, v in confusions.items()],
        "average_confidence": round(avg_conf, 4),
        "agreement": {
            "matched_pairs_total": agree_gt,
            "matched_pairs_agreeing_class": agree_class,
            "gt_ai_class_agreement_ratio": round(agree_ratio, 4),
        },
        "needs_box_drawing_images": len(needs_drawing),
        "highly_consistent_images": len(consistent),
        "scoring_formula": {
            "note": "REVIEW-ORDER score, NOT a correctness score.",
            "high_threshold": 40,
            "medium_threshold": 15,
            "many_objects_cap": _MANY_OBJECTS_CAP,
            "iou_match_threshold": _IOU_MATCH_THRESHOLD,
            "grape_tiny_box_area": GRAPE_TINY_BOX_AREA,
            "grape_min_boxes_for_conflict": GRAPE_MIN_BOXES_FOR_CONFLICT,
        },
    }

    PRIORITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PRIORITY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(PRIORITY_MD, "w", encoding="utf-8") as f:
        f.write(_render_markdown(summary, evidence))
    logger.info("Wrote %s and %s", PRIORITY_JSON, PRIORITY_MD)
    return summary
# --- markdown -----------------------------------------------------------------

def _render_markdown(summary: dict, evidence: List[dict]) -> str:
    lines = []
    lines.append("# Annotation Review Priority")
    lines.append("")
    lines.append("> **REVIEW-ORDER score** — it decides *which image to inspect first*, NOT whether an annotation is correct.")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append(f"- Total unresolved images: **{summary['total_images']}**")
    lines.append(f"- HIGH: **{summary['high_count']}**")
    lines.append(f"- MEDIUM: **{summary['medium_count']}**")
    lines.append(f"- LOW: **{summary['low_count']}**")
    lines.append("")
    lines.append("## Category counts")
    lines.append("")
    for cat, cnt in sorted(summary["category_counts"].items()):
        lines.append(f"- {cat}: {cnt}")
    lines.append("")
    lines.append("## Flag counts")
    lines.append("")
    for fl, cnt in sorted(summary["flag_counts"].items()):
        lines.append(f"- {fl}: {cnt}")
    lines.append("")
    lines.append("## Top 20 highest-priority images")
    lines.append("")
    lines.append("| Rank | Image | Score | Priority | Category | GT | AI | Disc | Missing | Extra | IoU |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, e in enumerate(summary["top20"], start=1):
        lines.append(
            f"| {i} | {e['image_filename']} | {e['score']} | {e['priority']} "
            f"| {e['category']} | {e['gt_count']} | {e['ai_count']} | "
            f"{e['class_disagreements']} | {e['missing']} | {e['extra']} | {e['mean_iou']} |"
        )
    lines.append("")
    lines.append("## Class-confusion counts (GT vs AI, matched boxes)")
    lines.append("")
    if summary["class_confusion_counts"]:
        for cc in summary["class_confusion_counts"]:
            lines.append(f"- {cc['gt_vs_ai']}: {cc['count']}")
    else:
        lines.append("_None._")
    lines.append("")
    lines.append("## Agreement statistics")
    lines.append("")
    lines.append(f"- Average AI confidence: **{summary['average_confidence']}**")
    lines.append(f"- Matched GT/AI pairs: **{summary['agreement']['matched_pairs_total']}**")
    lines.append(f"- Pairs agreeing on class: **{summary['agreement']['matched_pairs_agreeing_class']}**")
    lines.append(f"- GT/AI class-agreement ratio: **{summary['agreement']['gt_ai_class_agreement_ratio']}**")
    lines.append("")
    lines.append("## Estimated workload")
    lines.append("")
    lines.append(f"- Images likely requiring actual box drawing (no/partial GT): **{summary['needs_box_drawing_images']}**")
    lines.append(f"- Images where AI proposals look highly consistent (LOW priority): **{summary['highly_consistent_images']}**")
    lines.append("")
    lines.append("> High agreement does **not** mean the annotation is correct; it only lowers review priority.")
    lines.append("")
    return "\n".join(lines) + "\n"
# --- CLI ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Prioritize human review of AI annotation proposals.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/detection"), help="Detection dataset root.")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Phase 3.6 - Annotation Review Prioritization (read-only)")
    logger.info("=" * 70)

    data_root = _find_dataset_root(args.data_dir)
    if data_root is None:
        logger.error("Dataset root not found under %s", args.data_dir)
        return 1
    _, _, class_names = load_data_config(data_root)

    evidence = collect_evidence(data_root, class_names)
    logger.info("Analysed %d unresolved images.", len(evidence))

    summary = generate_reports(evidence, data_root)

    logger.info("=" * 70)
    logger.info("Priority summary: %d total | HIGH %d | MEDIUM %d | LOW %d",
                summary["total_images"], summary["high_count"],
                summary["medium_count"], summary["low_count"])
    logger.info("Average AI confidence: %.4f", summary["average_confidence"])
    logger.info("Images likely requiring box drawing: %d", summary["needs_box_drawing_images"])
    logger.info("Images where AI looks highly consistent: %d", summary["highly_consistent_images"])
    logger.info("V3 remains BLOCKED - no dataset or model was modified.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())