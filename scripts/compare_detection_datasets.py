#!/usr/bin/env python3
"""Compare the V2 and V3 SmartFreshAI detection datasets.

Produces a side-by-side, evidence-driven comparison of the frozen V2 dataset
(``data/detection``) and the V3 dataset (``data/detection_v3``):

    reports/detection_v2_vs_v3.json
    reports/detection_v2_vs_v3.md

Each metric is explicitly classified as ``IMPROVED``, ``UNCHANGED`` or
``REGRESSED`` relative to V2 — regressions are never hidden.

If ``data/detection_v3`` does not yet exist (the V3 review gate is still
blocked), the script writes a status summary instead of fabricating a
comparison.

This tool is read-only: it never modifies either dataset or ``best.pt``.

Usage:
    python scripts/compare_detection_datasets.py
    python scripts/compare_detection_datasets.py --v2 data/detection --v3 data/detection_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.audit_detection_dataset import (  # noqa: E402
    SPLIT_NAMES,
    _find_dataset_root,
    _list_images,
    _list_labels,
    _read_boxes,
    build_metadata,
    collect_split_data,
    compute_bbox_stats,
    load_data_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_V2 = Path("data/detection")
DEFAULT_V3 = Path("data/detection_v3")
DEFAULT_JSON_OUT = Path("reports/detection_v2_vs_v3.json")
DEFAULT_MD_OUT = Path("reports/detection_v2_vs_v3.md")


def _split_stats(data_root: Path, names: list, nc: int) -> dict:
    """Compute per-split statistics for one dataset using the V2 audit helpers."""
    root = _find_dataset_root(data_root) or Path(data_root)
    paths = [p for s in SPLIT_NAMES for p in _list_images(root / s / "images")]
    metadata = _minimal_metadata(paths)
    per_split = {}
    all_boxes = []
    for s in SPLIT_NAMES:
        im_dir = root / s / "images"
        lb_dir = root / s / "labels"
        if not im_dir.is_dir():
            per_split[s] = {"images": 0, "total_objects": 0, "objects_per_image": [],
                            "per_class_instances": {}, "issues": []}
            continue
        data = collect_split_data(s, im_dir, lb_dir, nc, names, metadata)
        per_split[s] = data
        all_boxes.extend(data["boxes"])
    return {"root": str(root), "names": names, "nc": nc,
            "per_split": per_split, "all_boxes": all_boxes}


def _dataset_summary(stats: dict) -> dict:
    per_split = stats["per_split"]
    total_images = sum(s["images"] for s in per_split.values())
    total_objects = sum(s["total_objects"] for s in per_split.values())
    root = Path(stats["root"])
    label_counts = {s: len(_list_labels(root / s / "labels")) for s in SPLIT_NAMES}
    issues = sum(len(s["issues"]) for s in per_split.values())
    empty_labels = sum(len(s["images_without_label"]) for s in per_split.values())

    objs = [n for s in per_split.values() for n in s["objects_per_image"]]
    avg = sum(objs) / len(objs) if objs else 0
    median = sorted(objs)[len(objs) // 2] if objs else 0

    class_counts = {c: 0 for c in stats["names"]}
    for s in per_split.values():
        for cid, v in s["per_class_instances"].items():
            if int(cid) < len(stats["names"]):
                class_counts[stats["names"][int(cid)]] += v

    return {
        "total_images": total_images,
        "total_labels": sum(label_counts.values()),
        "total_objects": total_objects,
        "empty_labels": empty_labels,
        "issues": issues,
        "avg_objects_per_image": round(avg, 4),
        "median_objects_per_image": median,
        "per_class": class_counts,
        "split_counts": {s: per_split[s]["images"] for s in SPLIT_NAMES},
        "label_counts": label_counts,
    }


def _minimal_metadata(paths: list):
    """Lightweight image-dimension metadata (no perceptual hash) for speed.

    The comparison metrics do not need phash/duplicate detection, but
    ``collect_split_data`` still requires ``{w, h}`` entries so that box pixel
    dimensions and area ratios resolve correctly.
    """
    from PIL import Image
    meta: Dict[str, Dict] = {}
    for p in paths:
        key = str(p)
        try:
            with Image.open(p) as im:
                meta[key] = {"w": float(im.size[0]), "h": float(im.size[1])}
        except Exception:  # noqa: BLE001
            meta[key] = {"w": None, "h": None}
    return meta


def _box_metrics(all_boxes, nc, names):
    """Aggregate tiny/huge box statistics across all boxes.

    Box tuple from ``collect_split_data`` is ``(cls, cx, cy, w, h, wp, hp,
    area_ratio)``: area_ratio lives at index 7.
    """
    areas = [b[7] for b in all_boxes if 0 <= b[0] < nc]
    tiny = sum(1 for a in areas if a < 0.005)   # < 0.5% image area (audit policy)
    large = sum(1 for a in areas if a >= 0.9025)  # >= 0.95*0.95 (huge-box policy)
    return {"tiny_boxes": tiny, "large_boxes": large,
            "area_min": round(min(areas), 6) if areas else 0,
            "area_max": round(max(areas), 6) if areas else 0}
def _classify(delta: float, bigger_is_better: bool, tol: float = 1e-9) -> str:
    """Classify a delta as IMPROVED/UNCHANGED/REGRESSED for a 'bigger-is-better' metric."""
    if abs(delta) <= tol:
        return "UNCHANGED"
    improved = delta > 0 if bigger_is_better else delta < 0
    return "IMPROVED" if improved else "REGRESSED"


def compare_datasets(v2_root: Path, v3_root: Path, class_names: list, nc: int) -> dict:
    """Compare two (V2, V3) datasets and classify each metric.

    Returns None for the v3 summary if V3 does not exist.
    """
    v2_split = _split_stats(v2_root, class_names, nc)
    v2_stats = _dataset_summary(v2_split)
    v2_box = _box_metrics(v2_split["all_boxes"], nc, class_names)

    v3_path = _find_dataset_root(v3_root) or v3_root
    if not (v3_path / "data.yaml").is_file():
        return {
            "status": "v3_not_available",
            "v2_summary": v2_stats, "v2_box_metrics": v2_box,
            "v3_summary": None, "metrics": {},
        }
    v3_split = _split_stats(v3_root, class_names, nc)
    v3_stats = _dataset_summary(v3_split)
    v3_box = _box_metrics(v3_split["all_boxes"], nc, class_names)

    # Per-class representation comparison.
    per_class = {}
    for c in class_names:
        v2c = v2_stats["per_class"].get(c, 0)
        v3c = v3_stats["per_class"].get(c, 0)
        per_class[c] = {
            "v2": v2c, "v3": v3c, "delta": v3c - v2c,
            "classification": _classify(v3c - v2c, True),
        }

    metrics = {
        "image_counts": {
            "v2": v2_stats["total_images"], "v3": v3_stats["total_images"],
            "delta": v3_stats["total_images"] - v2_stats["total_images"],
            "classification": _classify(v3_stats["total_images"] - v2_stats["total_images"], True),
        },
        "annotation_counts": {
            "v2": v2_stats["total_objects"], "v3": v3_stats["total_objects"],
            "delta": v3_stats["total_objects"] - v2_stats["total_objects"],
            "classification": _classify(v3_stats["total_objects"] - v2_stats["total_objects"], True),
        },
        "empty_labels": {
            "v2": v2_stats["empty_labels"], "v3": v3_stats["empty_labels"],
            "delta": v3_stats["empty_labels"] - v2_stats["empty_labels"],
            "classification": _classify(v3_stats["empty_labels"] - v2_stats["empty_labels"], False),
        },
        "tiny_boxes": {
            "v2": v2_box["tiny_boxes"], "v3": v3_box["tiny_boxes"],
            "delta": v3_box["tiny_boxes"] - v2_box["tiny_boxes"],
            "classification": _classify(v3_box["tiny_boxes"] - v2_box["tiny_boxes"], False),
        },
        "huge_boxes": {
            "v2": v2_box["large_boxes"], "v3": v3_box["large_boxes"],
            "delta": v3_box["large_boxes"] - v2_box["large_boxes"],
            "classification": _classify(v3_box["large_boxes"] - v2_box["large_boxes"], False),
        },
        "avg_objects_per_image": {
            "v2": v2_stats["avg_objects_per_image"], "v3": v3_stats["avg_objects_per_image"],
            "delta": v3_stats["avg_objects_per_image"] - v2_stats["avg_objects_per_image"],
            "classification": _classify(v3_stats["avg_objects_per_image"] - v2_stats["avg_objects_per_image"], True),
        },
        "median_objects_per_image": {
            "v2": v2_stats["median_objects_per_image"], "v3": v3_stats["median_objects_per_image"],
            "delta": v2_stats["median_objects_per_image"],  # corrected below
            "classification": None,
        },
        "per_class_representation": per_class,
        "split_distribution": {
            "v2": v2_stats["split_counts"], "v3": v3_stats["split_counts"],
        },
    }
    # Fix median classification (symmetric metric).
    med_v2, med_v3 = v2_stats["median_objects_per_image"], v3_stats["median_objects_per_image"]
    metrics["median_objects_per_image"].update({
        "delta": med_v3 - med_v2,
        "classification": _classify(med_v3 - med_v2, True),
    })

    return {
        "status": "comparison_complete",
        "v2_summary": v2_stats,
        "v3_summary": v3_stats,
        "v2_box_metrics": v2_box,
        "v3_box_metrics": v3_box,
        "metrics": metrics,
    }


def _build_markdown(result: dict) -> str:
    if result.get("status") == "v3_not_available":
        m = []
        m.append("# SmartFreshAI V2 vs V3 Dataset Comparison")
        m.append("")
        m.append("> **V3 dataset not available.** `data/detection_v3` does not exist yet "
                 "(the V3 review gate is still blocked).")
        m.append("  No comparison was produced; nothing was fabricated. Build V3 first via "
                 "`python scripts/build_detection_v3.py`.")
        v2 = result["v2_summary"]
        m.append("")
        m.append("## V2 baseline (frozen)")
        m.append(f"- Total images: {v2['total_images']}")
        m.append(f"- Total objects: {v2['total_objects']}")
        m.append(f"- Empty-label images: {v2['empty_labels']}")
        m.append(f"- Avg objects/image: {v2['avg_objects_per_image']}")
        return "\n".join(m)

    v2, v3 = result["v2_summary"], result["v3_summary"]
    metrics = result["metrics"]
    m = []
    m.append("# SmartFreshAI V2 vs V3 Dataset Comparison")
    m.append("")
    m.append("| Metric | V2 | V3 | Delta | Verdict |")
    m.append("| --- | ---: | ---: | ---: | --- |")
    def row(name, v2v, v3v, key):
        mm = metrics[key]
        m.append(f"| {name} | {v2v} | {v3v} | {mm['delta']:+} | **{mm['classification']}** |")
    row("Total images", v2["total_images"], v3["total_images"], "image_counts")
    row("Total objects (annotations)", v2["total_objects"], v3["total_objects"], "annotation_counts")
    row("Empty-label images", v2["empty_labels"], v3["empty_labels"], "empty_labels")
    row("Tiny boxes (<0.5% area)", result["v2_box_metrics"]["tiny_boxes"],
        result["v3_box_metrics"]["tiny_boxes"], "tiny_boxes")
    row("Large/huge boxes (>=25%)", result["v2_box_metrics"]["large_boxes"],
        result["v3_box_metrics"]["large_boxes"], "huge_boxes")
    row("Avg objects/image", v2["avg_objects_per_image"], v3["avg_objects_per_image"],
        "avg_objects_per_image")
    row("Median objects/image", v2["median_objects_per_image"], v3["median_objects_per_image"],
        "median_objects_per_image")
    m.append("")
    m.append("## Per-class representation (V2 -> V3)")
    m.append("")
    m.append("| Class | V2 | V3 | Delta | Verdict |")
    m.append("| --- | ---: | ---: | ---: | --- |")
    for c, info in metrics["per_class_representation"].items():
        m.append(f"| {c} | {info['v2']} | {info['v3']} | {info['delta']:+} | **{info['classification']}** |")
    m.append("")
    m.append("## Split distribution")
    m.append("")
    m.append("| Split | V2 images | V3 images |")
    m.append("| --- | ---: | ---: |")
    for s in SPLIT_NAMES:
        m.append(f"| {s} | {v2['split_counts'].get(s, 0)} | {v3['split_counts'].get(s, 0)} |")
    m.append("")
    m.append("> Note: a higher overall annotation count does **not** imply a better "
             "dataset. Class-by-class and edge-case (tiny/huge/empty) regressions "
             "are reported explicitly above.")
    return "\n".join(m)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare V2 and V3 detection datasets")
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--v3", type=Path, default=DEFAULT_V3)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    v2 = _find_dataset_root(args.v2)
    if v2 is None or not (v2 / "data.yaml").is_file():
        logger.error("V2 dataset not found at %s", args.v2)
        return 1
    _, _, names = load_data_config(v2)
    nc = len(names)

    result = compare_datasets(_REPO_ROOT / args.v2, _REPO_ROOT / args.v3, names, nc)

    json_out = args.output if args.output.is_absolute() else _REPO_ROOT / args.output
    md_out = args.markdown if args.markdown.is_absolute() else _REPO_ROOT / args.markdown
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md_out.write_text(_build_markdown(result), encoding="utf-8")

    print(f"V2 vs V3 comparison written: {json_out}, {md_out}")
    print(f"  status: {result['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

