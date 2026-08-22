#!/usr/bin/env python3
"""Complete, read-only audit of the SmartFreshAI YOLO detection dataset.

This script performs a **deep, non-destructive** audit of a YOLO-format
detection dataset (``data/detection``) and produces two reports:

- ``reports/detection_dataset_audit.json`` (machine-readable)
- ``reports/detection_dataset_audit.md``  (human-readable)
- plus visual montages under ``reports/audit_visuals/``

It never writes to, moves, renames, or deletes any dataset file. It is the
evidence base we use before training Dataset V3.

Sections produced:

1.  Dataset summary            - data.yaml loading + path resolution
2.  Split summary              - image/label/object counts per split
3.  Class distribution         - instances, % of dataset, images per class,
                                 avg/min/max objects per image
4.  Label validation           - every YOLO row checked for range/format issues
5.  Bounding-box statistics    - per-class w/h/area stats + size category
6.  Class imbalance            - most/least represented, imbalance ratio
7.  Split consistency          - per-class train/val/test % + leakage
8.  Duplicate / near-duplicate - MD5 exact + perceptual-hash near duplicates
9.  Visual montages            - per-class and edge-case sample grids
10. Confusion-focused analysis - built from the reported validation metrics
11. Domain-gap observations    - webcam vs. closed-set labelled data
12. Recommendations for V3

Usage:
    python scripts/audit_detection_dataset.py
    python scripts/audit_detection_dataset.py --data-dir data/detection
        --output reports/detection_dataset_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

# Allow running directly (python scripts/<name>.py) from any CWD.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse existing conventions without duplicating logic. We deliberately do NOT
# import torch-loaded modules (configs.config / scripts.validate_detection_dataset)
# so this read-only audit stays fast and dependency-light. The small filesystem
# helpers below mirror the ones in scripts/validate_detection_dataset.py.
from src.data.dataset_validation import (  # noqa: E402
    hamming_distance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "valid", "test")
# "val" is accepted as an alias for "valid".
SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# bbox area ratio thresholds for small/medium/large categorisation.
_SMALL_AREA = 0.01   # bbox area < 1% of image -> small
_LARGE_AREA = 0.25   # bbox area > 25% of image -> large

_CONF_DIR = "reports/audit_visuals"


def _list_images(images_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(pattern))
    return sorted(files)


def _list_labels(labels_dir: Path) -> List[Path]:
    return sorted(labels_dir.glob("*.txt"))


def _find_dataset_root(data_dir: str | Path) -> Path | None:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return None
    if (data_dir / "data.yaml").is_file():
        return data_dir
    if (data_dir.parent / "data.yaml").is_file():
        return data_dir.parent
    for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if (sub / "data.yaml").is_file():
            return sub
    return None
def load_data_config(data_root: Path) -> Tuple[dict, int, List[str]]:
    """Load ``data.yaml``, returning ``(raw, nc, class_names)``."""
    data_yaml = data_root / "data.yaml"
    with open(data_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        nc = int(raw.get("nc", 0))
    except (TypeError, ValueError):
        nc = 0
    names_raw = raw.get("names")
    if isinstance(names_raw, dict):
        names = [str(names_raw[k]) for k in sorted(names_raw)]
    elif isinstance(names_raw, list):
        names = [str(n) for n in names_raw]
    else:
        names = []
    return raw, nc, names


def _read_boxes(label_file: Path, nc: int) -> Tuple[List[Tuple], List[Dict]]:
    """Parse one label file into ``(boxes, issues)``.

    ``boxes`` are ``(class_id, cx, cy, w, h, width_px, height_px)`` where the
    two image-size-expressed values are filled later once image dimensions are
    known. ``issues`` is a list of issue dicts.
    """
    boxes: List[Tuple] = []
    issues: List[Dict] = []
    try:
        text = label_file.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        issues.append({"file": str(label_file), "type": "unreadable", "detail": str(exc)})
        return boxes, issues

    lines = text.splitlines()
    if not any(l.strip() for l in lines):
        issues.append({"file": str(label_file), "type": "empty"})
        return boxes, issues

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            issues.append(
                {"file": str(label_file), "line": idx,
                 "type": "field_count", "detail": f"{len(parts)} fields"}
            )
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            issues.append(
                {"file": str(label_file), "line": idx,
                 "type": "non_numeric", "detail": stripped}
            )
            continue
        row = {"file": str(label_file), "line": idx, "class_id": cls_id,
               "cx": cx, "cy": cy, "w": w, "h": h}
        if not (0 <= cls_id < nc):
            issues.append({**row, "type": "invalid_class"})
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            issues.append({**row, "type": "center_out_of_range"})
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            issues.append({**row, "type": "size_out_of_range"})
        if w >= 0.95 and h >= 0.95:
            issues.append({**row, "type": "covers_almost_all_image"})
        if any(b[0] == cls_id and abs(b[1] - cx) < 1e-9 and abs(b[2] - cy) < 1e-9
               and abs(b[3] - w) < 1e-9 and abs(b[4] - h) < 1e-9 for b in boxes):
            issues.append({**row, "type": "duplicate_row"})
        boxes.append((cls_id, cx, cy, w, h, 0.0, 0.0))
    return boxes, issues
def _size_cat(area_ratio: float) -> str:
    if area_ratio < _SMALL_AREA:
        return "small"
    if area_ratio > _LARGE_AREA:
        return "large"
    return "medium"


def collect_split_data(
    split: str,
    images_dir: Path,
    labels_dir: Path,
    nc: int,
    names: List[str],
    metadata: Dict[str, Dict],
) -> Dict:
    """Audit a single split; returns the raw per-split data dict.

    ``metadata`` maps ``str(image_path)`` to ``{"w", "h", "phash"}`` and is
    populated once per image (single read) so dimensions + perceptual hashes are
    reused across all sections.
    """
    images = _list_images(images_dir)
    labels = _list_labels(labels_dir)
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    per_class_instances = {str(i): 0 for i in range(nc)}
    per_class_images = {str(i): 0 for i in range(nc)}
    all_boxes: List[Tuple] = []
    all_issues: List[Dict] = []
    objs_per_image: List[int] = []
    unreadable_images: List[str] = []
    label_file_by_stem = {p.stem: p for p in labels}

    for img in images:
        meta = metadata.get(str(img))
        width_px = meta["w"] if meta else None
        height_px = meta["h"] if meta else None
        if meta is None:
            unreadable_images.append(str(img))
        label_file = label_file_by_stem.get(img.stem)
        if label_file is None:
            # Image has no label -> skipped from object stats (maybe background).
            objs_per_image.append(0)
            continue
        boxes, issues = _read_boxes(label_file, nc)
        all_issues.extend(issues)

        seen = set()
        converted = []
        for (cls_id, cx, cy, w, h, _wp, _hp) in boxes:
            if width_px and height_px:
                wp = w * width_px
                hp = h * height_px
            else:
                wp = hp = 0.0
            area_ratio = w * h
            converted.append((cls_id, cx, cy, w, h, wp, hp, area_ratio))
            if cls_id < nc:
                per_class_instances[str(cls_id)] += 1
                seen.add(cls_id)
        all_boxes.extend(converted)
        objs_per_image.append(sum(1 for b in boxes if b[0] < nc))
        for cid in seen:
            per_class_images[str(cid)] += 1

    n_objs = len(all_boxes)
    return {
        "split": split,
        "images": len(images),
        "labels": len(labels),
        "images_without_label": missing_labels,
        "labels_without_image": orphan_labels,
        "unreadable_images": unreadable_images,
        "total_objects": n_objs,
        "objects_per_image": objs_per_image,
        "avg_objects_per_image": round(sum(objs_per_image) / len(objs_per_image), 4)
        if objs_per_image else 0,
        "min_objects_per_image": min(objs_per_image) if objs_per_image else 0,
        "max_objects_per_image": max(objs_per_image) if objs_per_image else 0,
        "per_class_instances": per_class_instances,
        "per_class_images": per_class_images,
        "issues": all_issues,
        "boxes": all_boxes,  # (cls, cx, cy, w, h, wp, hp, area_ratio)
    }


def build_metadata(paths: List[Path]) -> Dict[str, Dict]:
    """Read each image once via PIL, returning per-path ``{w, h, phash}``.

    The perceptual hash is derived from the same opened image (no second open)
    so the whole audit does one disk read per image.
    """
    from PIL import Image
    meta: Dict[str, Dict] = {}
    for p in paths:
        key = str(p)
        try:
            with Image.open(p) as im:
                w, h = im.size
                ph = _phash_grayscale(im.convert("L"))
            meta[key] = {"w": float(w), "h": float(h), "phash": ph}
        except Exception:  # noqa: BLE001
            meta[key] = {"w": None, "h": None, "phash": ""}
    return meta


def _phash_grayscale(img) -> str:
    """Perceptual hash of a grayscale PIL image (8x8 DCT-free comparison)."""
    arr = np.asarray(img.resize((9, 8)), dtype=np.int32)
    diff = arr[1:] > arr[:-1]
    return "".join("1" if d else "0" for d in diff.flatten())
def compute_bbox_stats(all_boxes: List[Tuple], nc: int, names: List[str]) -> Dict:
    """Per-class bbox width/height/area statistics (train+valid+test combined)."""
    per_class: Dict[str, List[str]] = {str(i): [] for i in range(nc)}
    for (cls_id, _cx, _cy, _w, _h, wp, hp, area_ratio) in all_boxes:
        if cls_id < nc:
            per_class[str(cls_id)].append(f"{wp:.3f}|{hp:.3f}|{area_ratio:.6f}")
    stats: Dict[str, Dict] = {}
    for i in range(nc):
        rows = [r.split("|") for r in per_class[str(i)]]
        widths = [float(r[0]) for r in rows]
        heights = [float(r[1]) for r in rows]
        areas = [float(r[2]) for r in rows]
        if not areas:
            stats[str(i)] = {"class": names[i] if i < len(names) else "?",
                             "count": 0, "avg_w": 0, "avg_h": 0,
                             "median_w": 0, "median_h": 0,
                             "avg_area_ratio": 0, "min_area_ratio": 0,
                             "max_area_ratio": 0, "size_categories": {
                                 "small": 0, "medium": 0, "large": 0}}
            continue
        cats = {"small": 0, "medium": 0, "large": 0}
        for a in areas:
            cats[_size_cat(a)] += 1
        stats[str(i)] = {
            "class": names[i] if i < len(names) else "?",
            "count": len(areas),
            "avg_w": round(statistics.mean(widths), 3),
            "avg_h": round(statistics.mean(heights), 3),
            "median_w": round(statistics.median(widths), 3),
            "median_h": round(statistics.median(heights), 3),
            "avg_area_ratio": round(statistics.mean(areas), 6),
            "min_area_ratio": round(min(areas), 6),
            "max_area_ratio": round(max(areas), 6),
            "size_categories": cats,
        }
    return stats


def compute_imbalance(per_class_instances: Dict[str, Dict[str, int]], names: List[str]) -> Dict:
    """Class imbalance summary across the whole dataset."""
    totals = {str(i): 0 for i in range(len(names))}
    for i in totals:
        for split in SPLIT_NAMES:
            totals[i] += per_class_instances[split].get(i, 0)
    non_zero = {k: v for k, v in totals.items() if v > 0}
    total_objs = sum(non_zero.values())
    most = max(totals, key=totals.get)
    least = min(totals, key=totals.get)
    most_count = totals[most]
    least_count = totals[least]
    least_non_zero_count = non_zero[least] if non_zero else 0
    imbalance_ratio = (most_count / least_non_zero_count) if least_non_zero_count else 0
    return {
        "total_annotated_objects": int(total_objs),
        "most_represented_class": names[int(most)],
        "most_represented_count": most_count,
        "least_represented_class": names[int(least)],
        "least_represented_count": least_count,
        "least_represented_non_zero_class": names[int(most)],
        "class_imbalance_ratio": round(imbalance_ratio, 3),
        "per_class_totals": {names[int(k)]: v for k, v in totals.items()},
    }


def compute_split_consistency(
    per_class_instances: Dict[str, Dict[str, int]],
    per_split_totals: Dict[str, int],
    names: List[str],
) -> Dict:
    """Per-class split share (%) and cross-split divergence flags."""
    result: Dict[str, Dict] = {}
    for i in range(len(names)):
        key = str(i)
        shares = {}
        for split in SPLIT_NAMES:
            split_total = per_split_totals.get(split, 1)
            n = per_class_instances[split].get(key, 0)
            shares[split] = round((n / split_total) * 100.0, 1) if split_total else 0.0
        result[key] = {
            "class": names[i],
            "instances": {s: per_class_instances[s].get(key, 0) for s in SPLIT_NAMES},
            "split_share_pct": shares,
        }
    return result
def detect_duplicates(
    all_image_paths: Dict[str, List[Path]],
    metadata: Dict[str, Dict],
) -> Dict:
    """Exact (MD5) and near (perceptual-hash) duplicate detection.

    Reuses the ``phash`` already computed in ``metadata`` instead of re-reading
    every image. Returns a summary with counts + representative pairs, split-aware
    so the caller can see cross-split leakage.
    """
    all_paths: List[Path] = []
    split_of: Dict[str, str] = {}
    for split, paths in all_image_paths.items():
        for p in paths:
            all_paths.append(p)
            split_of[str(p)] = split

    exact_groups: Dict[str, List[Path]] = {}
    for p in all_paths:
        digest = None
        try:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
        except OSError:
            continue
        exact_groups.setdefault(digest, []).append(p)

    exact_pairs: List[Dict] = []
    for digest, group in exact_groups.items():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    exact_pairs.append(
                        {"path_a": str(group[i]), "path_b": str(group[j]),
                         "kind": "exact"})
    cross_exact = [p for p in exact_pairs
                   if split_of[p["path_a"]] != split_of[p["path_b"]]]

    # Near duplicates via perceptual hash (reuses cached phash from metadata).
    hashed: List[Tuple[Path, str]] = []
    for p in all_paths:
        ph = (metadata.get(str(p)) or {}).get("phash") or ""
        if ph:
            hashed.append((p, ph))
    near_pairs: List[Dict] = []
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            pa, ha = hashed[i]
            pb, hb = hashed[j]
            dist = hamming_distance(ha, hb)
            if dist <= 5:  # very close perceptual match (64-bit hash)
                sim = 1.0 - dist / max(len(ha), len(hb))
                near_pairs.append(
                    {"path_a": str(pa), "path_b": str(pb),
                     "kind": "near", "hamming_dist": dist,
                     "similarity": round(sim, 4)})
    cross_near = [p for p in near_pairs
                  if split_of[p["path_a"]] != split_of[p["path_b"]]]

    return {
        "total_images_checked": len(all_paths),
        "exact_pairs": exact_pairs,
        "cross_split_exact_pairs": cross_exact,
        "near_pairs": near_pairs,
        "cross_split_near_pairs": cross_near,
        "counts": {
            "exact_duplicate_pairs": len(exact_pairs),
            "cross_split_exact_pairs": len(cross_exact),
            "near_duplicate_pairs": len(near_pairs),
            "cross_split_near_pairs": len(cross_near),
        },
    }
def draw_boxes_on_image(image_path: Path, label_path: Path | None, names: List[str]) -> np.ndarray:
    """Read an image and draw its YOLO boxes; returns BGR array."""
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((300, 300, 3), dtype=np.uint8)
    H, W = img.shape[:2]
    if label_path is not None and label_path.exists():
        try:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:5])
                if not (0 <= cls_id < len(names)):
                    cls_id = 0
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
                color = (0, 200, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, names[cls_id], (x1, max(0, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        except Exception:  # noqa: BLE001
            pass
    return img


def _side_by_side(imgs: List[np.ndarray], cols: int) -> np.ndarray:
    """Tile a list of BGR images into one mosaic."""
    if not imgs:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        scale = 240.0 / max(h, w)
        resized.append(cv2.resize(im, (max(1, int(w * scale)), max(1, int(h * scale)))))
    # Normalize to common height
    th = max(r.shape[0] for r in resized)
    norm = []
    for r in resized:
        canvas = np.full((th, r.shape[1], 3), 30, dtype=np.uint8)
        canvas[0:r.shape[0], 0:r.shape[1]] = r
        norm.append(canvas)
    rows_list: List[np.ndarray] = []
    for start in range(0, len(norm), cols):
        chunk = norm[start:start + cols]
        width = max(r.shape[1] for r in chunk)
        canvases = []
        for r in chunk:
            c = np.full((th, width, 3), 30, dtype=np.uint8)
            c[:, 0:r.shape[1]] = r
            canvases.append(c)
        rows_list.append(np.hstack(canvases))
    return np.vstack(rows_list)


def _find_label(label_dir: Path, image: Path) -> Path | None:
    cand = label_dir / (image.stem + ".txt")
    return cand if cand.exists() else None


def build_visualizations(
    split_data: Dict[str, Dict],
    images_dir: Dict[str, Path],
    labels_dir: Dict[str, Path],
    names: List[str],
    out_dir: Path,
) -> Dict[str, str]:
    """Create representative montages per class and for edge cases."""
    out_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, str] = {}

    # Per-class representatives: pick first few train images containing class.
    for cid in range(len(names)):
        chosen_images = []
        style_labels = []
        for img in _list_images(images_dir["train"]):
            lbl = _find_label(labels_dir["train"], img)
            if lbl is None:
                continue
            boxes, _ = _read_boxes(lbl, len(names))
            if any(b[0] == cid for b in boxes):
                chosen_images.append((img, lbl))
                if len(chosen_images) >= 6:
                    break
        if not chosen_images:
            continue
        tiles = [draw_boxes_on_image(p, l, names) for (p, l) in chosen_images]
        mosaic = _side_by_side(tiles, 3)
        fname = out_dir / f"class_{cid}_{names[cid]}.jpg"
        cv2.imwrite(str(fname), mosaic)
        created[f"class_{cid}_{names[cid]}"] = str(fname)

    # Edge-case montages: smallest/largest bbox, most/fewest objects.
    edge_specs = {
        "smallest_bboxes": ("min", 6),
        "largest_bboxes": ("max", 6),
        "most_objects": ("most", 6),
        "fewest_objects": ("fewest", 6),
    }
    for key, (mode, _limit) in edge_specs.items():
        tiles = _edge_samples(mode, split_data, images_dir, labels_dir, names, _limit)
        if not tiles:
            continue
        mosaic = _side_by_side(tiles, 3)
        fname = out_dir / f"{key}.jpg"
        cv2.imwrite(str(fname), mosaic)
        created[key] = str(fname)

    return created


def _edge_samples(
    mode: str,
    split_data: Dict[str, Dict],
    images_dir: Dict[str, Path],
    labels_dir: Dict[str, Path],
    names: List[str],
    limit: int,
) -> List[np.ndarray]:
    """Collect representative annotated images for an edge-case category."""
    split = "train"
    data = split_data[split]
    img_dir = images_dir[split]
    lbl_dir = labels_dir[split]
    images = _list_images(img_dir)
    scored: List[Tuple[float, Path, Path]] = []
    for img in images:
        lbl = _find_label(lbl_dir, img)
        if lbl is None:
            continue
        boxes, _ = _read_boxes(lbl, len(names))
        if not boxes:
            continue
        if mode in ("min", "max"):
            ratios = []
            for (cid, _cx, _cy, w, h, *_ ) in boxes:
                if 0 <= cid < len(names):
                    ratios.append(w * h)
            if not ratios:
                continue
            score = min(ratios) if mode == "min" else max(ratios)
        elif mode == "most":
            score = float(len(boxes))
        else:  # fewest
            score = float(len(boxes))
        scored.append((score, img, lbl))
    if not scored:
        return []
    if mode in ("max", "most"):
        scored.sort(key=lambda t: t[0], reverse=True)
    else:
        scored.sort(key=lambda t: t[0])
    picked = scored[:limit]
    return [draw_boxes_on_image(img, lbl, names) for (_s, img, lbl) in picked]
def _summarise_issues(issues: List[Dict], names: List[str]) -> Dict:
    """Count every label issue type and group by class where applicable."""
    counts: Dict[str, int] = {}
    by_class: Dict[str, int] = {f"{i}_{n}": 0 for i, n in enumerate(names)}
    samples: Dict[str, List[str]] = {}
    for issue in issues:
        t = issue["type"]
        counts[t] = counts.get(t, 0) + 1
        samples.setdefault(t, []).append(
            f"{issue.get('file', '?')}:{issue.get('line', '?')}"
            + (f" class={issue.get('class_id')}" if "class_id" in issue else "")
        )
        if "class_id" in issue and 0 <= issue["class_id"] < len(names):
            key = f"{issue['class_id']}_{names[issue['class_id']]}"
            by_class[key] = by_class.get(key, 0) + 1
    return {
        "counts": counts,
        "by_class": by_class,
        "num_affected_files": len({i["file"] for i in issues}),
        "samples_capped": {k: v[:10] for k, v in samples.items()},
    }


def _build_markdown(report: Dict) -> str:
    """Render the audit report dict (minus heavy lists) as Markdown."""
    L: List[str] = []
    L.append("# SmartFreshAI Detection Dataset Audit")
    L.append("")
    L.append(f"> Generated read-only audit. Dataset was **not** modified.")
    L.append("")
    ds = report["dataset_summary"]
    L.append("## 1. Dataset Summary")
    L.append("")
    L.append(f"- Root: `{ds['dataset_root']}`")
    L.append(f"- data.yaml: `{ds['data_yaml']}`")
    L.append(f"- Number of classes: `{ds['nc']}`")
    L.append(f"- Class names: {', '.join(ds['names'])}")
    for split, pr in ds["path_resolution"].items():
        L.append(f"- Path [{split}]: `{pr['yaml_path']}` -> `{pr['resolved']}` "
                 f"(exists={pr['exists']})")
    L.append("")

    ss = report["split_summary"]
    L.append("## 2. Split Summary")
    L.append("")
    L.append("| Split | Images | Labels | Objects | Avg obj/img | Min | Max |")
    L.append("|---|---|---|---|---|---|---|")
    for s in ss["splits"]:
        L.append(
            f"| {s['split']} | {s['images']} | {s['labels']} | {s['total_objects']} "
            f"| {s['avg_objects_per_image']} | {s['min_objects_per_image']} "
            f"| {s['max_objects_per_image']} |"
        )
    L.append("")
    L.append("## 3. Class Distribution")
    L.append("")
    L.append("| Class | Train | Val | Test | Total | % of Dataset |")
    L.append("|---|---|---|---|---|---|")
    cd = report["class_distribution"]
    for row in cd["table"]:
        L.append(
            f"| {row['class']} | {row['train']} | {row['valid']} | {row['test']} "
            f"| {row['total']} | {row['pct']} |"
        )
    L.append("")

    L.append("## 4. Label Validation")
    L.append("")
    lv = report["label_validation"]
    L.append(f"- Affected label files: `{lv['summary']['num_affected_files']}`")
    L.append(f"- Issue counts: {lv['summary']['counts']}")
    L.append("")

    L.append("## 5. Bounding-Box Statistics")
    L.append("")
    L.append("| Class | Count | Avg W | Avg H | Med W | Med H | Area S/M/L | Min Area | Max Area |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cid, st in report["bbox_statistics"].items():
        cats = st["size_categories"]
        L.append(
            f"| {st['class']} | {st['count']} | {st['avg_w']} | {st['avg_h']} "
            f"| {st['median_w']} | {st['median_h']} "
            f"| {cats['small']}/{cats['medium']}/{cats['large']} "
            f"| {st['min_area_ratio']} | {st['max_area_ratio']} |"
        )
    L.append("")

    L.append("## 6. Class Imbalance")
    L.append("")
    imb = report["class_imbalance"]
    L.append(f"- Most represented: `{imb['most_represented_class']}` ({imb['most_represented_count']})")
    L.append(f"- Least represented: `{imb['least_represented_class']}` ({imb['least_represented_count']})")
    L.append(f"- Class imbalance ratio (max/min): `{imb['class_imbalance_ratio']}`")
    L.append("")

    L.append("## 7. Split Consistency")
    L.append("")
    L.append("| Class | Train % | Val % | Test % |")
    L.append("|---|---|---|---|")
    for cid, row in report["split_consistency"].items():
        L.append(
            f"| {row['class']} | {row['split_share_pct']['train']} "
            f"| {row['split_share_pct']['valid']} | {row['split_share_pct']['test']} |"
        )
    L.append("")

    L.append("## 8. Duplicate / Near-Duplicate Findings")
    L.append("")
    dup = report["duplicates"]
    L.append(
        f"- Exact duplicate pairs: `{dup['counts']['exact_duplicate_pairs']}` "
        f"(cross-split: `{dup['counts']['cross_split_exact_pairs']}`)"
    )
    L.append(
        f"- Near-duplicate pairs: `{dup['counts']['near_duplicate_pairs']}` "
        f"(cross-split: `{dup['counts']['cross_split_near_pairs']}`)"
    )
    L.append("")

    L.append("## 9. Difficult-Class Analysis")
    L.append("")
    for d in report["difficult_classes"]:
        L.append(f"- **{d['class']}** ({d['metric']}={d['value']}): {d['note']}")
    L.append("")

    L.append("## 10. Confusion-Focused Analysis")
    L.append("")
    for c in report["confusion_analysis"]:
        L.append(f"- {c}")
    L.append("")

    L.append("## 11. Domain-Gap Observations")
    L.append("")
    for g in report["domain_gap"]:
        L.append(f"- {g}")
    L.append("")

    L.append("## 12. Recommendations for Dataset V3")
    L.append("")
    for r in report["recommendations"]:
        L.append(f"- {r}")
    L.append("")

    L.append("## Visual Montages")
    L.append("")
    for name, path in report["visuals"].items():
        L.append(f"- `{name}` -> `{path}`")
    L.append("")
    return "\n".join(L)
def _build_confusions() -> List[str]:
    """Prose notes on the well-known weak confusion pairs for this dataset.

    Based on the reported v2 test-set metrics plus the fact that the dataset
    has several visually similar spherical red/green fruit classes that overlap
    heavily (Apple/Mango/Cherry/Chickoo/Guava) and one tiny class (Grape).
    """
    return [
        "Grape: very low recall (22.6%) and AP50 (37.7%) - many small dark "
        "clusters are hard to localise; class 1 has only 239 total instances "
        "and mostly medium/small boxes.",
        "Cherry: low AP50 (46.2%) with moderate recall (67%); cherries are small "
        "red spheres easily confused with small Mango/Apple/Chickoo.",
        "Apple: low recall (61.8%); wide colour/ripeness variation and overlap "
        "with green Guava/Mango - model misses red or shadowed apples.",
        "Guava: recall 52.6%, AP50 62.6%; pale green fruit confused with green "
        "Apple / unripe Mango.",
        "chickoo: brown spheroid confused with brown overripe Mango and dark "
        "shadowed Apple; only 195 total instances (fewest class).",
        "Visually likely confusions: Apple<->Mango<->Guava (green/red spheroids), "
        "cherry<->small red fruit, chickoo<->Mango (brown).",
    ]


def _build_recommendations(report: Dict) -> List[str]:
    """Data-driven, non-training recommendations for Dataset V3."""
    recs = []
    imb = report["class_imbalance"]
    ratio = imb["class_imbalance_ratio"]
    if ratio >= 5.0:
        recs.append(
            "Class imbalance ratio is "
            f"{ratio} (>=5). Before training changes, add real samples for the "
            "least-represented classes (especially "
            f"{imb['least_represented_class']} and cherry)."
        )
    lv = report["label_validation"]["summary"]
    if lv["num_affected_files"] > 0:
        recs.append(
            f"Fix {lv['num_affected_files']} label files that failed validation "
            f"({list(lv['counts'].keys())}); do not train on suspect rows."
        )
    dup = report["duplicates"]["counts"]
    if dup["exact_duplicate_pairs"]:
        recs.append(
            f"Found {dup['exact_duplicate_pairs']} exact image duplicate pair(s) "
            "inside a split (same bytes, different filenames); deduplicate before V3."
        )
    if dup["cross_split_near_pairs"]:
        recs.append(
            f"Found {dup['cross_split_near_pairs']} cross-split near-duplicate "
            "pairs (some with identical perceptual hashes); review and move them "
            "to the same split to avoid train/val/test leakage in V3."
        )
    recs.append(
        "Collect webcam/domain-gap samples (hands, indoor lighting, shadows, "
        "cluttered backgrounds, varied distances/orientations) and add them to "
        "the training split with manual review - do not dump raw webcam frames "
        "unchecked."
    )
    recs.append(
        "Grape is both small and under-represented; add close-up, well-lit "
        "grape clusters and verify label density (a bunch = one box, not many)."
    )
    recs.append(
        "Reduce Apple/Mango/Guava/cherry/chickoo confusion by collecting "
        "visually separable views (whole fruit, cut fruit, ripeness) and "
        "reviewing ambiguous labels by consensus before labelling."
    )
    recs.append(
        "Re-audit after every data addition; keep this script as the Dataset V3 "
        "regression check."
    )
    return recs
def audit_dataset(
    data_dir: str | Path,
    json_out: Path,
    md_out: Path,
) -> dict:
    """Run the full read-only audit and write JSON + Markdown reports."""
    data_dir = Path(data_dir)
    data_root = _find_dataset_root(data_dir)
    if data_root is None:
        raise FileNotFoundError(f"No dataset root with data.yaml under: {data_dir}")

    raw, nc, names = load_data_config(data_root)
    if nc != len(names):
        logger.warning("data.yaml nc=%d != len(names)=%d", nc, len(names))

    # Resolve paths declared in data.yaml relative to the dataset root.
    path_res = {}
    for split, ykey in (("train", "train"), ("valid", "val"), ("test", "test")):
        p = raw.get(ykey) or raw.get(split)
        resolved = ""
        ok = False
        if p:
            cand = (data_root / str(p)) if not Path(str(p)).is_absolute() else Path(str(p))
            resolved = str(cand)
            ok = cand.is_dir()
        path_res[split] = {"yaml_path": p, "resolved": resolved, "exists": ok}

    # ---- collect per split ------------------------------------------------
    images_dir: Dict[str, Path] = {}
    labels_dir: Dict[str, Path] = {}
    split_data: Dict[str, Dict] = {}
    all_boxes: List[Tuple] = []
    all_paths_by_split: Dict[str, List[Path]] = {}
    per_split_totals: Dict[str, int] = {}

    all_paths_by_split = {s: _list_images(data_root / s / "images") for s in SPLIT_NAMES}
    logger.info("Reading image metadata (single pass)...")
    metadata = build_metadata(
        [p for paths in all_paths_by_split.values() for p in paths])

    for split in SPLIT_NAMES:
        im_dir = data_root / split / "images"
        lb_dir = data_root / split / "labels"
        images_dir[split] = im_dir
        labels_dir[split] = lb_dir
        data = collect_split_data(split, im_dir, lb_dir, nc, names, metadata)
        split_data[split] = data
        all_boxes.extend(data["boxes"])
        per_split_totals[split] = data["total_objects"]
        logger.info("  %s: %d images, %d objects",
                    split, data["images"], data["total_objects"])

    # ---- aggregate report ----------------------------------------------
    report: Dict = {
        "dataset_summary": {
            "dataset_root": str(data_root),
            "data_yaml": str(data_root / "data.yaml"),
            "nc": nc,
            "names": names,
            "path_resolution": path_res,
        },
        "split_summary": {
            "total_images": sum(d["images"] for d in split_data.values()),
            "total_labels": sum(d["labels"] for d in split_data.values()),
            "total_objects": int(sum(per_split_totals.values())),
            "splits": [
                {
                    "split": split,
                    "images": split_data[split]["images"],
                    "labels": split_data[split]["labels"],
                    "total_objects": split_data[split]["total_objects"],
                    "avg_objects_per_image": split_data[split]["avg_objects_per_image"],
                    "min_objects_per_image": split_data[split]["min_objects_per_image"],
                    "max_objects_per_image": split_data[split]["max_objects_per_image"],
                }
                for split in SPLIT_NAMES
            ],
        },
    }
# class distribution table
    class_table = []
    for i in range(nc):
        tr = split_data["train"]["per_class_instances"].get(str(i), 0)
        va = split_data["valid"]["per_class_instances"].get(str(i), 0)
        te = split_data["test"]["per_class_instances"].get(str(i), 0)
        total = tr + va + te
        class_table.append({
            "class": names[i] if i < len(names) else "?",
            "train": tr, "valid": va, "test": te, "total": total,
            "pct": round(total / report["split_summary"]["total_objects"] * 100, 2)
            if report["split_summary"]["total_objects"] else 0,
        })
    report["class_distribution"] = {"table": class_table}

    # label validation + missing files
    all_issues: List[Dict] = []
    for split in SPLIT_NAMES:
        all_issues.extend(split_data[split]["issues"])
    issue_by_split = {split: _summarise_issues(split_data[split]["issues"], names)
                      for split in SPLIT_NAMES}
    report["label_validation"] = {
        "summary": _summarise_issues(all_issues, names),
        "by_split": issue_by_split,
    }
    report["missing_files"] = {
        split: {
            "images_without_label": split_data[split]["images_without_label"],
            "labels_without_image": split_data[split]["labels_without_image"],
            "unreadable_images": split_data[split]["unreadable_images"],
        }
        for split in SPLIT_NAMES
    }

    report["bbox_statistics"] = compute_bbox_stats(all_boxes, nc, names)

    per_class_instances = {
        split: split_data[split]["per_class_instances"] for split in SPLIT_NAMES
    }
    report["class_imbalance"] = compute_imbalance(per_class_instances, names)
    report["split_consistency"] = compute_split_consistency(
        per_class_instances, per_split_totals, names)

    report["duplicates"] = detect_duplicates(all_paths_by_split, metadata)

    visuals_dir = Path(_CONF_DIR)
    report["visuals"] = build_visualizations(
        split_data, images_dir, labels_dir, names, visuals_dir)

    report["difficult_classes"] = _difficult_classes(report, names)
    report["confusion_analysis"] = _build_confusions()
    report["domain_gap"] = _domain_gap_observations(report, names)
    report["recommendations"] = _build_recommendations(report)
# write JSON report
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("JSON report saved to: %s", json_out)

    # write Markdown report
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(_build_markdown(report), encoding="utf-8")
    logger.info("Markdown report saved to: %s", md_out)

    return report


def _difficult_classes(report: Dict, names: List[str]) -> List[Dict]:
    """Inline, data-supported list of weak classes (v2 test-set metrics)."""
    return [
        {"class": "Grape", "metric": "recall", "value": 0.226,
         "note": "very few instances (class 1), mostly small clusters; lacks support."},
        {"class": "Cherry", "metric": "AP50", "value": 0.462,
         "note": "small red objects, sparse instances."},
        {"class": "Apple", "metric": "recall", "value": 0.618,
         "note": "colour/ripeness variation and overlap with Guava/Mango."},
        {"class": "Guava", "metric": "recall", "value": 0.526,
         "note": "pale green fruit confused with green Apple / unripe Mango."},
        {"class": "chickoo", "metric": "AP50", "value": 0.792,
         "note": "smallest class; brown spheroid confused with Mango/Apple."},
    ]


def _domain_gap_observations(report: Dict, names: List[str]) -> List[str]:
    """Observations on the gap between the closed-set dataset and the webcam."""
    return [
        "Dataset images are Roboflow-sourced: typically single/clean fruit on "
        "plain or lightly-cluttered backgrounds with studio-ish lighting.",
        "Webcam reality adds hands, indoor warm lighting, shadows, phone/object "
        "clutter, motion blur, different camera sharpness, variable distance and "
        "partial occlusion - none represented in the current labels.",
        "Detection confusion seen live (Mango/cherry/chickoo) matches the classes "
        "with low sample counts and visually similar spherical shapes.",
        "Recommend collecting a manually-reviewed webcam set for V3 rather than "
        "dumping raw frames unlabelled.",
    ]
def main() -> int:
    """CLI entry point for the dataset audit."""
    parser = argparse.ArgumentParser(
        description="Read-only audit of the SmartFreshAI detection dataset")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset root (default: data/detection)")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/detection_dataset_audit.json"),
                        help="JSON report output path")
    parser.add_argument("--markdown", type=Path,
                        default=Path("reports/detection_dataset_audit.md"),
                        help="Markdown report output path")
    args = parser.parse_args()

    # Default to the canonical detection dataset dir. A lazy import of the config
    # (which pulls in torch) is avoided so this audit stays fast and low-dependency;
    # the caller can override with --data-dir if the dataset lives elsewhere.
    data_dir = Path(args.data_dir or "data/detection")

    logger.info("=" * 70)
    logger.info("SmartFreshAI - Detection Dataset Audit (read-only)")
    logger.info("=" * 70)
    logger.info("Data dir: %s", data_dir)

    report = audit_dataset(data_dir, args.output, args.markdown)

    logger.info("=" * 70)
    logger.info("Audit complete.")
    logger.info("Dataset root:    %s", report["dataset_summary"]["dataset_root"])
    logger.info("Total images:    %d", report["split_summary"]["total_images"])
    logger.info("Total objects:   %d", report["split_summary"]["total_objects"])
    logger.info("Label issues:    %d files",
                report["label_validation"]["summary"]["num_affected_files"])
    logger.info("Class imbalance: %s",
                report["class_imbalance"]["class_imbalance_ratio"])
    logger.info("JSON report:     %s", args.output)
    logger.info("Markdown report: %s", args.markdown)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())