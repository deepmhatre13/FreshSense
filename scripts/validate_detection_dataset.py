#!/usr/bin/env python3
"""Validate the FreshSense detection dataset structure and content.

Performs a structural + content audit of a local YOLO-format export
(``train/valid/test`` splits with ``images/`` and ``labels/`` folders)::

    <dataset_root>/
    ├── data.yaml
    ├── README.roboflow.txt
    ├── train/
    │   ├── images/*.jpg
    │   └── labels/*.txt
    ├── valid/   (or val/  - accepted as an alias)
    │   ├── images/*.jpg
    │   └── labels/*.txt
    └── test/
        ├── images/*.jpg
        └── labels/*.txt

Roboflow YOLO exports use ``valid``; some exports use ``val``. Both are
accepted and canonicalized to ``valid``.

Checks performed:
  - dataset root and ``data.yaml`` are found and parseable
  - every expected split exists with images/ and labels/ folders
  - every image has a matching label file and vice versa
  - every label row has exactly 5 fields, a valid class id, and normalized
    [0, 1] coordinates with positive width/height
  - per-split class distribution against ``nc``/``names`` from data.yaml

Emits a machine-readable report (default: ``reports/detection_dataset_validation.json``).

Usage:
    python scripts/validate_detection_dataset.py
    python scripts/validate_detection_dataset.py --data-dir data/detection --output reports/detection_dataset_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Allow running directly (python scripts/<name>.py) from any CWD.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.config import Config
from src.utils.environment import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Canonical splits. "val" is accepted as an alias for "valid".
EXPECTED_SPLITS = ("train", "valid", "test")
SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def _list_images(images_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(pattern))
    return sorted(files)


def _list_labels(labels_dir: Path) -> List[Path]:
    return sorted(labels_dir.glob("*.txt"))


def _find_dataset_root(data_dir: str | Path) -> Path | None:
    """Locate the directory that directly contains ``data.yaml``.

    ``data_dir`` may be given as a ``str`` or ``Path``; it is normalised to a
    ``Path`` immediately so all filesystem operations are reliable.

    Roboflow exports may place the dataset directly in ``data_dir`` or inside a
    version sub-folder (e.g. ``data/detection/fruits-test-1``).
    """
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


def _analyze_label_file(
    label_file: Path,
    nc: int,
    report: dict,
) -> Tuple[List[str], Dict[str, int]]:
    """Validate one YOLO label file.

    Returns ``(bad_rows, class_counts)``. A row is valid if it has exactly 5
    whitespace-separated fields: ``<class_id> <cx> <cy> <w> <h>`` with
    ``0 <= class_id < nc``, ``0 <= cx, cy <= 1`` and ``0 < w, h <= 1``.
    """
    bad: List[str] = []
    counts = {str(i): 0 for i in range(nc)}
    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        report["warnings"].append(f"Error reading {label_file}: {exc}")
        return bad, counts

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            bad.append(f"{label_file}:{line_no}: expected 5 fields, got {len(parts)}")
            report["errors"].append(f"Invalid label format in {label_file}:{line_no}: {line}")
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            bad.append(f"{label_file}:{line_no}: non-numeric field")
            report["errors"].append(f"Invalid label format in {label_file}:{line_no}: {line}")
            continue
        if not (0 <= cls_id < nc):
            bad.append(f"{label_file}:{line_no}: class id {cls_id} out of range [0,{nc})")
            report["errors"].append(f"Class id out of range in {label_file}:{line_no}")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            bad.append(f"{label_file}:{line_no}: center coordinates out of [0,1]")
            report["errors"].append(f"Coordinates out of range in {label_file}:{line_no}")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            bad.append(f"{label_file}:{line_no}: width/height not in (0,1]")
            report["errors"].append(f"Box size out of range in {label_file}:{line_no}")
        if 0 <= cls_id < nc:
            counts[str(cls_id)] += 1
    return bad, counts


def validate_dataset(data_dir: str | Path, output_report: Path | None = None) -> dict:
    """Validate detection dataset structure and content; return a report.

    ``data_dir`` may be given as a ``str`` or ``Path``; it is normalised to a
    ``Path`` immediately so all filesystem operations are reliable.
    """
    data_dir = Path(data_dir)
    report = {
        "status": "pass",
        "data_dir": str(data_dir),
        "dataset_root": None,
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    data_root = _find_dataset_root(data_dir)
    if data_root is None:
        report["status"] = "fail"
        report["errors"].append(f"No dataset root with data.yaml found under: {data_dir}")
        return report
    report["dataset_root"] = str(data_root)

    data_yaml = data_root / "data.yaml"
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f) or {}
    except Exception as exc:
        report["status"] = "fail"
        report["errors"].append(f"Failed to parse data.yaml ({data_yaml}): {exc}")
        return report

    try:
        nc = int(data_config.get("nc", 0))
    except (TypeError, ValueError):
        nc = 0

    names_raw = data_config.get("names")
    class_names: List[str] = []
    if isinstance(names_raw, dict):
        class_names = [str(names_raw[k]) for k in sorted(names_raw)]
    elif isinstance(names_raw, list):
        class_names = [str(n) for n in names_raw]

    if nc and nc != len(class_names):
        report["warnings"].append(f"data.yaml nc={nc} but names has {len(class_names)} entries")
    if not nc:
        report["status"] = "fail"
        report["errors"].append("data.yaml is missing nc (number of classes)")
        return report

    report["summary"]["data_yaml"] = {
        "path": str(data_yaml),
        "nc": nc,
        "names": class_names,
    }
    report["summary"]["splits"] = {}
    logger.info("Found data.yaml: %s", data_yaml)
    logger.info("Classes (%d): %s", nc, class_names)

    found_splits = []
    for name in data_root.iterdir():
        if not name.is_dir():
            continue
        split_key = SPLIT_ALIASES.get(name.name)
        if split_key is not None:
            found_splits.append((split_key, name))

    if not found_splits:
        report["status"] = "fail"
        report["errors"].append("No image split folders (train/valid/test) found")
        return report

    total_images = 0
    total_labels = 0
    class_histogram = {str(i): 0 for i in range(nc)}

    for split_key, split_path in found_splits:
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
        split_info = {
            "folder": split_path.name,
            "canonical_split": split_key,
            "images_dir": None,
            "labels_dir": None,
            "images": 0,
            "labels": 0,
            "images_without_label": [],
            "labels_without_image": [],
            "class_distribution": {},
            "bad_label_rows": [],
        }

        if images_dir.is_dir():
            split_info["images_dir"] = str(images_dir)
            images = _list_images(images_dir)
            split_info["images"] = len(images)
            image_stems = {p.stem for p in images}
        else:
            images, image_stems = [], set()
            report["warnings"].append(f"{split_key}: missing images folder: {images_dir}")

        if labels_dir.is_dir():
            split_info["labels_dir"] = str(labels_dir)
            labels = _list_labels(labels_dir)
            split_info["labels"] = len(labels)
            label_stems = {p.stem for p in labels}
        else:
            labels, label_stems = [], set()
            report["warnings"].append(f"{split_key}: missing labels folder: {labels_dir}")

        # Every image must have a label; every label must have an image.
        missing_label_stems = image_stems - label_stems
        orphan_label_stems = label_stems - image_stems
        split_info["images_without_label"] = sorted(missing_label_stems)[:10]
        split_info["labels_without_image"] = sorted(orphan_label_stems)[:10]
        if missing_label_stems:
            report["warnings"].append(
                f"{split_key}: {len(missing_label_stems)} image(s) have no label file"
            )
        if orphan_label_stems:
            report["warnings"].append(
                f"{split_key}: {len(orphan_label_stems)} label file(s) have no image"
            )

        class_dist = {str(i): 0 for i in range(nc)}
        for label_file in labels:
            bad_rows, counts = _analyze_label_file(label_file, nc, report)
            split_info["bad_label_rows"].extend(bad_rows)
            for cls_id, count in counts.items():
                class_dist[cls_id] += count
                class_histogram[cls_id] += count

        for cls_id in range(nc):
            if class_dist[str(cls_id)] == 0:
                cls_name = class_names[cls_id] if cls_id < len(class_names) else "?"
                report["warnings"].append(
                    f"{split_key}: class {cls_id} ({cls_name}) has 0 boxes"
                )
        split_info["class_distribution"] = class_dist

        total_images += split_info["images"]
        total_labels += split_info["labels"]
        report["summary"]["splits"][split_key] = split_info
        logger.info(
            "  %s/%s: %d images, %d labels",
            split_path.name, split_key, split_info["images"], split_info["labels"],
        )

    report["summary"]["total_images"] = total_images
    report["summary"]["total_labels"] = total_labels
    report["summary"]["class_histogram"] = class_histogram
    logger.info("Total images: %d, labels: %d", total_images, total_labels)

    canonical_found = {key for key, _ in found_splits}
    for expected in EXPECTED_SPLITS:
        if expected not in canonical_found:
            report["warnings"].append(f"Missing split: {expected}")

    if report["errors"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "warning"

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Report saved to: %s", output_report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate FreshSense detection dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root directory (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/detection_dataset_validation.json"),
        help="Path to save validation report JSON",
    )
    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")
    data_dir = Path(args.data_dir or config.detection_dataset.detection_data_dir)

    logger.info("=" * 70)
    logger.info("FreshSense AI - Validate Detection Dataset")
    logger.info("=" * 70)
    logger.info("Data dir: %s", data_dir)
    logger.info("=" * 70)

    report = validate_dataset(data_dir, args.output)

    logger.info("=" * 70)
    logger.info("Validation status: %s", report["status"].upper())
    for err in report["errors"]:
        logger.error("  ERROR: %s", err)
    for warn in report["warnings"]:
        logger.warning("  WARNING: %s", warn)
    logger.info("=" * 70)

    sys.exit(0 if report["status"] != "fail" else 1)


if __name__ == "__main__":
    main()