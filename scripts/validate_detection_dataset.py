#!/usr/bin/env python3
"""Validate the FreshSense detection dataset structure and content."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from configs.config import Config
from src.utils.environment import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_dataset(data_dir: Path, output_report: Path | None = None) -> dict:
    """Validate detection dataset structure and return a report."""
    report = {
        "status": "pass",
        "data_dir": str(data_dir),
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    if not data_dir.exists():
        report["status"] = "fail"
        report["errors"].append(f"data_dir does not exist: {data_dir}")
        return report

    required_dirs = ["train", "val", "test"]
    found_dirs = []
    for d in required_dirs:
        dir_path = data_dir / d
        if dir_path.is_dir():
            found_dirs.append(d)
            report["summary"][d] = {
                "images": 0,
                "labels": 0,
                "exists": True,
            }
        else:
            report["warnings"].append(f"Missing split directory: {d}")

    if not found_dirs:
        report["status"] = "fail"
        report["errors"].append("No split directories found (train/val/test)")
        return report

    # Check data.yaml
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        data_yaml = data_dir.parent / "data.yaml"
    if not data_yaml.exists():
        report["status"] = "fail"
        report["errors"].append("data.yaml not found")
        return report

    logger.info("Found data.yaml: %s", data_yaml)
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
        report["summary"]["data_yaml"] = {
            "path": str(data_yaml),
            "nc": data_config.get("nc", 0),
            "names": data_config.get("names", []),
        }
        logger.info("Classes (%d): %s", data_config.get("nc", 0), data_config.get("names", []))
    except Exception as exc:
        report["status"] = "fail"
        report["errors"].append(f"Failed to parse data.yaml: {exc}")
        return report

    for d in found_dirs:
        split_dir = data_dir / d
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if images_dir.is_dir():
            images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            report["summary"][d]["images"] = len(images)
            logger.info("  %s images: %d", d, len(images))

        if labels_dir.is_dir():
            labels = list(labels_dir.glob("*.txt"))
            report["summary"][d]["labels"] = len(labels)
            logger.info("  %s labels: %d", d, len(labels))

            for label_file in labels[:3]:
                try:
                    content = label_file.read_text(encoding="utf-8").strip().splitlines()
                    for line in content[:2]:
                        parts = line.split()
                        if len(parts) != 5:
                            report["warnings"].append(
                                f"Invalid label format in {label_file}: {line}"
                            )
                            break
                        cls_id, x, y, w, h = parts
                        cls_id = int(cls_id)
                        x, y, w, h = float(x), float(y), float(w), float(h)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            report["warnings"].append(
                                f"Coordinates out of range in {label_file}"
                            )
                except Exception as exc:
                    report["warnings"].append(f"Error reading {label_file}: {exc}")

    total_images = sum(report["summary"].get(d, {}).get("images", 0) for d in found_dirs)
    total_labels = sum(report["summary"].get(d, {}).get("labels", 0) for d in found_dirs)
    report["summary"]["total_images"] = total_images
    report["summary"]["total_labels"] = total_labels
    logger.info("Total images: %d, labels: %d", total_images, total_labels)

    if report["errors"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "warning"

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
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
        default=None,
        help="Path to save validation report JSON",
    )
    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")
    data_dir = args.data_dir or config.detection_dataset.detection_data_dir

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

﻿