#!/usr/bin/env python3
"""Analyze collected real-world dataset for Phase 4."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics for the real-world dataset."""
    total_samples: int = 0
    accepted_samples: int = 0
    rejected_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    brightness_values: List[float] = field(default_factory=list)
    contrast_values: List[float] = field(default_factory=list)
    blur_values: List[float] = field(default_factory=list)
    confidence_values: List[float] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)


def load_metadata_files(metadata_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON metadata files."""
    metadata = []
    for json_file in metadata_dir.glob("*.json"):
        if json_file.name.startswith("session_"):
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            metadata.append(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", json_file, exc)
    return metadata


def analyze_dataset(metadata: List[Dict[str, Any]]) -> DatasetStats:
    """Analyze metadata to compute dataset statistics."""
    stats = DatasetStats()

    for item in metadata:
        stats.total_samples += 1

        if item.get("accepted", True):
            stats.accepted_samples += 1
            label = item.get("label", "unknown")
            stats.class_distribution[label] = stats.class_distribution.get(label, 0) + 1
        else:
            stats.rejected_samples += 1

        if item.get("session_id"):
            stats.session_ids.append(item["session_id"])

        quality = item.get("quality")
        if quality:
            if quality.get("brightness") is not None:
                stats.brightness_values.append(quality["brightness"])
            if quality.get("contrast") is not None:
                stats.contrast_values.append(quality["contrast"])
            if quality.get("blur_score") is not None:
                stats.blur_values.append(quality["blur_score"])

        if item.get("predicted_confidence") is not None:
            stats.confidence_values.append(item["predicted_confidence"])

    return stats


def detect_duplicates(metadata: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Detect potential duplicate images by metadata similarity."""
    duplicates = []
    seen_hashes: Dict[str, str] = {}

    for item in metadata:
        img_path = item.get("image_path", "")
        if not img_path:
            continue

        try:
            import hashlib
            full_path = Path("data/real_world") / img_path
            if full_path.exists():
                img = cv2.imread(str(full_path))
                if img is not None:
                    img_small = cv2.resize(img, (32, 32))
                    img_hash = hashlib.md5(img_small.tobytes()).hexdigest()
                    if img_hash in seen_hashes:
                        duplicates.append((img_path, seen_hashes[img_hash]))
                    else:
                        seen_hashes[img_hash] = img_path
        except Exception:  # noqa: BLE001
            pass

    return duplicates


def generate_report(stats: DatasetStats, duplicates: List[Tuple[str, str]], output_path: Path) -> None:
    """Generate Markdown report."""
    lines = [
        "# Real-World Dataset Analysis Report",
        "",
        "## Overview",
        "",
        f"- **Total samples**: {stats.total_samples}",
        f"- **Accepted**: {stats.accepted_samples} ({100.0*stats.accepted_samples/max(stats.total_samples,1):.1f}%)",
        f"- **Rejected**: {stats.rejected_samples} ({100.0*stats.rejected_samples/max(stats.total_samples,1):.1f}%)",
        f"- **Unique sessions**: {len(set(stats.session_ids))}",
        "",
    ]

    if stats.class_distribution:
        lines.extend(["## Class Distribution", ""])
        total = sum(stats.class_distribution.values())
        for cls, count in sorted(stats.class_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / total if total > 0 else 0.0
            lines.append(f"- **{cls}**: {count} ({pct:.1f}%)")
        lines.append("")

    if stats.brightness_values:
        lines.extend([
            "## Image Quality Statistics",
            "",
            "### Brightness",
            "",
            f"- Mean: {np.mean(stats.brightness_values):.2f}",
            f"- Std: {np.std(stats.brightness_values):.2f}",
            f"- Min: {min(stats.brightness_values):.2f}",
            f"- Max: {max(stats.brightness_values):.2f}",
            "",
        ])

    if stats.contrast_values:
        lines.extend([
            "### Contrast",
            "",
            f"- Mean: {np.mean(stats.contrast_values):.2f}",
            f"- Std: {np.std(stats.contrast_values):.2f}",
            "",
        ])

    if stats.blur_values:
        lines.extend([
            "### Blur Score",
            "",
            f"- Mean: {np.mean(stats.blur_values):.2f}",
            f"- Std: {np.std(stats.blur_values):.2f}",
            "",
        ])

    if stats.confidence_values:
        lines.extend([
            "## Prediction Confidence Distribution",
            "",
            f"- Mean: {np.mean(stats.confidence_values):.3f}",
            f"- Std: {np.std(stats.confidence_values):.3f}",
            f"- Min: {min(stats.confidence_values):.3f}",
            f"- Max: {max(stats.confidence_values):.3f}",
            "",
        ])

    if duplicates:
        lines.extend([
            "## Potential Duplicates Detected",
            "",
            f"Found {len(duplicates)} potential duplicate pairs.",
            "",
        ])
        for img1, img2 in duplicates[:10]:
            lines.append(f"- `{img1}` and `{img2}`")

    lines.extend([
        "",
        "## Notes",
        "",
        "- This analysis is based on metadata collected during capture",
        "- Quality thresholds are configurable in collection settings",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to %s", output_path)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze real-world dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_world"))
    parser.add_argument("--output", type=Path, default=Path("reports/real_world_dataset_report.md"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    metadata_dir = args.data_dir / "metadata"
    if not metadata_dir.exists():
        logger.error("Metadata directory not found: %s", metadata_dir)
        return 1

    metadata = load_metadata_files(metadata_dir)
    if not metadata:
        logger.warning("No metadata files found in %s", metadata_dir)
        return 0

    stats = analyze_dataset(metadata)
    duplicates = detect_duplicates(metadata)
    generate_report(stats, duplicates, args.output)

    print(f"Analyzed {stats.total_samples} samples")
    print(f"Accepted: {stats.accepted_samples}, Rejected: {stats.rejected_samples}")
    print(f"Report saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
