#!/usr/bin/env python3
"""Check dataset for leakage between train/val/test splits."""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class LeakageReport:
    """Report of dataset leakage findings."""
    exact_duplicates: List[Tuple[str, str, str]] = field(default_factory=list)
    near_duplicates: List[Tuple[str, str, str, float]] = field(default_factory=list)
    cross_split_violations: List[Tuple[str, str, str]] = field(default_factory=list)
    corrupted_images: List[str] = field(default_factory=list)
    total_checked: int = 0
    splits_analyzed: List[str] = field(default_factory=list)


def compute_md5(image_path: Path) -> str:
    """Compute MD5 hash of image file."""
    return hashlib.md5(image_path.read_bytes()).hexdigest()


def compute_perceptual_hash(image_path: Path, hash_size: int = 8) -> str:
    """Compute perceptual hash of image."""
    try:
        img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size))
        pixels = np.array(img)
        diff = pixels[1:] > pixels[:-1]
        return "".join("1" if d else "0" for d in diff.flatten())
    except Exception as exc:  # noqa: BLE001
        return ""

def find_exact_duplicates(image_paths: List[Path]) -> List[Tuple[str, str, str]]:
    """Find exact duplicate images by MD5 hash."""
    hash_map: Dict[str, List[Tuple[Path, str]]] = {}

    for path in image_paths:
        try:
            file_hash = compute_md5(path)
            split = path.parts[-3] if len(path.parts) >= 3 else "unknown"
            hash_map.setdefault(file_hash, []).append((path, split))
        except Exception as exc:  # noqa: BLE001
            pass

    duplicates = []
    for hash_val, paths in hash_map.items():
        if len(paths) > 1:
            for i, (path1, split1) in enumerate(paths):
                for path2, split2 in paths[i + 1:]:
                    duplicates.append((str(path1), str(path2), f"same_hash_{hash_val[:8]}"))
    return duplicates


def find_near_duplicates(
    image_paths: List[Path],
    threshold: int = 10,
) -> List[Tuple[str, str, str, float]]:
    """Find near-duplicate images using perceptual hash."""
    hashes = []
    for path in image_paths:
        phash = compute_perceptual_hash(path)
        if phash:
            split = path.parts[-3] if len(path.parts) >= 3 else "unknown"
            hashes.append((path, phash, split))

    near_dups = []
    for i, (path1, hash1, split1) in enumerate(hashes):
        for path2, hash2, split2 in hashes[i + 1:]:
            dist = hamming_distance(hash1, hash2)
            similarity = 1.0 - dist / max(len(hash1), len(hash2))
            if dist <= threshold:
                near_dups.append((str(path1), str(path2), f"phash_dist_{dist}", similarity))
    return near_dups


def check_corrupted_images(image_paths: List[Path]) -> List[str]:
    """Check for corrupted/unreadable images."""
    corrupted = []
    for path in image_paths:
        try:
            img = cv2.imread(str(path))
            if img is None:
                corrupted.append(str(path))
        except Exception:  # noqa: BLE001
            corrupted.append(str(path))
    return corrupted


def check_dataset_leakage(
    dataset_root: Path,
    output_dir: Path = Path("reports"),
) -> LeakageReport:
    """Check dataset for leakage across splits."""
    report = LeakageReport()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "validation", "test"]
    found_splits = [d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.lower() in splits]
    report.splits_analyzed = found_splits

    if not found_splits:
        return report

    all_images: List[Path] = []
    for split in found_splits:
        split_dir = dataset_root / split
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            all_images.extend(split_dir.rglob(ext))

    report.total_checked = len(all_images)
    logger.info("Checking %d images across splits: %s", len(all_images), found_splits)

    report.corrupted_images = check_corrupted_images(all_images)
    report.exact_duplicates = find_exact_duplicates(all_images)
    report.near_duplicates = find_near_duplicates(all_images, threshold=10)

    split_map: Dict[str, str] = {}
    for path in all_images:
        for split in found_splits:
            if split in path.parts:
                split_map[str(path)] = split
                break

    for path1, path2, reason in report.exact_duplicates:
        split1 = split_map.get(path1, "unknown")
        split2 = split_map.get(path2, "unknown")
        if split1 != split2:
            report.cross_split_violations.append((path1, path2, f"exact_duplicate_{reason}"))

    for path1, path2, reason, similarity in report.near_duplicates:
        split1 = split_map.get(path1, "unknown")
        split2 = split_map.get(path2, "unknown")
        if split1 != split2:
            report.cross_split_violations.append((path1, path2, f"near_duplicate_{reason}"))

    _save_leakage_report(report, output_dir)
    return report


def _save_leakage_report(report: LeakageReport, output_dir: Path) -> None:
    """Save leakage report to Markdown."""
    lines = [
        "# Dataset Leakage Report",
        "",
        f"**Total images checked**: {report.total_checked}",
        f"**Splits analyzed**: {', '.join(report.splits_analyzed)}",
        "",
        "## Summary",
        "",
        f"- Exact duplicates: {len(report.exact_duplicates)}",
        f"- Near duplicates: {len(report.near_duplicates)}",
        f"- Cross-split violations: {len(report.cross_split_violations)}",
        f"- Corrupted images: {len(report.corrupted_images)}",
        "",
    ]

    if report.cross_split_violations:
        lines.extend([
            "## CRITICAL: Cross-Split Leakage Found!",
            "",
            "The following images appear in multiple splits:",
            "",
        ])
        for path1, path2, reason in report.cross_split_violations[:20]:
            lines.append(f"- `{path1}` and `{path2}` ({reason})")
        if len(report.cross_split_violations) > 20:
            lines.append(f"- ... and {len(report.cross_split_violations) - 20} more")

    if report.corrupted_images:
        lines.extend(["", "## Corrupted Images", ""])
        for path in report.corrupted_images[:20]:
            lines.append(f"- `{path}`")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "- Remove all cross-split duplicates",
        "- Re-run train/val/test split after deduplication",
        "- Verify test set remains untouched during development",
        "",
    ])

    report_path = output_dir / "dataset_leakage_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Leakage report saved to %s", report_path)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Check dataset for leakage")
    parser.add_argument("dataset_root", type=Path, help="Root directory of dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = check_dataset_leakage(args.dataset_root, args.output_dir)

    print("=" * 60)
    print("DATASET LEAKAGE CHECK")
    print("=" * 60)
    print(f"Images checked: {report.total_checked}")
    print(f"Exact duplicates: {len(report.exact_duplicates)}")
    print(f"Near duplicates: {len(report.near_duplicates)}")
    print(f"Cross-split violations: {len(report.cross_split_violations)}")
    print(f"Corrupted images: {len(report.corrupted_images)}")
    print("=" * 60)

    if report.cross_split_violations:
        print("\nWARNING: Cross-split leakage detected!")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
