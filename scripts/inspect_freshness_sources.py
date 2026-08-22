"""Inspect freshness source datasets before canonical dataset construction.

Usage:
    python scripts/inspect_freshness_sources.py
    python scripts/inspect_freshness_sources.py --json reports/freshness_sources_inspection.json

Inspects:
    1. data/Original Image/   (Mendeley)
    2. data/Quality Dataset/   (Kaggle)
    3. data/raw/dataset/dataset/  (Legacy -- read-only)

Reports for each:
    - total files / images
    - image extensions
    - immediate subdirectories
    - discovered class/label names
    - images per class
    - zero-byte files
    - unreadable/corrupt images
    - suspicious files
    - nested directories
    - duplicate filenames
    - sample paths
    - label mapping (source -> canonical)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.freshness_dataset_builder import (
    CANONICAL_CLASS_MAPPING,
    load_freshness_config,
    inspect_all_sources,
    format_inspection_report,
    _match_quality_fruit,
    _check_contradiction,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect freshness source datasets."
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Optional path to save JSON inspection report.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to freshness_sources.yaml config.",
    )
    args = parser.parse_args()

    config = load_freshness_config(Path(args.config) if args.config else None)
    reports = inspect_all_sources(config)

    # Print human-readable reports
    print("=" * 70)
    print("FRESHNESS SOURCE INSPECTION REPORT")
    print("=" * 70)

    # Determine which source contains which fruit
    print("\n--- SOURCE-TO-CANONICAL-FRUIT MAPPING ---")
    print("\nMendeley Original Image directory labels -> canonical:")
    mendeley_cfg = config.get("mendeley_original_image", {})
    for label, mapped in mendeley_cfg.get("label_mapping", {}).items():
        if mapped.get("accept"):
            print(f"  {label} -> {mapped['canonical_class']}  ({mapped['fruit']}, {mapped['freshness_state']})")
        else:
            print(f"  {label} -> REJECTED ({mapped.get('reject_reason', 'unsupported')})")

    print("\nQuality Dataset (directory=state, filename=fruit) -> canonical:")
    qd_cfg = config.get("quality_dataset", {})
    fruit_kw = qd_cfg.get("fruit_keywords", {})
    for fruit, fc in sorted(fruit_kw.items()):
        canonical_examples = [f"{fruit}_fresh", f"{fruit}_rotten"]
        print(f"  filename keyword '{fc['keywords']}' -> fruit '{fruit}' -> {canonical_examples}")

    print("\nLegacy dataset directory labels -> canonical:")
    legacy_cfg = config.get("legacy_fresh_rotten", {})
    for label, mapped in legacy_cfg.get("label_mapping", {}).items():
        if mapped.get("accept"):
            print(f"  {label} -> {mapped['canonical_class']}")

    # Which source contains which fruit
    print("\n--- WHICH SOURCE CONTAINS WHICH FRUIT ---")
    taxonomy_fruits = set()
    for cls in CANONICAL_CLASS_MAPPING.values():
        parts = cls.split("_")
        taxonomy_fruits.add(parts[0])

    for fruit in sorted(taxonomy_fruits):
        sources = []
        # Check Mendeley
        for label, mapped in mendeley_cfg.get("label_mapping", {}).items():
            if mapped.get("fruit") == fruit and mapped.get("accept"):
                sources.append("Mendeley")
                break
        # Check Quality Dataset (would need deeper check)
        # Check Legacy
        for label, mapped in legacy_cfg.get("label_mapping", {}).items():
            if mapped.get("fruit") == fruit and mapped.get("accept"):
                sources.append("Legacy")
                break
        if not sources:
            sources.append("Quality (filename only)")
        print(f"  {fruit}: {', '.join(sources)}")

    # Print individual reports
    for key, report in reports.items():
        print()
        print(format_inspection_report(report))

    # Save JSON if requested
    if args.json:
        json_path = ROOT_DIR / args.json
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = {}
        for key, report in reports.items():
            json_data[key] = {
                "source_name": report.source_name,
                "source_path": report.source_path,
                "total_files": report.total_files,
                "total_images": report.total_images,
                "image_extensions": report.image_extensions,
                "immediate_subdirs": report.immediate_subdirs,
                "class_counts": report.class_counts,
                "zero_byte_files": report.zero_byte_files,
                "corrupt_images": report.corrupt_images,
                "suspicious_files": report.suspicious_files,
                "nested_directories": report.nested_directories,
                "duplicate_filenames": report.duplicate_filenames,
                "non_image_files": report.non_image_files,
                "accepted_labels": report.accepted_labels,
                "rejected_labels": report.rejected_labels,
                "accepted_count": report.accepted_count,
                "rejected_count": report.rejected_count,
                "label_mapping": report.label_mapping,
                "sample_paths": report.sample_paths,
            }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nJSON report saved to: {json_path}")


if __name__ == "__main__":
    main()
