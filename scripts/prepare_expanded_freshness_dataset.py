"""Expanded Freshness Dataset Assembly Script â€” Phase 4A.

Builds a reproducible, non-destructive ingestion pipeline that creates:
    data/freshness/

containing the 20-class expanded freshness dataset (10 fruits x 2 states).

Supports:
    --dry-run
    --verify
    --output
    --force
    --inspect

Guarantees:
- Never modifies data/raw/dataset/dataset/ or production checkpoints.
- Performs SHA256 exact deduplication and pHash near-duplication checks.
- Enforces strict split isolation (no leakage across train/valid/test).
- Generates data/freshness/class_mapping.json, metadata.json,
  dataset_manifest.json, and reports/*.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.freshness_dataset_builder import (
    load_freshness_config,
    collect_all,
    deduplicate_exact,
    split_by_class,
    find_near_duplicates,
    validate_canonical_dataset,
    format_inspection_report,
    inspect_all_sources,
    record_production_hashes,
    build_canonical_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_expanded_freshness_dataset")


def print_summary(summary: dict) -> None:
    """Print the final assembly summary."""
    print()
    print("=" * 60)
    print("FRESHNESS DATASET EXPANSION â€” INGESTION RESULT")
    print("=" * 60)
    print("Canonical dataset:")
    for key in ("Apple_fresh", "Apple_rotten", "Grape_fresh", "Grape_rotten",
                "Kiwi_fresh", "Kiwi_rotten", "Mango_fresh", "Mango_rotten",
                "Orange_fresh", "Orange_rotten", "Strawberry_fresh",
                "Strawberry_rotten", "banana_fresh", "banana_rotten",
                "cherry_fresh", "cherry_rotten", "chickoo_fresh",
                "chickoo_rotten", "guava_fresh", "guava_rotten"):
        class_count = summary.get("class_counts", {}).get(key, {})
        total = class_count.get("total", 0) if isinstance(class_count, dict) else 0
        print(f"  {key}: {total}")
    print("-" * 60)
    print("TOTAL:")
    print(f"    Train: {summary.get('train', 0)}")
    print(f"    Valid: {summary.get('valid', 0)}")
    print(f"    Test:  {summary.get('test', 0)}")
    print(f"  Exact duplicates: {summary.get('exact_duplicates_removed', 0)}")
    print(f"  Near duplicates:  {summary.get('near_duplicates_marked', 0)}")
    print(f"  Rejected:  {summary.get('rejected_count', 0)}")
    print(f"  Cross-split leakage: {summary.get('leakage_check', 'PENDING')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare expanded 20-class freshness dataset.")
    parser.add_argument("--output", type=str, default="data/freshness",
                        help="Output freshness directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configs/freshness_sources.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform dry run without writing files")
    parser.add_argument("--inspect", action="store_true",
                        help="Run source inspection only")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing dataset manifest and files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory if present")

    args = parser.parse_args()
    output = ROOT_DIR / args.output

    # Record production hashes before processing
    logger.info("Recording production hash baseline...")
    record_production_hashes()

    # Optional: inspect sources
    if args.inspect:
        print("\n=== SOURCE INSPECTION ===")
        reports = inspect_all_sources(load_freshness_config(Path(args.config) if args.config else None))
        for key, report in reports.items():
            print()
            print(format_inspection_report(report))
        return

    # Verify existing dataset
    if args.verify:
        if not (output / "dataset_manifest.json").exists():
            logger.error("Verification failed: Manifest %s does not exist!", output / "dataset_manifest.json")
            sys.exit(1)
        success = validate_canonical_dataset(output)
        if not success:
            sys.exit(1)
        print("Dataset verification successful.")
        return

    # Build canonical dataset
    result = build_canonical_dataset(
        config_path=Path(args.config) if args.config else None,
        output_dir=output,
        dry_run=args.dry_run,
        force=args.force,
    )

    if result.get("status") == "skipped":
        logger.error("Build skipped: %s", result.get("reason"))
        sys.exit(1)

    if not args.dry_run:
        # Validate after real build
        success = validate_canonical_dataset(output)
        if not success:
            logger.error("Validation failed after build.")
            sys.exit(1)

    display_summary = {
        **result,
        "class_counts": {},
    }
    if not args.dry_run and (output / "metadata.json").exists():
        with open(output / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        display_summary["class_counts"] = meta.get("class_summary", {})

    print_summary(display_summary)


if __name__ == "__main__":
    main()
