#!/usr/bin/env python3
"""Create a deterministic, physical-fruit-grouped train/val/test split.

Phase 5 - Evaluable real-world dataset foundation.

CRITICAL RULE enforced here:
    Images belonging to the same physical fruit (``physical_fruit_id``) MUST
    NEVER appear in more than one split. This script groups by physical fruit
    by construction and then re-verifies with an independent leakage check.
    Any violation aborts with a non-zero exit code.

Usage:
    python scripts/create_dataset_split.py --manifest data/real_world/manifest.csv

Outputs (into --out-dir, default ``data/real_world/splits``):
    train.csv, val.csv, test.csv   - canonical split manifests
    split_report.json              - machine-readable split summary

Exit codes:
    0 - split successful, no leakage
    1 - manifest invalid, leakage found, or split failed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.real_world_schema import (  # noqa: E402
    CanonicalRecord,
    find_physical_fruit_leakage,
    find_session_leakage,
    load_canonical_manifest,
    split_manifest_by_physical_fruit,
    validate_canonical_manifest,
    write_manifest_csv,
)

logger = logging.getLogger(__name__)


def _print_finding_counts(report) -> None:
    """Print a concise summary of the manifest validation findings."""
    print("Manifest validation:")
    print(f"  rows         : {report.total_rows}")
    print(f"  valid rows   : {report.valid_rows}")
    print(f"  physical fruits   : {report.physical_fruit_count}")
    print(f"  capture sessions  : {report.capture_session_count}")
    print(f"  classes      : {sorted(report.class_distribution)}")
    print(f"  class distribution : {report.class_distribution}")
    print(f"  imbalance ratio (max/min): {report.imbalance_ratio:.2f} "
          f"({'balanced' if report.is_balanced else 'IMBALANCED'})")
    print(f"  errors       : {report.error_count}")
    print(f"  warnings     : {report.warning_count}")
    if report.missing_required:
        print(f"  missing required metadata: {len(report.missing_required)}")
    if report.invalid_labels:
        print(f"  invalid labels: {len(report.invalid_labels)}")
    if report.invalid_values:
        print(f"  invalid values: {len(report.invalid_values)}")
    if report.duplicate_image_ids:
        print(f"  duplicate image ids: {len(report.duplicate_image_ids)}")
    if report.exact_duplicate_files:
        print(f"  exact duplicate files: {len(report.exact_duplicate_files)}")
    if report.missing_image_files:
        print(f"  missing image files: {len(report.missing_image_files)}")
    if report.impossible_combinations:
        print(f"  impossible metadata combinations: {len(report.impossible_combinations)}")
def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deterministic physical-fruit-grouped dataset splitter (Phase 5)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/real_world/manifest.csv"),
        help="Path to the canonical manifest (CSV or JSON).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/real_world"),
        help="Dataset root; image_path values in the manifest are resolved "
        "relative to this directory for validation checks.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/real_world/splits"),
        help="Directory to write train.csv / val.csv / test.csv and "
        "split_report.json.",
    )
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio.")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the manifest validation step (split only).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manifest_path = args.manifest
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        print(
            "No real-world benchmark exists yet. Collect labeled data per "
            "docs/REAL_WORLD_DATASET.md, write the manifest, then re-run.",
            file=sys.stderr,
        )
        return 1

    records: List[CanonicalRecord] = load_canonical_manifest(manifest_path)
    if not records:
        print(f"ERROR: manifest {manifest_path} contains no rows.", file=sys.stderr)
        return 1
    print(f"Loaded {len(records)} rows from {manifest_path}")

    # ------------------------------------------------------------------
    # 1. Validate the manifest.
    # ------------------------------------------------------------------
    report = validate_canonical_manifest(records, data_root=args.data_dir)
    _print_finding_counts(report)
    if not args.no_validate and not report.is_pass():
        print(
            "ABORT: manifest failed validation with blocking errors "
            "(see findings above). Fix the manifest and re-run.",
            file=sys.stderr,
        )
        return 1
# ------------------------------------------------------------------
    # 2. Grouped physical-fruit split.
    # ------------------------------------------------------------------
    result = split_manifest_by_physical_fruit(
        records,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )
    print("\nSplit ratios:", args.train, args.val, args.test, "seed:", args.seed)
    print("Image counts :", result.counts)
    print("Fruit counts :", result.fruit_counts)

    # ------------------------------------------------------------------
    # 3. Independent leakage verification.
    # ------------------------------------------------------------------
    split_records = {
        "train": result.train,
        "val": result.val,
        "test": result.test,
    }
    fruit_leaks = find_physical_fruit_leakage(split_records)
    session_leaks = find_session_leakage(split_records)
    if fruit_leaks:
        print(
            "ABORT: CRITICAL physical-fruit leakage detected across splits:",
            file=sys.stderr,
        )
        for fruit_id, split_a, split_b in fruit_leaks[:20]:
            print(
                f"  physical_fruit_id={fruit_id} in both {split_a!r} and "
                f"{split_b!r}",
                file=sys.stderr,
            )
        return 1
    print("Leakage check : PASS (no physical_fruit_id spans multiple splits)")
    if session_leaks:
        print(f"NOTE: session leakage (advisory): {len(session_leaks)} session(s) cross splits")

    # ------------------------------------------------------------------
    # 4. Class-balance report.
    # ------------------------------------------------------------------
    print("\n" + result.class_balance_report())

    # ------------------------------------------------------------------
    # 5. Write outputs.
    # ------------------------------------------------------------------
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = write_manifest_csv(result.train, out_dir / "train.csv")
    val_csv = write_manifest_csv(result.val, out_dir / "val.csv")
    test_csv = write_manifest_csv(result.test, out_dir / "test.csv")

    summary = result.to_dict()
    summary["manifest"] = str(manifest_path)
    summary["validation"] = report.to_dict()
    summary["session_leakage_crossing_splits"] = [
        f"{s[0]} ({s[1]} & {s[2]})" for s in session_leaks[:50]
    ]
    summary_path = out_dir / "split_report.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\nWrote:")
    for path in (train_csv, val_csv, test_csv, summary_path):
        print(f"  {path}")
    print("\nSPLIT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())