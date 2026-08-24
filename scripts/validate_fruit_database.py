"""Validate fruit_database.json against the 10 YOLO-detected classes (Phase 3).

Creates NO data and fixes NOTHING: it reports exactly which of the 10
canonical detected fruits have complete shelf-life metadata in
fruit_database.json, and flags anything missing or invalid.

Usage:
    python scripts/validate_fruit_database.py [--db path/to/fruit_database.json]

Exit code 0 on success; nonzero if any required fruit is missing/incomplete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.fruit_metadata import FruitMetadataDatabase  # noqa: E402

# Canonical 10-class fruit vocabulary produced by the frozen YOLO detector,
# as defined in src/detection/__init__.py (SUPPORTED_CLASSES).
REQUIRED_FRUITS = [
    "Apple",
    "Grape",
    "Kiwi",
    "Mango",
    "Orange",
    "Strawberry",
    "banana",
    "cherry",
    "chickoo",
    "guava",
]

REQUIRED_FIELDS = ("scientific_name", "optimal_storage", "typical_shelf_life_days")


def _validate_range(value) -> Optional[Tuple[int, int]]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        return None
    lo, hi = value
    if lo < 0 or hi <= 0 or lo > hi:
        return None
    return (lo, hi)


def validate(db_path: str) -> int:
    db = FruitMetadataDatabase(db_path)
    print(f"Database           : {Path(db_path).resolve()}")
    print(f"Metadata available : {db.metadata_available}")
    print(f"Validation issues  : {db.validation_issues or 'none'}")
    print(f"Fruits loaded      : {len(db.names())}")
    print()
    print(f"{'Detected class':<12} {'canonical?':<12} {'range':<18} {'status':<14} missing/invalid")
    print("-" * 80)

    failures = 0
    for label in REQUIRED_FRUITS:
        key = label.lower()
        meta = db.get(key)
        if meta is None:
            status = "MISSING"
            failures += 1
            print(f"{label:<12} {'---':<12} {'---':<18} {status:<14} 'fruit_database.json'")
            continue
        rng = _validate_range(meta.typical_shelf_life_days)
        range_str = f"[{rng[0]}, {rng[1]}]" if rng else "INVALID"
        missing = [f for f in REQUIRED_FIELDS if not getattr(meta, f, "")]
        if rng is None or missing:
            status = "INCOMPLETE"
            failures += 1
            problems = ([f"typical_shelf_life_days={meta.typical_shelf_life_days!r}"] if rng is None else []) + missing
            print(f"{label:<12} {'YES':<12} {range_str:<18} {status:<14} {problems}")
            continue
        print(f"{label:<12} {'YES':<12} {range_str:<18} {'OK':<14}")

    print("-" * 80)
    print(f"Required classes   : {len(REQUIRED_FRUITS)} | OK count : {len(REQUIRED_FRUITS) - failures} | failures : {failures}")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate fruit_database.json metadata for the 10 detected fruits.")
    parser.add_argument("--db", default="fruit_database.json", help="Path to fruit_database.json (default: repository root).")
    args = parser.parse_args()
    return validate(args.db)


if __name__ == "__main__":
    raise SystemExit(main())