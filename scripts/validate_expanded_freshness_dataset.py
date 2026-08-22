"""Validation script for the 20-class expanded freshness dataset (data/freshness/).

Validates (via src/data/freshness_dataset_builder.validate_canonical_dataset):
    1. All 20 canonical class directories exist for train, valid, test.
    2. class_mapping.json matches canonical taxonomy (0..19).
    3. Every accepted image is readable, no zero-byte files.
    4. No exact duplicates across splits (SHA256), no SHA256 leakage.
    5. Near-duplicate review report exists.
    6. Provenance / license / source-url present for every accepted entry.
    7. No ambiguous labels silently accepted.
    8. Immutability: data/raw/dataset/dataset/, models/checkpoints/best_model.pth,
       models/detection/detector/weights/best.pt unchanged vs. recorded baseline.

Exit code 0 on success, non-zero on any failure (deterministic).

Usage:
    python scripts/validate_expanded_freshness_dataset.py
    python scripts/validate_expanded_freshness_dataset.py --dataset data/freshness
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.freshness_dataset_builder import validate_canonical_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_expanded_freshness_dataset")


def validate_dataset(dataset_dir: Path) -> bool:
    """Validate the canonical expanded freshness dataset.

    Returns True if all checks pass, False otherwise.
    """
    result = validate_canonical_dataset(dataset_dir)
    all_pass = result.get("all_pass", False)

    print("\n--------------------------------------------------------")
    print("EXPANDED DATASET VALIDATION REPORT (PHASE 4A)")
    print("--------------------------------------------------------")
    checks = result.get("checks", {})
    for check_name, check in sorted(checks.items()):
        status = "PASS" if check.get("passed") else "FAIL"
        detail = check.get("detail", "")
        print(f"  [{status}] {check_name}")
        if detail and not check.get("passed"):
            print(f"           {detail}")

    print("--------------------------------------------------------")
    print(f"Total Images Scanned   : {result.get('total_images_scanned', 0)}")
    if all_pass:
        print("STATUS               : PASS (All checks successful)")
    else:
        print("STATUS               : FAIL (One or more checks failed)")
    print("--------------------------------------------------------\n")
    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate expanded freshness dataset.")
    parser.add_argument("--dataset", type=str, default="data/freshness",
                        help="Path to data/freshness directory")
    args = parser.parse_args()

    logger.info("Starting validation of expanded freshness dataset at %s", args.dataset)
    success = validate_dataset(Path(args.dataset))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())