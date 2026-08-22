#!/usr/bin/env python3
"""Read-only audit of the SmartFreshAI V3 detection dataset.

Reuses the existing, battle-tested audit implementation in
``scripts/audit_detection_dataset.py`` (same quality checks as V2) but targets
the freshly built ``data/detection_v3``. Produces:

    reports/detection_v3_audit.json
    reports/detection_v3_audit.md

The audit is strictly non-destructive: it never writes to ``data/detection_v3``
beyond reading it, and never touches ``data/detection`` or ``best.pt``.

If ``data/detection_v3`` does not (yet) exist -- e.g. because the V3 review gate
is still blocked -- the script reports that status instead of fabricating a
dataset.

Usage:
    python scripts/audit_detection_v3.py
    python scripts/audit_detection_v3.py --data-dir data/detection_v3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.audit_detection_dataset import audit_dataset, _find_dataset_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/detection_v3")
DEFAULT_JSON_OUT = Path("reports/detection_v3_audit.json")
DEFAULT_MD_OUT = Path("reports/detection_v3_audit.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the SmartFreshAI V3 dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="V3 dataset root to audit")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUT,
                        help="JSON report path")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD_OUT,
                        help="Markdown report path")
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir.is_absolute() else _REPO_ROOT / args.data_dir
    root = _find_dataset_root(data_dir)
    if root is None or not (root / "data.yaml").is_file():
        logger.warning("V3 dataset not found at %s (V3 gate may still be blocked). "
                       "No dataset was audited and nothing was written.", data_dir)
        summary = {
            "status": "v3_dataset_not_available",
            "audit_run": False,
            "reason": "data/detection_v3 does not exist; build it after the V3 review gate passes.",
            "data_dir": str(data_dir),
            "output_json": str(args.output),
            "output_md": str(args.markdown),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"V3 dataset not available: {data_dir}")
        return 0

    json_out = args.output if args.output.is_absolute() else _REPO_ROOT / args.output
    md_out = args.markdown if args.markdown.is_absolute() else _REPO_ROOT / args.markdown
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    report = audit_dataset(root, json_out, md_out)
    logger.info("V3 audit complete. JSON=%s MD=%s", json_out, md_out)
    # Surface a few headline numbers for quick reading.
    ss = report.get("split_summary", {})
    print("V3 AUDIT COMPLETE")
    print(f"  images: {ss.get('total_images', '-')}")
    print(f"  objects: {ss.get('total_objects', '-')}")
    print(f"  json: {json_out}")
    print(f"  md:   {md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
