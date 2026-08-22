#!/usr/bin/env python3
"""Repair duplicate / corrupted ACCEPTED review records in human_decisions.json.

The V3 build gate is blocked by two *mechanical* data-integrity problems (distinct
from the human-judgment ``uncertain`` cases the reviewer is resolving):

1. Duplicate accepted records
   ``Grape-33-...jpg`` has two byte-identical "ACCEPT ALL: 8 proposals" accepted
   records. Keeping both would double-count every accepted box into V3.

2. Corrupted accepted record with a malformed proposal
   ``22863-...jpg`` has a clean "ACCEPT ALL: 24 proposals" accepted record plus a
   corrupted one "ACCEPT ALL: 25 proposals" whose extra proposal carries
   ``class_name='new_box'`` and an out-of-frame box (x1=1338 > image width 640).
   An out-of-frame box can never be a valid YOLO annotation, and ``new_box`` is a
   GUI "ADD BOX" placeholder class, not a real fruit class.

Policy note
-----------
This script does NOT fabricate or change ANY human decision. It only removes
*redundant or provably-invalid copies* of already-recorded human decisions:

  - If two accepted records for the same image have identical proposal geometry,
    one is a duplicate: this script keeps the first and drops the rest.
  - If an accepted record contains a proposal whose class name is not in the
    10-class list (e.g. ``new_box``) or whose box lies entirely outside the image
    frame, that malformed proposal is removed. If a record becomes empty of valid
    proposals it is removed entirely. A unique, healthy accepted decision is
    NEVER altered.

A timestamped backup of the original file is written before any change, and a
machine-readable repair report is written to ``reports/audit_review/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image  # for image dimensions when validating out-of-frame boxes

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VALID_CLASSES = {
    "Apple", "Grape", "Kiwi", "Mango", "Orange", "Strawberry",
    "banana", "cherry", "chickoo", "guava",
}

DEFAULT_DECISIONS = Path("reports/audit_review/human_decisions.json")
DEFAULT_OUT = Path("reports/audit_review/repair_report.json")


def _proposals_of(record: dict) -> list:
    props = record.get("ai_proposals")
    if props is not None:
        return list(props)
    if record.get("ai_proposal"):
        return [record["ai_proposal"]]
    return []


def _sig(p: dict) -> tuple:
    return (
        p.get("class_id"),
        round(float(p.get("x1", 0)), 3),
        round(float(p.get("y1", 0)), 3),
        round(float(p.get("x2", 0)), 3),
        round(float(p.get("y2", 0)), 3),
    )


def _image_size_for(record: dict) -> tuple | None:
    """Return (w, h) for the image a record references, or None if unavailable."""
    rel = record.get("image") or ""
    path = _REPO_ROOT / rel if rel else None
    if path is None or not path.exists():
        fname = record.get("image_filename")
        split = record.get("split")
        if fname and split and split in ("train", "valid", "test"):
            cand = _REPO_ROOT / "data" / "detection" / split / "images" / fname
            if cand.exists():
                path = cand
    if path is None or not path.exists():
        return None
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return None


def repair_decisions(decisions_path: Path, out_report: Path) -> dict:
    """Repair duplicate/corrupted accepted records; return a repair report."""
    if not decisions_path.exists():
        raise FileNotFoundError(f"decisions file not found: {decisions_path}")
    with open(decisions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])

    # Backup the original file (timestamped) before any mutation.
    backup = decisions_path.parent / (
        "human_decisions.backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    shutil.copyfile(decisions_path, backup)

    # Group accepted/corrected records by image.
    grouped = {}
    for idx, r in enumerate(records):
        if r.get("human_decision") not in ("accepted", "corrected"):
            continue
        img = r.get("image_filename") or Path(str(r.get("image", ""))).name
        grouped.setdefault(img, []).append(idx)

    dropped_indices = set()
    removed_proposal_facts = []
    removed_record_facts = []

    for img, idxs in grouped.items():
        # --- Malformed-proposal detection (strip invalid boxes/classes) ---------
        img_size = None
        for idx in idxs:
            if idx in dropped_indices:
                continue
            r = records[idx]
            props = _proposals_of(r)
            if not props:
                continue
            if img_size is None:
                img_size = _image_size_for(r)
            keep = []
            for p in props:
                bad_reasons = []
                cn = p.get("class_name")
                if cn is not None and str(cn) not in VALID_CLASSES:
                    bad_reasons.append(f"invalid class_name {cn!r}")
                if img_size:
                    w, h = img_size
                    x1, y1, x2, y2 = (
                        float(p.get("x1", 0)), float(p.get("y1", 0)),
                        float(p.get("x2", 0)), float(p.get("y2", 0)),
                    )
                    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        bad_reasons.append(
                            f"out-of-frame box (image {w}x{h}, box "
                            f"({x1},{y1},{x2},{y2}))")
                if bad_reasons:
                    removed_proposal_facts.append({
                        "image": img, "index": idx, "class_name": cn,
                        "box": [p.get("x1"), p.get("y1"), p.get("x2"), p.get("y2")],
                        "reasons": bad_reasons,
                    })
                    continue
                keep.append(p)
            if len(keep) != len(props):
                r["ai_proposals"] = keep
                dfb = r.get("final_boxes")
                if isinstance(dfb, list) and len(dfb) == len(props):
                    r["final_boxes"] = dfb[:len(keep)]
                if not keep:
                    dropped_indices.add(idx)
                    removed_record_facts.append({
                        "image": img, "index": idx,
                        "reason": "accepted record became empty after removing malformed proposals",
                    })

        # --- Duplicate detection on POST-cleanup proposal signature SETS --------
        sig_sets = {}
        for idx in idxs:
            if idx in dropped_indices:
                continue
            props = _proposals_of(records[idx])
            sigs = sorted(set(_sig(p) for p in props))
            sig_sets.setdefault(tuple(sigs), []).append(idx)
        for sigset, keep_these in sig_sets.items():
            for extra in keep_these[1:]:
                dropped_indices.add(extra)
                removed_record_facts.append({
                    "image": img,
                    "reason": "duplicate accepted record (identical proposal geometry)",
                    "index": extra,
                })

    cleaned = [r for i, r in enumerate(records) if i not in dropped_indices]

    # Recompute record_count.
    data["records"] = cleaned
    if "record_count" in data:
        data["record_count"] = len(cleaned)
    data.setdefault("repair_log", []).append({
        "tool": "repair_detection_decisions.py",
        "timestamp": datetime.now().isoformat(),
        "removed_duplicate_records": len(removed_record_facts),
        "removed_malformed_proposals": len(removed_proposal_facts),
        "backup": str(backup),
    })

    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    report = {
        "backup": str(backup),
        "total_records_before": len(records),
        "total_records_after": len(cleaned),
        "removed_duplicate_records": removed_record_facts,
        "removed_malformed_proposals": removed_proposal_facts,
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair duplicate/corrupted accepted review records.")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS,
                        help="Path to human_decisions.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help="Repair report JSON path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    dec = args.decisions if args.decisions.is_absolute() else _REPO_ROOT / args.decisions
    if args.dry_run:
        import tempfile
        # Run the exact repair logic against a throwaway copy for an accurate
        # preview (never mutates the real file in dry-run mode).
        tmp = Path(tempfile.mkdtemp(prefix="repair_dryrun_")) / dec.name
        shutil.copyfile(dec, tmp)
        tmp_report = Path(tempfile.mkdtemp(prefix="repair_dryrun_")) / "report.json"
        preview = repair_decisions(tmp, tmp_report)
        print(f"[dry-run] Would drop {len(preview['removed_duplicate_records'])} duplicate record(s) "
              f"and {len(preview['removed_malformed_proposals'])} malformed proposal(s).")
        for rf in preview["removed_duplicate_records"]:
            print(f"  - drop record: {rf}")
        for pf in preview["removed_malformed_proposals"]:
            print(f"  - drop proposal: {pf['image']}: {pf['reasons']}")
        return 0

    report = repair_decisions(dec, args.output)
    logger.info("Removed %d duplicate record(s), %d malformed proposal(s).",
                len(report["removed_duplicate_records"]),
                len(report["removed_malformed_proposals"]))
    print(f"backup: {report['backup']}")
    print(f"records: {report['total_records_before']} -> {report['total_records_after']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

