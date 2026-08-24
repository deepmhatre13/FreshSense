"""Independently count data/freshness on the filesystem and report class
distribution + missing/low-data classes (Phase 12 / Phase 13).

Does NOT trust metadata.json / dataset_manifest.json: counts the filesystem
first, then compares against both metadata files.

Usage:
    python scripts/report_freshness_class_distribution.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.freshness_dataset_builder import CANONICAL_CLASS_MAPPING

MIN_PER_CLASS = 50          # training-readiness threshold
ABSOLUTE_MIN = 20           # below this -> critical
MIN_VAL_TEST = 10           # minimum valid/test coverage per class


def count_filesystem(data_dir: Path) -> dict:
    counts = {}
    for cls in CANONICAL_CLASS_MAPPING.values():
        counts[cls] = {}
        total = 0
        for split in ("train", "valid", "test"):
            d = data_dir / split / cls
            n = sum(1 for p in d.glob("*") if p.is_file()) if d.exists() else 0
            counts[cls][split] = n
            total += n
        counts[cls]["total"] = total
    return counts


def main() -> int:
    data_dir = ROOT_DIR / "data" / "freshness"
    print("=" * 70)
    print("SMARTFRESHAI - FRESHNESS DATASET CLASS DISTRIBUTION (FILESYSTEM)")
    print("=" * 70)

    if not data_dir.exists():
        print("data/freshness does NOT exist. Build has not run.")
        return 1

    fs = count_filesystem(data_dir)

    header = "{:<22}{:>8}{:>8}{:>8}{:>8}".format(
        "CLASS", "TRAIN", "VALID", "TEST", "TOTAL")
    print("\n" + header)
    print("-" * 54)
    grand = {"train": 0, "valid": 0, "test": 0}
    for cls in CANONICAL_CLASS_MAPPING.values():
        c = fs[cls]
        print("{:<22}{:>8}{:>8}{:>8}{:>8}".format(
            cls, c["train"], c["valid"], c["test"], c["total"]))
        for s in ("train", "valid", "test"):
            grand[s] += c[s]
    print("-" * 54)
    tot = sum(grand.values())
    print("{:<22}{:>8}{:>8}{:>8}{:>8}".format(
        "TOTAL", grand["train"], grand["valid"], grand["test"], tot))

    # ---- metadata / manifest consistency ----
    meta_ok = False
    manifest_ok = False
    n_manifest_entries = 0
    meta_path = data_dir / "metadata.json"
    manifest_path = data_dir / "dataset_manifest.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        summary = meta.get("class_summary", {})
        mismatches = []
        for cls in CANONICAL_CLASS_MAPPING.values():
            m = summary.get(cls, {})
            for split in ("train", "valid", "test"):
                if int(m.get(split, -1)) != fs[cls][split]:
                    mismatches.append("%s.%s: metadata=%s filesystem=%s" % (
                        cls, split, m.get(split), fs[cls][split]))
        meta_ok = not mismatches
        print("\nmetadata.json vs filesystem:", "MATCH" if meta_ok else "MISMATCH")
        for mm in mismatches[:10]:
            print("  MISMATCH", mm)
    else:
        print("\nmetadata.json MISSING")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("accepted_entries", [])
        n_manifest_entries = len(entries)
        missing = [e["path"] for e in entries
                   if e.get("path") and not (data_dir / e["path"]).exists()]
        total_expected = sum(c["total"] for c in fs.values())
        if missing or n_manifest_entries != total_expected:
            print("manifest accepted_entries:", n_manifest_entries,
                  "| fs total:", total_expected,
                  "| missing files on disk:", len(missing))
            for mf in missing[:5]:
                print("  MISSING", mf)
        else:
            manifest_ok = True
            print("dataset_manifest.json vs filesystem: MATCH (%d entries)" % n_manifest_entries)
    else:
        print("dataset_manifest.json MISSING")

    # ---- missing / low-data classes ----
    problems = []
    for cls in CANONICAL_CLASS_MAPPING.values():
        c = fs[cls]
        entry = {
            "class": cls,
            "available": c["total"],
            "required": MIN_PER_CLASS,
            "train": c["train"],
            "valid": c["valid"],
            "test": c["test"],
        }
        if c["total"] == 0:
            entry.update({"status": "DATA COLLECTION REQUIRED",
                          "reason": "zero legitimate images in any local source",
                          "source_gap": True})
            problems.append(entry)
        elif c["total"] < ABSOLUTE_MIN:
            entry.update({"status": "DATA COLLECTION REQUIRED",
                          "reason": "insufficient images (%d < %d)" % (c["total"], ABSOLUTE_MIN),
                          "source_gap": True})
            problems.append(entry)
        elif c["total"] < MIN_PER_CLASS:
            entry.update({"status": "DATA COLLECTION REQUIRED",
                          "reason": "below readiness threshold (%d < %d)" % (c["total"], MIN_PER_CLASS),
                          "source_gap": True})
            problems.append(entry)
        elif c["valid"] < MIN_VAL_TEST or c["test"] < MIN_VAL_TEST:
            entry.update({"status": "INSUFFICIENT SPLIT COVERAGE",
                          "reason": "valid=%d, test=%d below %d" % (c["valid"], c["test"], MIN_VAL_TEST),
                          "source_gap": False})
            problems.append(entry)

    print("\n--- CLASSES WITH INSUFFICIENT DATA ---")
    if not problems:
        print("None. All 20 classes meet thresholds.")
    for pe in problems:
        print("  %s: available=%d (train=%d, valid=%d, test=%d) -> %s" % (
            pe["class"], pe["available"], pe["train"], pe["valid"],
            pe["test"], pe["status"]))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filesystem_counts": fs,
        "totals": {**grand, "total": tot},
        "metadata_matches_filesystem": meta_ok,
        "manifest_matches_filesystem": manifest_ok,
        "manifest_entry_count": n_manifest_entries,
        "missing_or_low_classes": problems,
    }
    out = ROOT_DIR / "reports" / "freshness_missing_classes.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nReport saved to:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
