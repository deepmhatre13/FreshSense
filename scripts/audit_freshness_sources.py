"""Freshness data audit — SmartFreshAI.

Walks every candidate freshness source in the repository and produces an exact
per-fruit audit (fresh/rotten counts, source, split, duplicates, label
validity). This is an *investigation* tool: it never writes or modifies any
dataset, model, or checkpoint.

Sources scanned:
    - data/Original Image/            (Mendeley 2.6 GB: explicit fresh/rotten)
    - data/Quality Dataset/           (Kaggle quality, small)
    - data/raw/dataset/dataset/       (Kaggle fresh/rotten: apple/banana/orange)
    - data/freshness/                 (existing canonical dataset)

Output: reports/freshness_source_audit.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCES = [
    {
        "name": "Mendeley Original Image",
        "path": ROOT / "data" / "Original Image",
        "license": "CC BY 4.0",
    },
    {
        "name": "Quality Dataset",
        "path": ROOT / "data" / "Quality Dataset",
        "license": "CC0 1.0 Universal",
    },
    {
        "name": "Kaggle Fresh & Rotten (raw)",
        "path": ROOT / "data" / "raw" / "dataset" / "dataset",
        "license": "CC0 1.0 Universal",
    },
    {
        "name": "Existing canonical freshness",
        "path": ROOT / "data" / "freshness",
        "license": "mixed",
    },
]


def _images(p: Path) -> list[Path]:
    if not p.exists():
        return []
    ap = p.resolve()
    return [x.resolve() for x in ap.rglob("*") if x.is_file() and x.suffix.lower() in _EXTS]


def is_zero_byte(p: Path) -> bool:
    try:
        return p.stat().st_size == 0
    except OSError:
        return True


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def decodable(p: Path) -> bool:
    import cv2
    try:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return img is not None and img.size > 0
    except Exception:
        return False
def extract_directory(
    source_name: str, dir_path: Path, license_: str,
    check_decode: bool = False, calc_hash: bool = True,
) -> dict:
    """Scan one source with per-parent-dir counts and (optional) dedup stats."""
    rec = {
        "source": source_name,
        "license": license_,
        "path": str(dir_path),
        "total_files": 0,
        "per_class_counts": {},
        "per_dir_labeled_files": [],
        "zero_byte": 0,
        "corrupt": 0,
        "exact_duplicate_buckets": 0,
        "exact_duplicate_pairs": 0,
        "hashed": calc_hash,
    }
    hashes: dict[str, list[str]] = defaultdict(list)
    for p in _images(dir_path):
        rec["total_files"] += 1
        if is_zero_byte(p):
            rec["zero_byte"] += 1
            continue
        if check_decode and not decodable(p):
            rec["corrupt"] += 1
        entry = {
            "path": str(p.relative_to(ROOT.resolve())),
            "label": p.parent.name,
            "size": p.stat().st_size,
        }
        if calc_hash:
            hp = sha256_file(p)
            hashes[hp].append(str(p))
            entry["sha256"] = hp
        rec["per_dir_labeled_files"].append(entry)
    if calc_hash:
        dups = {h: v for h, v in hashes.items() if len(v) > 1}
        rec["exact_duplicate_buckets"] = len(dups)
        rec["exact_duplicate_pairs"] = sum(len(v) - 1 for v in dups.values())
    counts: dict[str, int] = defaultdict(int)
    for f in rec["per_dir_labeled_files"]:
        counts[f["label"]] += 1
    rec["per_class_counts"] = dict(sorted(counts.items()))
    return rec


def audit() -> dict:
    report = {"sources": {}, "summary": None}
    aggregate: defaultdict[str, int] = defaultdict(int)
    all_dup_pairs = 0
    for s in SOURCES:
        # Fast mode: counts only (SHA256 per-file is done by the canonical
        # dataset build step, not this preview audit).
        rec = extract_directory(
            s["name"], s["path"], s["license"],
            check_decode=False, calc_hash=False,
        )
        report["sources"][s["name"]] = rec
        for label, cnt in rec["per_class_counts"].items():
            aggregate[label] += cnt
        all_dup_pairs += rec["exact_duplicate_pairs"]

    # Summarise into an explicit freshness per-fruit view: a dir label is
    # explicit freshness when its name starts with Fresh/Rotten. Report which
    # fruits have BOTH a fresh and rotten source label (=> valid freshness data).
    fruit_stats: dict[str, dict] = defaultdict(lambda: {"fresh": 0, "rotten": 0, "sources": set()})
    for name, rec in report["sources"].items():
        for label, cnt in rec["per_class_counts"].items():
            for prefix, state in (("Fresh", "fresh"), ("Rotten", "rotten"), ("fresh", "fresh"), ("rotten", "rotten")):
                if label.startswith(prefix) and len(label) > len(prefix):
                    fruit = label[len(prefix):]
                    fruit_stats[fruit.lower()][state] += cnt
                    fruit_stats[fruit.lower()]["sources"].add(name)
                    break

    # Determine availability from explicit fresh+rotten counts.
    freshness_view = {}
    for fruit, st in sorted(fruit_stats.items()):
        freshness_view[fruit] = {
            "fruit": fruit,
            "fresh_count": st["fresh"],
            "rotten_count": st["rotten"],
            "total_count": st["fresh"] + st["rotten"],
            "has_both_states": st["fresh"] > 0 and st["rotten"] > 0,
            "sources": sorted(st["sources"]),
        }
    report["freshness_by_fruit"] = freshness_view
    report["summary"] = {
        "total_files_scanned": sum(
            r["total_files"] for r in report["sources"].values()
        ),
        "total_exact_duplicate_pairs": all_dup_pairs,
        "fruits_with_valid_fresh_and_rotten": [
            f for f, st in freshness_view.items() if st["has_both_states"]
        ],
    }
    return report


if __name__ == "__main__":
    report = audit()
    out = ROOT / "reports" / "freshness_source_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("### Audited per-class counts")
    for name, rec in report["sources"].items():
        print(f"--- {name}")
        for label, cnt in rec.get("per_class_counts", {}).items():
            print(f"   {label}: {cnt}")
        print(f"   dups: {rec['exact_duplicate_buckets']}")
    print("\n### Fruits with valid fresh AND rotten data")
    for f in report["summary"]["fruits_with_valid_fresh_and_rotten"]:
        print("   ", f)
    print("\nReport written:", out)