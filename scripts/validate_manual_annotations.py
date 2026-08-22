#!/usr/bin/env python3
"""Validate manually created YOLO annotations for SmartFreshAI Dataset V3.

Checks annotation files are well-formed and correspond to valid images.
Never modifies the original V2 dataset.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CLASS = ["Apple", "Grape", "Kiwi", "Mango", "Orange",
         "Strawberry", "banana", "cherry", "chickoo", "guava"]
VALID_CLASS = set(range(10))


def validate_row(line: str):
    """Validate one YOLO row. Returns (ok, error_msg)."""
    parts = line.strip().split()
    if len(parts) != 5:
        return False, "needs 5 values"
    try:
        vals = [float(p) for p in parts[:5]]
    except ValueError:
        return False, "non-numeric"
    cls_id, cx, cy, w, h = vals
    if cls_id not in VALID_CLASS:
        return False, f"bad class {cls_id}"
    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        return False, "coords out of range"
    if w <= 0 or h <= 0:
        return False, "zero dims"
    if cx + w / 2 > 1 or cy + h / 2 > 1 or cx - w / 2 < 0 or cy - h / 2 < 0:
        return False, "beyond image"
    return True, None


def validate_dir(directory: Path):
    """Validate all annotation files in directory."""
    img_dir = directory / "images"
    lbl_dir = directory / "labels"
    res = {"total": 0, "valid": 0, "invalid": 0, "errors": [],
           "missing": [], "ann": []}
    if not img_dir.exists():
        return res
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in exts)
    res["total"] = len(imgs)
    for ip in imgs:
        lp = directory / "labels" / (ip.stem + ".txt")
        if not ip.exists():
            res["missing"].append(str(ip))
            res["invalid"] += 1
            continue
        res["valid"] += 1
        if not lp.exists():
            res["errors"].append(f"missing label {ip.name}")
            res["invalid"] += 1
            continue
        ann = []
        bad = False
        for ln in lp.read_text(encoding="utf-8").split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            ok, err = validate_row(ln)
            if not ok:
                res["errors"].append(f"{ip.name}: {err}")
                bad = True
                res["invalid"] += 1
                break
            p = ln.split()
            a = int(p[0])
            ann.append({"cid": a, "name": CLASS[a] if a < len(CLASS) else "?",
                        "yolo": [a, float(p[1]), float(p[2]),
                                 float(p[3]), float(p[4])]})
        if not bad:
            res["ann"].append({"image": ip.name, "annotations": ann,
                               "status": "valid"})
    return res


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 \
        else "reports/audit_review/manual_annotations"
    target = Path(arg)
    res = validate_dir(target)
    out = Path("reports/audit_review/manual_annotation_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"total:{res['total']} valid:{res['valid']} "
          f"invalid:{res['invalid']} errors:{len(res['errors'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())