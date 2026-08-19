#!/usr/bin/env python3
"""Compare a YOLO evaluation report against the frozen V2 baseline.

Reads two evaluation JSON reports produced by ``scripts/evaluate_detector.py``
and writes a human-readable comparison markdown report with absolute and
percentage deltas for every metric (precision, recall, mAP50, mAP50-95) and
every per-class AP50.

Usage:
    python scripts/compare_yolo_results.py \
        --baseline reports/detection_baseline_test.json \
        --experiment reports/detection_v3_yolo11n_test.json \
        --output reports/yolo/EXP_V3_YOLO11N_COMPARISON.md

This tool is read-only with respect to models and datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


AGG_METRICS = ("precision", "recall", "map50", "map50_95")
CLASSES = [
    "Apple", "Grape", "Kiwi", "Mango", "Orange", "Strawberry",
    "banana", "cherry", "chickoo", "guava",
]


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def build_comparison(baseline: dict, experiment: dict) -> str:
    def find_cls(report: dict, name: str) -> dict:
        per = report.get("per_class", {}) or {}
        for key in per:
            if key.lower() == name.lower():
                return per[key]
        return {}

    lines = []
    lines.append("# V3 (YOLO11n) vs Baseline (YOLO11n) Model Comparison")
    lines.append("")
    lines.append("| Metric | Baseline (YOLO11n/V2) | V3 (YOLO11n) | Abs Delta | % Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for m in AGG_METRICS:
        b = _num(baseline.get(m))
        e = _num(experiment.get(m))
        abs_d = e - b
        pct_d = (abs_d / b * 100.0) if b else float("nan")
        lines.append(
            f"| {m} | {b:.4f} | {e:.4f} | {abs_d:+.4f} | {pct_d:+.2f}% |")
    lines.append("")
    lines.append("## Per-class AP50")
    lines.append("")
    lines.append("| Class | Baseline AP50 | V3 AP50 | Abs Delta | % Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for c in CLASSES:
        b = _num(find_cls(baseline, c).get("ap50"))
        e = _num(find_cls(experiment, c).get("ap50"))
        abs_d = e - b
        pct_d = (abs_d / b * 100.0) if b else float("nan")
        lines.append(
            f"| {c} | {b:.4f} | {e:.4f} | {abs_d:+.4f} | {pct_d:+.2f}% |")
    lines.append("")
    lines.append("> Deltas are computed as V3 - Baseline. The benchmark is the")
    lines.append("> **unchanged V2 TEST set**. A positive delta means V3 improved.")
    return "\n".join(lines)


def find_cls(report: dict, name: str) -> dict:
    per = report.get("per_class", {}) or {}
    for key in per:
        if key.lower() == name.lower():
            return per[key]
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare YOLO reports vs baseline")
    parser.add_argument("--baseline", type=Path, required=False,
                        default=Path("reports/detection_baseline_test.json"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/yolo/EXP_V3_YOLO11N_COMPARISON.md"))
    args = parser.parse_args()

    baseline_path = args.baseline if args.baseline.is_absolute() else _REPO_ROOT / args.baseline
    exp_path = args.experiment if args.experiment.is_absolute() else _REPO_ROOT / args.experiment
    out_path = args.output if args.output.is_absolute() else _REPO_ROOT / args.output

    if not baseline_path.exists():
        sys.stderr.write(f"baseline report not found: {baseline_path}\n")
        return 1
    if not exp_path.exists():
        sys.stderr.write(f"experiment report not found: {exp_path}\n")
        return 1

    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(exp_path, encoding="utf-8") as f:
        experiment = json.load(f)

    md = build_comparison(baseline, experiment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Comparison written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())