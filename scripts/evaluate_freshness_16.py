"""Evaluate the 16-class freshness checkpoint on the held-out test set.

This script:
  - Loads models/checkpoints/freshness_efficientnet_b0_16class.pth via the
    production Predictor (the exact path the API uses).
  - Runs it on data/freshness/test/* for all supported fruits.
  - Computes accuracy, per-fruit fresh recall, per-fruit rotten recall and the
    fresh<->rotten confusion for every fruit.
  - Reports the uncertainty (top-1 softmax) calibration on VALIDATION samples
    (never the test set) so the production uncertainty policy is defensible.
  - Writes reports/freshness/ artifacts.

NOTE: production best_model.pth is NEVER touched by this script. The 16-class
checkpoint is evaluated and only later explicitly promoted.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import Predictor
from src.freshness import ID_TO_CLASS

IMGS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CHECKPOINT = ROOT / "models" / "checkpoints" / "freshness_efficientnet_b0_16class.pth"
REPORTS = ROOT / "reports" / "freshness"


def load_class_mapping() -> list:
    cm = json.loads((ROOT / "data" / "freshness" / "class_mapping.json").read_text())
    ordered = [None] * len(cm)
    for k, v in cm.items():
        ordered[int(k)] = v
    return [c for c in ordered if c]


def parse_fruit_class(class_name: str):
    """Return (fruit, state) for a 16-class name like 'Apple_fresh'."""
    lower = class_name.lower()
    for suffix, state in (("_fresh", "fresh"), ("_rotten", "rotten"), ("_stale", "stale")):
        if lower.endswith(suffix):
            return class_name[: -len(suffix)], state
    return class_name, "unknown"
def main() -> int:
    if not CHECKPOINT.exists():
        print("ERROR: 16-class checkpoint not found:", CHECKPOINT)
        return 1
    class_names = load_class_mapping()
    predictor = Predictor(checkpoint_path=str(CHECKPOINT))
    print("Loaded predictor:", predictor.num_classes, "classes, device", predictor.device)

    test_base = ROOT / "data" / "freshness" / "test"
    per_fruit = defaultdict(lambda: {"fresh_tp": 0, "fresh_total": 0,
                                     "rotten_tp": 0, "rotten_total": 0,
                                     "fresh_confused_rotten": 0,
                                     "rotten_confused_fresh": 0,
                                     "uncertain": []})
    y_true, y_pred = [], []
    for i, cls_name in enumerate(class_names):
        fruit, state = parse_fruit_class(cls_name)
        d = test_base / cls_name
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMGS:
                continue
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            res = predictor.predict(frame)
            pred_state = res.freshness_class
            cell = per_fruit[fruit]
            cell["fresh_total" if state == "fresh" else "rotten_total"] += 1
            if state == "fresh":
                if pred_state == "fresh":
                    cell["fresh_tp"] += 1
                elif pred_state == "rotten":
                    cell["fresh_confused_rotten"] += 1
            else:
                if pred_state == "rotten":
                    cell["rotten_tp"] += 1
                elif pred_state == "fresh":
                    cell["rotten_confused_fresh"] += 1
            cell["uncertain"].append(1 if pred_state == "uncertain" else 0)
            y_true.append(i)
            y_pred.append(res.predicted_class_index)

    print("\n===== PER-FRUIT TEST RECALL =====")
    summary = {}
    for fruit in sorted(per_fruit):
        c = per_fruit[fruit]
        f_recall = c["fresh_tp"] / c["fresh_total"] if c["fresh_total"] else 0.0
        r_recall = c["rotten_tp"] / c["rotten_total"] if c["rotten_total"] else 0.0
        summary[fruit] = {
            "fresh_recall": round(f_recall, 4),
            "rotten_recall": round(r_recall, 4),
            "fresh_confused_rotten": c["fresh_confused_rotten"],
            "rotten_confused_fresh": c["rotten_confused_fresh"],
        }
        n = c["fresh_total"] + c["rotten_total"]
        u = sum(c["uncertain"])
        print(f"{fruit:14s} n={n:3d} fresh_recall={f_recall:.3f} "
              f"rotten_recall={r_recall:.3f} uncertain={u}")

    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0
    print("\nTest accuracy (16-class top-1):", round(acc, 4))

    # Uncertainty / confidence calibration on the VALIDATION set (not test).
    collect_val_freshness_probs(predictor, class_names)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "per_fruit_recall.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS / "evaluation_summary.md").write_text(
        build_markdown(summary), encoding="utf-8")
    print("\nArtifacts written under", REPORTS)
    return 0


def collect_val_freshness_probs(predictor, class_names):
    """Return softmax top-1 confidence values on the validation split."""
    val_probs = []
    val_base = ROOT / "data" / "freshness" / "valid"
    for c_name in class_names:
        d = val_base / c_name
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMGS:
                continue
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            res = predictor.predict(frame)
            val_probs.append(res.confidence)
    if val_probs:
        arr = np.array(val_probs)
        print("\nUncertainty validation (VALIDATION set, NOT test):")
        print("  median conf:", round(float(np.median(arr)), 4),
              "| mean conf:", round(float(arr.mean()), 4))
        for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            frac = float(np.mean(arr < t))
            print(f"  p(conf<{t:.1f}) = {frac:.4f}")
    return val_probs


def build_markdown(summary: dict) -> str:
    lines = [
        "# Freshness Model Evaluation Summary",
        "",
        "Model: EfficientNet-B0 (16-class), checkpoint `freshness_efficientnet_b0_16class.pth`",
        "",
        "## Per-fruit held-out test recall (fresh/rotten)",
        "",
        "| fruit | fresh_recall | rotten_recall | fresh_to_rotten | rotten_to_fresh |",
        "|---|---|---|---|---|",
    ]
    for fruit in sorted(summary):
        s = summary[fruit]
        lines.append(
            f"| {fruit} | {s['fresh_recall']:.3f} | {s['rotten_recall']:.3f} "
            f"| {s['fresh_confused_rotten']} | {s['rotten_confused_fresh']} |"
        )
    lines += [
        "",
        "**Fresh/rotten confusion** is the count of test images where a fresh "
        "sample was labelled rotten (or vice-versa). A validated freshness "
        "prediction may be fresh, rotten, or uncertain.",
        "",
        "> Freshness is an ML classification result only for fruits covered by "
        "the trained freshness model. Fruits without a validated model return "
        "data_not_available and shelf-life is not estimated.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())