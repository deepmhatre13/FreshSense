"""Calibrate uncertainty_threshold from validation-set softmax confidences."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import Predictor

IMGS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CHECKPOINT = ROOT / "models" / "checkpoints" / "freshness_efficientnet_b0_16class.pth"


def load_class_mapping():
    cm = json.loads((ROOT / "data" / "freshness" / "class_mapping.json").read_text())
    ordered = [None] * len(cm)
    for k, v in cm.items():
        ordered[int(k)] = v
    return [c for c in ordered if c]


def main():
    class_names = load_class_mapping()
    predictor = Predictor(checkpoint_path=str(CHECKPOINT))
    val_base = ROOT / "data" / "freshness" / "valid"

    confs = []
    correct = []
    for i, cls_name in enumerate(class_names):
        d = val_base / cls_name
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMGS:
                continue
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            res = predictor.predict(frame)
            confs.append(res.confidence)
            pred_idx = res.predicted_class_index
            correct.append(pred_idx == i)

    confs = np.array(confs)
    correct = np.array(correct)
    correct_confs = confs[correct] if len(correct) > 0 else np.array([])

    threshold = float(np.percentile(correct_confs, 5)) if len(correct_confs) > 0 else 0.5
    threshold = max(threshold, 0.30)

    print(f"Validation: {len(confs)} samples, {correct.sum()} correct, acc={correct.mean():.4f}")
    print(f"  median correct conf: {np.median(correct_confs):.4f}")
    print(f"  5th pct correct conf (threshold): {threshold:.4f}")
    print(f"  mean conf: {confs.mean():.4f}")
    for t in (0.30, threshold, 0.50, 0.70):
        print(f"  p(conf<{t:.2f}) = {np.mean(confs < t):.4f}")

    report = {
        "val_total": int(len(confs)),
        "val_correct": int(correct.sum()),
        "val_accuracy": float(correct.mean()) if len(confs) > 0 else 0.0,
        "threshold_method": "5th percentile of correct-prediction softmax confidences",
        "calibrated_threshold": round(threshold, 4),
        "median_correct_conf": float(np.median(correct_confs)) if len(correct_confs) > 0 else 0.0,
        "mean_conf": float(confs.mean()) if len(confs) > 0 else 0.0,
    }
    REPORT_DIR = ROOT / "reports" / "freshness"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "uncertainty_calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nThreshold: {threshold:.4f}")
    print(f"Report: {REPORT_DIR / 'uncertainty_calibration.json'}")


if __name__ == "__main__":
    main()
