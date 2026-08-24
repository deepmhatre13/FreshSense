"""Real-image end-to-end validation through the LIVE production API.

Sends held-out TEST-set images (never seen in training) for every supported
fruit to POST /api/v1/inference/image and records freshness + shelf-life.
Distinguishes between:  (1) correct-fruit+correct-freshness -> TP,
  (2) wrong-detected-fruit -> data_not_available (correct contract),
  (3) no-detection -> counted separately.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

API = "http://127.0.0.1:8000/api/v1/inference/image"
TEST_BASE = ROOT / "data" / "freshness" / "test"
IMGS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_PER_CLASS = 5  # keep runtime sane on CPU


def load_class_mapping():
    cm = json.loads((ROOT / "data" / "freshness" / "class_mapping.json").read_text())
    ordered = [None] * len(cm)
    for k, v in cm.items():
        ordered[int(k)] = v
    return [c for c in ordered if c]


def parse_fruit_class(name):
    low = name.lower()
    for suffix, state in (("_fresh", "fresh"), ("_rotten", "rotten")):
        if low.endswith(suffix):
            return name[: -len(suffix)], state
    return name, "unknown"


def normalize_fruit(name):
    return (name or "").strip().lower()


def main():
    results = defaultdict(lambda: {
        "tp": 0, "total": 0, "no_detection": 0,
        "states": [], "days": [], "status": [], "det_fruits": [],
    })
    for cls_name in load_class_mapping():
        fruit, true_state = parse_fruit_class(cls_name)
        d = TEST_BASE / cls_name
        if not d.exists():
            continue
        sent = 0
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMGS:
                continue
            if sent >= MAX_PER_CLASS:
                break
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            try:
                r = requests.post(API, files={"image": ("img.jpg", buf.tobytes())}, timeout=120)
            except Exception as e:
                print(f"  REQUEST FAILED {fruit}: {e}")
                break
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} for {fruit}: {r.text[:120]}")
                continue
            data = r.json()
            cell = results[(fruit, true_state)]
            if not data.get("fruits"):
                cell["no_detection"] += 1
                print(f"  no YOLO detection for {cls_name} sample ({fruit})")
                continue
            f = data["fruits"][0]
            pred = f["freshness"]
            det_fruit = f.get("fruit", "")
            cell["total"] += 1
            cell["states"].append(pred)
            cell["det_fruits"].append(det_fruit)
            shelf = f.get("shelf_life") or {}
            cell["days"].append(shelf.get("remaining_days"))
            cell["status"].append(shelf.get("shelf_life_status"))
            # TP = YOLO detects the correct fruit AND freshness label matches
            if normalize_fruit(det_fruit) == normalize_fruit(fruit) and pred == true_state:
                cell["tp"] += 1
            sent += 1

    print("\n===== REAL-IMAGE API VALIDATION (held-out test images via live API) =====")
    summary = {}
    for (fruit, true_state), cell in sorted(results.items()):
        rec = cell["tp"] / cell["total"] if cell["total"] else 0.0
        summary[f"{fruit}_{true_state}"] = {
            "n": cell["total"],
            "no_detection": cell["no_detection"],
            "recall": round(rec, 3),
            "predicted_states": cell["states"],
            "detected_fruits": cell["det_fruits"],
            "shelf_statuses": cell["status"],
            "remaining_days_sample": [d for d in cell["days"] if d is not None][:3],
        }
        print(f"{fruit:14s} true={true_state:7s} n={cell['total']:2d} "
              f"no_det={cell['no_detection']:2d} recall={rec:.2f} "
              f"states={cell['states']}")

    out = ROOT / "reports" / "freshness" / "real_image_validation.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

