# FreshSense AI — YOLO Detector Baseline

> **Status: TEMPLATE — metrics not yet populated.**
>
> This document is the authoritative template for the first trusted YOLO
> detection baseline. The numeric values must be filled in **only** from an
> actually-executed training + evaluation run. No metric below has been
> fabricated, and none is implied to be "production ready".

## Pipeline responsibility split

- **YOLO detector** = *what* object is present and *where* (bounding box).
- **EfficientNet classifier** = freshness state (`fresh` / `stale` / `rotten`).
- The two confidence values are kept separate (`detector_confidence` and
  `classifier_confidence`); fusion weights are `detection_weight=0.4` and
  `classification_weight=0.6` (unchanged).

## Dataset (Roboflow Version 1)

| Split | Images | Labels |
|-------|-------:|-------:|
| train | 777 | 777 |
| valid | 218 | 218 |
| test  | 111 | 111 |
| **total** | **1106** | **1106** |

- Workspace: `deepam-mhatre`
- Project: `fruits-test-ajvf8-duncc`
- Version: `1`
- Local YOLO dataset: `data/detection/`
- Classes: 10

## Class names

Apple, Grape, Kiwi, Mango, Orange, Strawberry, banana, cherry, chickoo, guava.

## Model

- Architecture: **YOLO11n** (pretrained)
- Image size: **640**
- Epochs: **50**
- Batch: **16**
- Patience: **10**
- Device: auto (CPU on this machine, CUDA on GPU/Colab)

## Commands

```powershell
# Download
python -m scripts.download_detection_dataset

# Validate
python -m scripts.validate_detection_dataset

# Train
python -m scripts.train_detector

# Validation-set evaluation
python -m scripts.evaluate_detector --model models/detection/detector/weights/best.pt --split val --output reports/detection_evaluation_val.json

# Final test-set evaluation
python -m scripts.evaluate_detector --model models/detection/detector/weights/best.pt --split test --output reports/detection_evaluation_test.json
```

## Validation-set metrics — Not Found

Not found - evaluation report artifacts (e.g., `reports/detection_evaluation_val.json`) do not currently exist in the repository.

## Test-set metrics — Not Found

Not found - evaluation report artifacts (e.g., `reports/detection_evaluation_test.json`) do not currently exist in the repository.

## Per-class results — Not Found

Not found - per-class evaluation artifacts do not currently exist in the repository.

## Known limitations

- The detector recognises **10** fruit classes.
- The freshness classifier currently supports only **apple, banana, orange**
  (six-class taxonomy: fresh/rotten per fruit). Unsupported fruits (grape, kiwi,
  mango, strawberry, cherry, chickoo, guava) are reported with a freshness of
  `unknown` via `freshness_supported()` — **never** silently mapped to a wrong
  freshness class.
- Real-world / webcam performance has not been established. In-domain
  (Roboflow test-set) metrics are **not** proof of real-camera performance.