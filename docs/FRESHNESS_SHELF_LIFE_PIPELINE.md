# Freshness & Shelf-Life Intelligence Pipeline Documentation

## 1. Architecture Map

```
Input Frame (OpenCV BGR Image)
    │
    ▼
Image Quality Assessor (`src/inference/quality.py`)
    │
    ▼
YOLO Object Detector (`src/detection/detector.py`, `models/detection/detector/weights/best.pt`)
    │
    ▼
Fruit Bounding Box Detections (`src/detection/base_detector.py`)
    │
    ▼
Multi-Object Tracker (`src/inference/detection_tracker.py`)
    │
    ▼
Fruit Cropper & Resizer (`src/inference/cropper.py`)
    │
    ▼
Freshness Support Gate (`src/inference/detection_pipeline.py::freshness_supported`)
    ├── Supported (Apple, Banana, Orange) ──► EfficientNet Classifier (`src/inference/predictor.py`)
    └── Unsupported (Grape, Kiwi, Mango, Strawberry, Cherry, Chickoo, Guava) ──► "unknown"
    │
    ▼
Per-Fruit Temporal Stabilizer (`src/inference/stabilizer.py`)
    │
    ▼
Confidence Fusion Engine (`src/inference/confidence_fusion.py`)
    │
    ▼
Shelf-Life Estimator (`src/inference/shelf_life.py` + `fruit_database.json`)
    │
    ▼
Structured Multi-Fruit Result (`src/inference/fruit_result.py::MultiFruitResult`)
```

---

## 2. Component Implementation Status

| Stage | Implementation File | Status | Notes / Mechanism |
|---|---|---|---|
| Image Quality | `src/inference/quality.py` | IMPLEMENTED + TESTED | Assesses brightness, contrast, blur, and motion stability. |
| YOLO Detection | `src/detection/detector.py` | IMPLEMENTED + TESTED | YOLO11n baseline checkpoint (`models/detection/detector/weights/best.pt`), 10 classes. |
| Fruit Cropper | `src/inference/cropper.py` | IMPLEMENTED + TESTED | 8% margin expansion, min side/area gates, resizes to 224x224. |
| Tracker | `src/inference/detection_tracker.py` | IMPLEMENTED + TESTED | Multi-object IoU/centroid assignment across frames. |
| Freshness Classifier | `src/inference/predictor.py` | IMPLEMENTED + TESTED | EfficientNet-B0 trained on fresh/rotten x apple/banana/orange. |
| Confidence Fusion | `src/inference/confidence_fusion.py` | IMPLEMENTED + TESTED | Fuses detector (0.4) and classifier (0.6) confidence scores. |
| Shelf-Life Estimation | `src/inference/shelf_life.py` | IMPLEMENTED + TESTED | Rule/metadata heuristic combining `fruit_database.json` and fused confidence. |

---

## 3. Supported vs Unsupported Fruits

The YOLO detector supports 10 fruit classes:
- **Classifier-supported fruits**: `Apple`, `Banana`, `Orange`
  - Freshness is predicted by the EfficientNet-B0 checkpoint (`models/checkpoints/best_model.pth`).
  - Freshness classes: `fresh`, `stale`, `rotten`.
- **Unsupported fruits**: `Grape`, `Kiwi`, `Mango`, `Strawberry`, `Cherry`, `Chickoo`, `Guava`
  - Explicit fallback: reported as `freshness_class="unknown"` and `is_uncertain=True`.
  - The freshness classifier is **never** invoked on unsupported crops to prevent misclassification.
  - Shelf-life for rotten/unknown items explicitly reports `0-0 days` with basis `metadata_heuristic` or `unavailable`.

---

## 4. Shelf-Life Estimation Logic

Shelf life is a **metadata-driven heuristic** based on domain storage data in `fruit_database.json`. It is **not** an end-to-end ML regression model.

1. If metadata for the fruit is missing:
   - `basis_type = "unavailable"`
   - `min_days = 0, max_days = 0`
2. If fruit is rotten (`freshness_class == "rotten"` or `is_fresh == False`):
   - `basis_type = "metadata_heuristic"`
   - `min_days = 0, max_days = 0`
   - `basis = "Not suitable for storage - consume immediately"`
3. If fruit is fresh:
   - `range = typical_shelf_life_days` from `fruit_database.json` (e.g. `[14, 30]` for Apple)
   - `min_days = int(round(range_min * fused_confidence))`
   - `max_days = int(round(range_max * fused_confidence))`
   - `basis_type = "metadata_heuristic"`

---

## 5. Structured Result Schema (`MultiFruitResult`)

The output schema is structured and JSON-serializable via `to_dict()`:

```json
{
  "fruits": [
    {
      "tracking_id": 0,
      "fruit": "Apple",
      "freshness": "fresh",
      "confidence": 0.973,
      "detection_confidence": 0.933,
      "stabilized_confidence": 0.996,
      "is_uncertain": false,
      "is_locked": false,
      "shelf_life": "14-29 days",
      "bounding_box": {"x1": 61, "y1": 49, "x2": 189, "y2": 176},
      "center": [125, 112]
    }
  ],
  "fruit_count": 1,
  "unidentified_count": 0,
  "frame_width": 640,
  "frame_height": 480
}
```

---

## 6. Performance Benchmarks

Measured on standard CPU execution across sample test images:
- **Average Pipeline Latency**: `371.88 ms`
- **Minimum Latency**: `197.05 ms`
- **Maximum Latency**: `1110.99 ms`
