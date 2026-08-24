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
  - Explicit fallback: reported as `freshness_class="unsupported"` (or `uncertain`
    when the freshness signal is unstable); `is_uncertain=True` where applicable.
  - The freshness classifier is **never** invoked on unsupported crops (prevents
    misclassification), and the pipeline does **NOT** fabricate a freshness or a
    shelf-life for them — `remaining_days=null`.

---

## 4. Shelf-Life Estimation Logic

Shelf life is a **metadata-driven deterministic heuristic** based on the typical
storage range in `fruit_database.json` combined with freshness state and
confidence. It is **not** an end-to-end ML regression model and it is **not** a
validated time-to-event spoilage predictor.

1. Freshness state machine (order matters):
   - **`rotten` / `stale`** → `shelf_life_status="expired"`, `remaining_days=0`.
   - **`unsupported`** → `shelf_life_status="unsupported"`, `remaining_days=null`
     (never converted to expired).
   - **`unknown`** → `shelf_life_status="unknown"`, `remaining_days=null`
     (never converted to fresh).
   - **`uncertain`** → `shelf_life_status="uncertain"`, `remaining_days=null`, and,
     equivalently, malformed/unusable confidence → `uncertain`.
2. If the fruit has no entry in `fruit_database.json`:
   - `shelf_life_status="unsupported"`, `remaining_days=null`,
     `basis="metadata_unavailable"`.
3. If `typical_shelf_life_days` is missing/invalid (never a fabricate default):
   - `shelf_life_status="unsupported"`, `remaining_days=null`,
     `basis="metadata_invalid"`.
4. If fruit is **`fresh`** with usable confidence:
   - `range = typical_shelf_life_days` from `fruit_database.json` (e.g. `[14, 30]` for apple).
   - `remaining_days = max(1, min(typical_max, round(typical_max * confidence)))`.
   - `shelf_life_status="estimated"`.
   - This is a **heuristic**, not a probability or an exact expiry.
5. Storage condition: `ambient` or `refrigerated`; validated strictly; passed
   through API → pipeline → estimator; recorded as **context only** (no
   condition-specific durations exist), appended to the explanation as
   "Estimate assumes {condition} storage."

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
      "shelf_life": {
          "fruit": "apple",
          "freshness_class": "fresh",
          "freshness_confidence": 0.973,
          "shelf_life_status": "estimated",
          "remaining_days": 29,
          "typical_min_days": 14,
          "typical_max_days": 30,
          "unit": "days",
          "basis": "fruit_typical_range + freshness_state + freshness_confidence",
          "storage_condition": "refrigerated",
          "explanation": "Fresh fruit with high confidence; estimate is close to maximum typical storage."
      },
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

Per-stage (CPU), real apple image (2 detections):
- **YOLO detection**: `281.23 ms`
- **Freshness (EfficientNet)**: `178.12 ms` (~89 ms/fruit)
- **Shelf-life heuristic**: `< 1.00 ms` (in-memory, O(1))
- **TOTAL `process_frame`**: `415.67 ms`

End-to-end mean across 6 real validation images: `440.52 ms`
(min `172.12 ms`, max `998.77 ms`). Shelf-life remains negligible.

## 7. Scientific Honesty

Shelf-life output is an **estimated remaining shelf life** derived from fruit
metadata + freshness state + freshness confidence + an assumed storage
condition. It is **not** an exact expiry and **not** a spoilage probability.
The system does not claim sensor-measured storage history. `remaining_days`
is `null` whenever the evidence is insufficient (unknown/unsupported/uncertain
or malformed confidence), and `0` only for expired (rotten/stale) states.

## 8. Testing & Validation

- Unit + integration suites: `tests/test_shelf_life.py`,
  `tests/test_freshness_shelf_life.py`,
  `tests/test_detection_pipeline_integration.py`, `tests/test_api.py`.
- Real-image validation: `scripts/validate_real_images_freshness.py` (6 species).
- Database validation: `scripts/validate_fruit_database.py` (all 10 fruits OK).
- Metadata validation is also exercised in `tests/test_shelf_life.py`
  (`TestMetadataFailureModes`) — missing database, malformed JSON, non-object
  JSON, missing/invalid typical ranges are all covered and never crash.

Run everything with:
```
python -m pytest tests/test_shelf_life.py tests/test_freshness_shelf_life.py \
    tests/test_detection_pipeline_integration.py tests/test_api.py -q
```
