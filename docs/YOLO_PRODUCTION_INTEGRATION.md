# SmartFreshAI — Production YOLO Detector Integration

> **Production Status:** ACTIVE / INTEGRATED
> **Production Model Weight:** `models/detection/detector/weights/best.pt` (Frozen Baseline)
> **Architecture:** YOLO11n (Pretrained baseline)

## 1. Overview & Decision Record

The YOLO detection experimentation phase is complete. All experimental iterations (including V3) failed the adoption gate requirement of +0.0100 mAP50 improvement. Consequently, the original frozen V2 baseline checkpoint was retained as the sole production detector.

### Baseline Performance Metrics:
- **Precision:** 0.7878
- **Recall:** 0.6605
- **mAP50:** 0.7155
- **mAP50-95:** 0.5456

---

## 2. Model & Configuration Specification

- **Weights Path:** `models/detection/detector/weights/best.pt`
- **Default Resolution:** `640 x 640` (`imgsz=640`)
- **Default Confidence Threshold:** `0.45`
- **Default IoU Threshold:** `0.45`
- **Device Support:** Auto-detection (`cuda` if GPU available, fallback to `cpu`)

---

## 3. Frozen 10-Class Taxonomy Mapping

The detector maps predictions to the following 10 canonical fruit classes by numeric `class_id`:

| `class_id` | Class Name | Freshness Support |
|------------|------------|-------------------|
| 0 | Apple | Supported |
| 1 | Grape | Unknown (Fallback) |
| 2 | Kiwi | Unknown (Fallback) |
| 3 | Mango | Unknown (Fallback) |
| 4 | Orange | Supported |
| 5 | Strawberry | Unknown (Fallback) |
| 6 | banana | Supported |
| 7 | cherry | Unknown (Fallback) |
| 8 | chickoo | Unknown (Fallback) |
| 9 | guava | Unknown (Fallback) |

*Note: Unsupported fruits (Grape, Kiwi, Mango, Strawberry, cherry, chickoo, guava) explicitly return freshness state `"unknown"` without misclassifying downstream.*

---

## 4. Production Detection Schema Contract

The `YOLODetector` output contract yields structured `Detection` objects with the following schema:

```python
{
    "class_id": 0,
    "class_name": "Apple",
    "class": "Apple",
    "confidence": 0.8837,
    "x1": 44,
    "y1": 38,
    "x2": 341,
    "y2": 277,
    "bounding_box": {
        "x1": 44,
        "y1": 38,
        "x2": 341,
        "y2": 277
    },
    "area": 71003,
    "center": [192, 157],
    "tracking_id": 0,
    "timestamp": 1724155200.0
}
```

---

## 5. End-to-End Pipeline Architecture Integration

```
  Input Image (Frame)
          │
          ▼
   QualityAssessor ──► [Quality Gating Warnings]
          │
          ▼
     YOLODetector ──► [Loads models/detection/detector/weights/best.pt]
          │
          ▼
  DetectionTracker ──► [Multi-object tracking IDs]
          │
          ▼
       Cropper     ──► [224x224 bounding box crops]
          │
          ▼
  Predictor / Fusion ─► [Freshness state & fused confidence]
          │
          ▼
ShelfLifeEstimator ──► [Remaining shelf-life prediction in days]
          │
          ▼
 MultiFruitResult  ──► [Structured multi-object output]
```

---

## 6. How to Run Detection Locally

### Running the Smoke Test:
```bash
python scripts/smoke_test_detector.py --image data/detection/test/images/apple_12_jpg.rf.7f4a14511957f2baef662e3855fd513b.jpg
```

### Running End-to-End Pipeline Validation:
```bash
python scripts/verify_pipeline_end_to_end.py
```

### Running Real-time Webcam Stream:
```bash
python scripts/run_webcam.py --model models/detection/detector/weights/best.pt
```

### Running Test Suite:
```bash
pytest tests/test_yolo_production_contract.py
pytest tests/test_detection_pipeline.py
```
