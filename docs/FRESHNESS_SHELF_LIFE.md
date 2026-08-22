# SmartFreshAI — Freshness & Shelf-Life Production Architecture

> **Status:** PRODUCTION READY / AUDITED
> **Freshness Classifier Model:** `models/checkpoints/best_model.pth` (EfficientNet-B0)
> **Metadata Source of Truth:** `fruit_database.json`

---

## 1. Pipeline & Architectural Overview

The SmartFreshAI inference pipeline operates as a multi-stage, multi-model orchestrator combining real deep learning models, computer vision heuristics, and botanical domain metadata:

```
  Input Frame
       │
       ▼
 1. QualityAssessor ────────► [RULE-BASED: OpenCV Brightness, Contrast, Blur, Motion Statistics]
       │
       ▼
 2. YOLODetector    ────────► [REAL ML: YOLO11n Frozen Baseline `models/detection/detector/weights/best.pt`]
       │
       ▼
 3. DetectionTracker ───────► [ALGORITHM: IoU + Centroid Distance Multi-Object Tracking]
       │
       ▼
 4. Cropper         ────────► [RULE-BASED: 8% Bounding Box Expansion & 224x224 Resizing]
       │
       ▼
 5. Predictor       ────────► [REAL ML / FALLBACK: EfficientNet-B0 `models/checkpoints/best_model.pth`]
       │                       - Supported Fruits (Apple, Banana, Orange): 6-class ML prediction
       │                       - Unsupported Fruits (7 species): Returns `freshness_class="unsupported"`
       ▼
 6. Stabilizer      ────────► [ALGORITHM: EMA Confidence Smoothing, Sliding Window Voting & Lock]
       │
       ▼
 7. Confidence Fusion ──────► [HEURISTIC FORMULA: 0.4 * DetConf + 0.6 * ClassConf]
       │
       ▼
 8. ShelfLifeEstimator ─────► [LOOKUP + HEURISTIC: `fruit_database.json` + Confidence Scaling]
       │
       ▼
 9. MultiFruitResult ───────► [MACHINE-READABLE SCHEMA CONTRACT]
```

---

## 2. Component Implementation Status & Categorization

| Component | Implementation File | Method / Class | Implementation Type | Supported Models / Assets | Production Status |
|---|---|---|---|---|---|
| **Quality Assessment** | `src/inference/quality.py` | `QualityAssessor` | `RULE-BASED` | OpenCV Stats (Laplacian, Mean, Std) | `COMPLETE` |
| **Object Detection** | `src/detection/detector.py` | `YOLODetector` | `REAL ML` | `models/detection/detector/weights/best.pt` | `COMPLETE` |
| **Object Tracking** | `src/inference/tracker.py` | `DetectionTracker` | `ALGORITHM` | IoU / Centroid Tracking | `COMPLETE` |
| **Fruit Cropping** | `src/inference/cropper.py` | `Cropper` | `RULE-BASED` | Box Expansion & Gating | `COMPLETE` |
| **Freshness Classification**| `src/inference/predictor.py` | `Predictor` | `REAL ML / FALLBACK` | `models/checkpoints/best_model.pth` (EfficientNet-B0) | `PARTIAL` (3/10 ML, 7/10 Fallback) |
| **Prediction Stabilization** | `src/inference/stabilizer.py` | `PredictionStabilizer` | `ALGORITHM` | EMA ($\alpha=0.2$) + Window Voting ($N=15$) | `COMPLETE` |
| **Confidence Fusion** | `src/inference/fusion.py` | `ConfidenceFusion` | `HEURISTIC` | Weighted Sum ($0.4 / 0.6$) | `COMPLETE` |
| **Shelf-Life Estimation** | `src/inference/shelf_life.py` | `ShelfLifeEstimator` | `LOOKUP / HEURISTIC` | `fruit_database.json` Botanical Ranges | `COMPLETE` |

---

## 3. Fruit Support Matrix

The YOLO detector recognizes 10 fruit/vegetable classes. Freshness and shelf-life logic differ based on ML model training:

| Fruit Class | YOLO Detection Supported? | Freshness Supported? | Freshness Method | Shelf-Life Method | Fallback Behavior |
|---|---|---|---|---|---|
| **Apple** | YES | **YES** | `REAL ML` (EfficientNet-B0) | `LOOKUP + HEURISTIC` | N/A |
| **Grape** | YES | **NO** | `UNSUPPORTED` | `LOOKUP + HEURISTIC` | Returns `freshness="unsupported"`, typical shelf-life |
| **Kiwi** | YES | **NO** | `UNSUPPORTED` | `LOOKUP + HEURISTIC` | Returns `freshness="unsupported"`, typical shelf-life |
| **Mango** | YES | **NO** | `UNSUPPORTED` | `LOOKUP + HEURISTIC` | Returns `freshness="unsupported"`, typical shelf-life |
| **Orange** | YES | **YES** | `REAL ML` (EfficientNet-B0) | `LOOKUP + HEURISTIC` | N/A |
| **Strawberry** | YES | **NO** | `UNSUPPORTED` | `LOOKUP + HEURISTIC` | Returns `freshness="unsupported"`, typical shelf-life |
| **banana** | YES | **YES** | `REAL ML` (EfficientNet-B0) | `LOOKUP + HEURISTIC` | N/A |
| **cherry** | YES | **NO** | `UNSUPPORTED` | `LOOKUP + HEURISTIC` | Returns `freshness="unsupported"`, typical shelf-life |
| **chickoo** | YES | **NO** | `UNSUPPORTED` | `LOOKUP + HEURISTIC` | Returns `freshness="unsupported"`, typical shelf-life |
| **guava** | YES | **NO** | `UNSUPPORTED` | `LOOKUP + HEURISTIC` | Returns `freshness="unsupported"`, typical shelf-life |

---

## 4. Freshness Model Artifacts & Specifications

- **Checkpoint Path:** `models/checkpoints/best_model.pth`
- **Architecture:** `FreshSenseEfficientNet` (EfficientNet-B0)
- **Trained Classes (6-class taxonomy):**
  1. `freshapples`
  2. `freshbanana`
  3. `freshoranges`
  4. `rottenapples`
  5. `rottenbanana`
  6. `rottenoranges`
- **Input Tensor Specification:** $3 \times 224 \times 224$ RGB normalized tensor (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

---

## 5. Shelf-Life Estimation Formula

Shelf-life estimation uses botanical storage ranges from `fruit_database.json` combined with fused confidence scaling:

$$\text{Remaining Max Days} = \max\left(0, \lfloor \text{Typical Max Days} \times \text{Fused Confidence} \rceil\right)$$
$$\text{Remaining Min Days} = \max\left(0, \lfloor \text{Typical Min Days} \times \text{Fused Confidence} \rceil\right)$$

### Freshness State Mappings:
- **`fresh`**: Standard confidence scaling applied to typical remaining days.
- **`stale`**: Returns `0-1 days` ("Fruit is stale - consume within 24 hours").
- **`rotten`**: Returns `0-0 days` ("Fruit is spoiled - consume immediately or discard").
- **`unsupported` / `unknown`**: Scales botanical typical range by detection confidence and explicitly notes `freshness model unsupported`.

---

## 6. Output Schema Contract (`MultiFruitResult`)

```json
{
  "fruits": [
    {
      "tracking_id": 0,
      "fruit": "Apple",
      "freshness": "fresh",
      "confidence": 0.8404,
      "detection_confidence": 0.8837,
      "stabilized_confidence": 0.88,
      "is_uncertain": false,
      "shelf_life": "12-25 days",
      "bounding_box": {
        "x1": 44,
        "y1": 38,
        "x2": 341,
        "y2": 277
      }
    }
  ],
  "fruit_count": 1,
  "unidentified_count": 0,
  "frame_width": 450,
  "frame_height": 428
}
```

---

## 7. Latency Profile & Performance Benchmarks

Measured across test images on standard CPU hardware:

| Sample Image | Detected Fruit | Detections | Total Latency (ms) |
|---|---|---|---|
| Apple (`apple_12.jpg`) | Apple (x2) | 2 | 1442.76 ms |
| Grape (`Grape-45.jpg`) | Grape (x3) | 3 | 586.14 ms |
| Mango (`IMG_7533.jpg`) | Cherry (x1) | 1 | 1855.51 ms |
| Orange (`Curiosidades.jpg`) | Orange (x2) | 2 | 850.82 ms |
| Guava (`guava-1.jpg`) | Guava (x2) | 2 | 308.04 ms |

- **Mean Pipeline Latency:** `1008.65 ms`
- **Min Pipeline Latency:** `308.04 ms`
- **Max Pipeline Latency:** `1855.51 ms`

---

## 8. Limitations & Future Roadmap

1. **Freshness Class Coverage Gap:** The EfficientNet classifier model (`best_model.pth`) currently supports only 3 of the 10 detected fruit species (Apple, Banana, Orange). Training datasets for Grape, Kiwi, Mango, Strawberry, Cherry, Chickoo, and Guava are required to expand ML freshness grading.
2. **Shelf-Life Heuristic Basis:** Shelf-life is currently derived from published botanical storage guidelines (`fruit_database.json`) scaled by prediction confidence, rather than a direct multi-spectral ML regression model.
