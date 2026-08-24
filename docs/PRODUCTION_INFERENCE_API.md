# Production Inference API Documentation

## 1. Overview
The SmartFreshAI Inference API provides a fully integrated, HTTP-based production endpoint to analyze images of fruits. It orchestrates the frozen YOLO detection model, the EfficientNet freshness classifier, the metadata-driven shelf-life heuristic, and a **deterministic intelligence / decision layer** (rule engine — not a trained risk model) into a single coherent request.

## 2. API Endpoints

### `GET /health`
Verifies the application is running and ML components are successfully loaded.

**Response Schema (`HealthResponse`)**:
```json
{
  "status": "ok",
  "inference_ready": true,
  "detector_loaded": true,
  "classifier_loaded": true,
  "metadata_available": true,
  "device": "cpu"
}
```

### `POST /api/v1/inference/image`
Analyzes an uploaded image, detecting fruits, their freshness, and estimated shelf life.

**Request Form Data (`multipart/form-data`)**:
- `image` (file, required): The image file to analyze (JPEG, PNG).
- `storage_condition` (string, optional): Assumed storage condition (e.g., "ambient", "refrigerated").

**Response Schema (`InferenceResponse`)**:
```json
{
  "success": true,
  "message": "Fruits detected successfully.",
  "image": {
    "width": 640,
    "height": 480
  },
  "processing": {
    "latency_ms": 1150.2,
    "unidentified_count": 0,
    "fruit_count": 1,
    "intelligence_latency_ms": 0.4
  },
  "fruits": [
    {
      "tracking_id": 0,
      "fruit": "apple",
      "freshness": "fresh",
      "confidence": 0.88,
      "detection_confidence": 0.94,
      "stabilized_confidence": 0.88,
      "ema_confidence": 0.88,
      "is_uncertain": false,
      "is_locked": false,
      "lock_count": 0,
      "majority_label": "fresh",
      "vote_counts": {"fresh": 1},
      "shelf_life": {
        "fruit": "apple",
        "freshness_class": "fresh",
        "freshness_confidence": 0.88,
        "shelf_life_status": "estimated",
        "remaining_days": 26,
        "typical_min_days": 14,
        "typical_max_days": 30,
        "unit": "days",
        "basis": "fruit_typical_range + freshness_state + freshness_confidence",
        "storage_condition": "ambient",
        "explanation": "Fresh fruit with high confidence; estimate is close to maximum typical storage."
      },
      "bounding_box": {
        "x1": 50,
        "y1": 50,
        "x2": 250,
        "y2": 250
      },
      "center": [150, 150]
    }
  ],
  "intelligence": {
    "status": "ok",
    "overall_status": "all_clear",
    "risk_level": "low",
    "fruit_count": 1,
    "fresh_count": 1,
    "at_risk_count": 0,
    "expired_count": 0,
    "unsupported_count": 0,
    "inventory_summary": {},
    "priority_items": [],
    "recommendations": [],
    "waste_insight": "All detected fruits are currently low-risk.",
    "agent_state": {
      "detections": [],
      "freshness_results": [],
      "shelf_life_results": [],
      "risk_analysis": {},
      "recommendations": []
    },
    "errors": [],
    "message": null
  }
}
```

The `intelligence` object is produced by `FruitIntelligenceEngine` from the **actual** `fruits` list. See `docs/INTELLIGENCE_ENGINE.md`. Pre-intelligence clients can ignore the field. Values in the example above are illustrative of shape only; live responses use real model output.

## 3. Pipeline Lifecycle & Model Loading
Models (YOLO and EfficientNet) are loaded **once** at application startup via FastAPI lifespan context manager. They remain resident in memory to ensure low latency for incoming requests. The `DetectionPipeline` instances are reused safely. 

## 4. Hardware Acceleration (CPU/GPU)
The API dynamically inspects the host hardware. If CUDA or MPS is available, it binds the models to those accelerators. If no GPU is found, it automatically falls back to CPU execution without failing. This status is observable via the `device` key in the `/health` payload.

## 5. Supported Capabilities & Limitations
- **Supported Detection**: 10 distinct fruit classes (Apple, Banana, Orange, Grape, Kiwi, Mango, Strawberry, Cherry, Chickoo, Guava).
- **Supported Freshness**: Only Apple, Banana, and Orange support actual ML freshness classification. The rest natively and cleanly fall back to `unsupported` and propagate this status to the `shelf_life`.
- **Shelf-Life Caveat**: Shelf-life estimation is a heuristic based on botanical metadata and confidence scaling. It is **NOT** a certified or validated expiry prediction model, and must not be used for critical food safety decisions.

## 6. Error Handling
- Invalid image uploads return `400 Bad Request` with "corrupted image" or "invalid format".
- Missing payload returns `422 Unprocessable Entity`.
- ML execution crashes (e.g. out of memory) yield standard `500 Internal Server Error`.
- Unrecognized or unsupported classes do **not** trigger HTTP errors. They seamlessly populate the response payload with their appropriate fallback states (`unsupported`, `unknown`).

## 7. Performance & Latency
- **Cold Start (Initial Request / Model Warmup)**: Can take up to 2,000-3,500ms depending on the disk/memory bandwidth for caching.
- **Warm Inference (Subsequent Requests)**: Typically executes between 300ms (single fruit) and 1200ms (multi-fruit) on CPU hardware. GPU execution drops this to <100ms.
- **Intelligence layer**: Deterministic Python rules over the inference result. Measured separately as `processing.intelligence_latency_ms`; expected to be milliseconds or less relative to YOLO + EfficientNet. Measure with `python scripts/benchmark_intelligence.py --real`.
