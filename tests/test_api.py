"""API tests: storage-condition pass-through, shelf-life semantics, serialization.

The intelligence layer was intentionally removed from the production path
(direction change); these tests verify the shelf-life subsystem contract:

* ``storage_condition`` is validated strictly (400 on arbitrary text) and is
  actually passed through API -> DetectionPipeline -> ShelfLifeEstimator.
* ``remaining_days`` is an integer ONLY for estimated/expired states.
* ``remaining_days`` is null for unknown/uncertain/unsupported states.
* ``remaining_days = 0`` occurs ONLY for expired/rotten/stale.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import app
from src.api.schemas import ShelfLifeSchema
from src.detection import BoundingBox, Detection
from src.inference.fruit_result import FruitResult, MultiFruitResult
from src.inference.shelf_life import ShelfLifeConfig, ShelfLifeEstimator


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _jpg_bytes() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((80, 80, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


class _FakeStabilized:
    """Minimal stand-in for StabilizedPrediction (attribute duck-typing)."""

    def __init__(self):
        self.confidence = 0.9
        self.ema_confidence = 0.9
        self.is_locked = False
        self.lock_count = 1
        self.majority_label = "fresh"
        self.vote_counts = {"fresh": 1}
        self.is_uncertain = False


def _make_fruit(label: str, freshness_class: str, confidence: float = 0.9,
                tracking_id: int = 0) -> FruitResult:
    """Build a FruitResult whose shelf_life comes from the REAL estimator."""
    estimator = ShelfLifeEstimator(ShelfLifeConfig())
    estimate = estimator.estimate(
        fruit=label,
        fused_confidence=confidence,
        freshness_class=freshness_class,
    )
    det = Detection(
        label=label,
        confidence=confidence,
        bbox=BoundingBox(10, 10, 50, 50),
        tracking_id=tracking_id,
    )
    return FruitResult(
        detection=det,
        stabilized=_FakeStabilized(),
        fused_confidence=confidence,
        freshness_class=freshness_class,
        shelf_life=estimate,
    )


class _FakePipeline:
    """Returns canned FruitResults but recomputes shelf-life with the
    storage condition received — mirroring the real pipeline's pass-through
    (API -> pipeline -> ShelfLifeEstimator)."""

    def __init__(self, fruits):
        self._fruits = fruits
        self.detector = object()  # /health touches .detector
        self.seen_conditions = []
        self._estimator = ShelfLifeEstimator(ShelfLifeConfig())

    def process_frame(self, frame, storage_condition=None):
        self.seen_conditions.append(storage_condition)
        reconstructed = []
        est = self._estimator
        for fruit in self._fruits:
            estimate = est.estimate(
                fruit=fruit.detection.label,
                fused_confidence=fruit.fused_confidence,
                freshness_class=fruit.freshness_class,
                storage_condition=storage_condition,
            )
            reconstructed.append(FruitResult(
                detection=fruit.detection,
                stabilized=fruit.stabilized,
                fused_confidence=fruit.fused_confidence,
                freshness_class=fruit.freshness_class,
                shelf_life=estimate,
            ))
        return MultiFruitResult(
            fruits=reconstructed,
            frame_width=80,
            frame_height=80,
            unidentified_count=0,
        )


@pytest.fixture
def sample_image():
    image_path = Path(
        "data/detection/test/images/apple_12_jpg.rf.7f4a14511957f2baef662e3855fd513b.jpg"
    )
    if not image_path.exists():
        return _jpg_bytes()
    return image_path.read_bytes()


# ----------------------------------------------------------------------------
# Request-level behavior
# ----------------------------------------------------------------------------

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["inference_ready"] in (True, False)
        assert "metadata_available" in data


def test_inference_no_image():
    with TestClient(app) as client:
        response = client.post("/api/v1/inference/image")
        assert response.status_code == 422


def test_inference_invalid_image():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference/image",
            files={"image": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "corrupted image" in response.json()["detail"].lower()


def test_inference_invalid_storage_condition_rejected(monkeypatch):
    """Arbitrary storage text must be a 400 BEFORE any inference work."""
    fake = _FakePipeline([_make_fruit("apple", "fresh")])
    monkeypatch.setattr(app_module, "pipeline", fake)
    client = TestClient(app_module.app)

    response = client.post(
        "/api/v1/inference/image",
        files={"image": ("img.jpg", _jpg_bytes(), "image/jpeg")},
        data={"storage_condition": "freezer"},
    )
    assert response.status_code == 400, response.text
    assert "storage_condition" in response.json()["detail"]
    assert fake.seen_conditions == [], "pipeline must not run for invalid input"


def test_schema_accepts_null_freshness_confidence():
    payload = {
        "fruit": "mango",
        "freshness_class": "data_not_available",
        "freshness_confidence": None,
        "shelf_life_status": "data_not_available",
        "remaining_days": None,
        "typical_min_days": None,
        "typical_max_days": None,
        "unit": "days",
        "basis": "freshness_model_unsupported_or_uncertain",
        "storage_condition": "ambient",
        "explanation": "No trained freshness model available.",
    }
    schema = ShelfLifeSchema.model_validate(payload)
    assert schema.freshness_confidence is None
    assert schema.remaining_days is None


# ----------------------------------------------------------------------------
# Storage-condition pass-through and shelf-life semantics (mocked pipeline)
# ----------------------------------------------------------------------------

def _post(monkeypatch, fake, **form):
    monkeypatch.setattr(app_module, "pipeline", fake)
    client = TestClient(app_module.app)
    return client.post(
        "/api/v1/inference/image",
        files={"image": ("img.jpg", _jpg_bytes(), "image/jpeg")},
        data=form,
    )


def test_storage_condition_missing_defaults_to_ambient(monkeypatch):
    fake = _FakePipeline([_make_fruit("apple", "fresh", 0.88)])
    response = _post(monkeypatch, fake)
    assert response.status_code == 200, response.text
    data = response.json()
    assert fake.seen_conditions == [None]
    shelf = data["fruits"][0]["shelf_life"]
    assert shelf["storage_condition"] == "ambient"
    assert isinstance(shelf["remaining_days"], int)
    assert 14 <= shelf["remaining_days"] <= 30


def test_storage_condition_ambient_and_refrigerated_recorded(monkeypatch):
    """Both conditions are echoed; the number is identical because the
    database has NO condition-specific durations (context-only semantics)."""
    days = {}
    for condition in ("ambient", "refrigerated"):
        fake = _FakePipeline([_make_fruit("apple", "fresh", 0.88)])
        response = _post(monkeypatch, fake, storage_condition=condition)
        assert response.status_code == 200, response.text
        shelf = response.json()["fruits"][0]["shelf_life"]
        assert shelf["storage_condition"] == condition
        days[condition] = shelf["remaining_days"]
    assert days["ambient"] == days["refrigerated"]
    assert isinstance(days["ambient"], int)


def test_fresh_fruit_contract(monkeypatch):
    fake = _FakePipeline([_make_fruit("banana", "fresh", 0.75)])
    data = _post(monkeypatch, fake).json()
    fruit = data["fruits"][0]
    shelf = fruit["shelf_life"]
    assert fruit["freshness"] == "fresh"
    assert shelf["shelf_life_status"] == "estimated"
    assert isinstance(shelf["remaining_days"], int)
    assert 1 <= shelf["remaining_days"] <= 7
    assert shelf["typical_min_days"] == 3
    assert shelf["typical_max_days"] == 7
    assert shelf["unit"] == "days"
    assert shelf["basis"] == "fruit_typical_range + freshness_state + freshness_confidence"
    assert "assumes ambient storage" in shelf["explanation"]


def test_rotten_fruit_zero_days_only_for_expired(monkeypatch):
    fake = _FakePipeline([_make_fruit("orange", "rotten")])
    data = _post(monkeypatch, fake).json()
    shelf = data["fruits"][0]["shelf_life"]
    assert shelf["shelf_life_status"] == "expired"
    assert shelf["remaining_days"] == 0
    assert shelf["freshness_confidence"] > 0



def test_unsupported_freshness_null_days(monkeypatch):
    """Mango has no freshness model -> data_not_available, NEVER fabricated."""
    fake = _FakePipeline([_make_fruit("mango", "data_not_available", 0.0)])
    data = _post(monkeypatch, fake).json()
    fruit = data["fruits"][0]
    shelf = fruit["shelf_life"]
    assert fruit["freshness"] == "data_not_available"
    # API contract (spec): fruit-level confidence must be null when no
    # freshness model exists — never serialize a fake confidence.
    assert fruit["confidence"] is None
    assert shelf["shelf_life_status"] == "data_not_available"
    # No fabricated remaining days for a fruit without a freshness model.
    assert shelf["remaining_days"] is None


def test_multi_fruit_independent_shelf_life(monkeypatch):
    fruits = [
        _make_fruit("apple", "fresh", 0.9, tracking_id=0),
        _make_fruit("orange", "rotten", 0.8, tracking_id=1),
        _make_fruit("mango", "data_not_available", 0.0, tracking_id=2),
    ]
    data = _post(monkeypatch, _FakePipeline(fruits)).json()
    assert data["processing"]["fruit_count"] == 3
    by_id = {f["tracking_id"]: f for f in data["fruits"]}
    assert len(by_id) == 3
    statuses = {fid: f["shelf_life"]["shelf_life_status"] for fid, f in by_id.items()}
    assert statuses == {0: "estimated", 1: "expired", 2: "data_not_available"}
    remaining = {fid: f["shelf_life"]["remaining_days"] for fid, f in by_id.items()}
    assert isinstance(remaining[0], int) and remaining[0] > 0
    assert remaining[1] == 0
    assert remaining[2] is None


def test_response_serialization_roundtrip(monkeypatch):
    fruits = [
        _make_fruit("apple", "fresh", 0.9, tracking_id=0),
        _make_fruit("kiwi", "unknown", 0.5, tracking_id=1),
    ]
    response = _post(monkeypatch, _FakePipeline(fruits))
    assert response.status_code == 200
    payload = response.json()  # FastAPI validation already enforced types
    json.dumps(payload)  # must be plain-JSON serializable
    for fruit in payload["fruits"]:
        assert {"fruit", "freshness", "shelf_life"}.issubset(fruit.keys())
        status = fruit["shelf_life"]["shelf_life_status"]
        if status not in ("estimated", "expired"):
            assert fruit["shelf_life"]["remaining_days"] is None


class _FakeDetector:
    """Minimal detector returning canned raw detections for the preview endpoint."""

    def __init__(self, detections):
        self._detections = detections  # list of Detection objects
        self.seen_frames = 0

    def detect(self, frame):
        import time as _t

        self.seen_frames += 1
        from src.detection import DetectionResult

        return DetectionResult(
            detections=list(self._detections),
            frame_width=80,
            frame_height=80,
            latency_ms=2.5,
        )


class _FakePipelineWithDetector(_FakePipeline):
    """Pipelines that can also serve the live-detection preview endpoint."""

    def __init__(self, fruits, detections=None):
        super().__init__(fruits)
        self.detector = _FakeDetector(detections or [])


# ----------------------------------------------------------------------------
# Live-detection preview endpoint
# ----------------------------------------------------------------------------

def _post_preview(monkeypatch, fake, body: bytes = None):
    monkeypatch.setattr(app_module, "pipeline", fake)
    client = TestClient(app_module.app)
    return client.post(
        "/api/v1/detection/preview",
        files={"image": ("frame.jpg", body if body is not None else _jpg_bytes(), "image/jpeg")},
    )


def test_detection_preview_returns_detections(monkeypatch):
    dets = [
        Detection(label="Apple", confidence=0.93, bbox=BoundingBox(10, 20, 110, 140), class_id=0),
        Detection(label="banana", confidence=0.41, bbox=BoundingBox(200, 50, 300, 200), class_id=6),
    ]
    fake = _FakePipelineWithDetector([], detections=dets)
    response = _post_preview(monkeypatch, fake)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert len(data["detections"]) == 2
    first = data["detections"][0]
    assert first["label"] == "Apple"
    assert first["confidence"] == 0.93
    assert first["bbox"] == {"x1": 10, "y1": 20, "x2": 110, "y2": 140}
    assert "shelf_life" not in first, "preview must not run the full pipeline"


def test_detection_preview_invalid_image_returns_empty(monkeypatch):
    dets = []
    fake = _FakePipelineWithDetector([], detections=dets)
    response = _post_preview(monkeypatch, fake, body=b"not an image")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is False
    assert data["detections"] == []
    assert fake.detector.seen_frames == 0, "detector must not run on undecodable input"


def test_detection_preview_no_detection(monkeypatch):
    fake = _FakePipelineWithDetector([], detections=[])
    response = _post_preview(monkeypatch, fake)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["detections"] == []
    assert fake.detector.seen_frames == 1


# ----------------------------------------------------------------------------
# REAL end-to-end (loads actual YOLO + EfficientNet via app lifespan)
# ----------------------------------------------------------------------------

def test_inference_real_image_end_to_end(sample_image):
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference/image",
            files={"image": ("image.jpg", sample_image, "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["fruits"], list)
        assert "intelligence" not in data
        assert "intelligence_latency_ms" not in data["processing"]

        for fruit in data["fruits"]:
            shelf = fruit.get("shelf_life")
            assert shelf is not None
            status = shelf["shelf_life_status"]
            days = shelf["remaining_days"]
            if status == "estimated":
                assert isinstance(days, int) and days >= 1
                hi = shelf["typical_max_days"]
                assert hi is not None and days <= hi
            elif status == "expired":
                assert days == 0
            else:
                assert days is None
            assert shelf["storage_condition"] in ("ambient", "refrigerated")

        print("REAL IMAGE latency_ms:", data["processing"]["latency_ms"])

