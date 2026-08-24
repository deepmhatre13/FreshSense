"""Integration tests for DetectionPipeline with YOLO detector."""

import numpy as np
import pytest

from configs.config import Config
from src.detection import BoundingBox, Detection
from src.inference.detection_pipeline import (
    DetectionPipeline,
    DetectionPipelineConfig,
)


class TestDetectionPipelineIntegration:
    """Verify DetectionPipeline works with YOLO detector config."""

    def test_pipeline_with_mock_detector(self):
        """DetectionPipeline should work end-to-end with mock detector."""
        pipe = DetectionPipeline(
            DetectionPipelineConfig(detector_name="mock"),
            predictor=None,
        )
        pipe.initialize()

        frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        frame[100:300, 100:300] = 128

        det = Detection(
            label="apple",
            confidence=0.9,
            bbox=BoundingBox(100, 100, 300, 300),
            timestamp=0.0,
        )
        pipe.detector._detections = [det]

        result = pipe.process_frame(frame)
        assert len(result.fruits) == 1
        assert result.fruits[0].detection.label == "apple"
        assert result.fruits[0].detection.tracking_id == 0

        pipe.shutdown()

    def test_pipeline_config_defaults(self):
        """Pipeline config should have sensible defaults."""
        cfg = DetectionPipelineConfig()
        assert cfg.detector_name == "yolo"
        assert cfg.detector_weights == "models/detection/detector/weights/best.pt"
        assert cfg.confidence_threshold == 0.45
        assert cfg.classify_every_n_frames == 3

    def test_yolo_detector_config_from_settings(self):
        """Config should load YOLO settings from settings.yaml."""
        config = Config.from_yaml("configs/settings.yaml")
        assert config.detection_dataset.detector_model == "yolo11n.pt"
        assert config.detection_dataset.detector_epochs == 50
        assert config.detection_dataset.detector_batch == 16
        assert config.detection_dataset.detector_imgsz == 640

    def test_pipeline_config_matches_settings_yaml(self):
        """DetectionPipelineConfig should be buildable from settings.yaml."""
        config = Config.from_yaml("configs/settings.yaml")
        ds = config.detection_dataset
        d = config.detection

        pipe_cfg = DetectionPipelineConfig(
            detector_name="yolo",
            detector_weights=d.detector_weights,
            confidence_threshold=d.detection_confidence,
            iou_threshold=d.detection_iou,
            max_detections=d.max_detections,
            crop_expand_scale=d.crop_expand_scale,
            crop_min_side=d.crop_min_side,
            crop_min_area=d.crop_min_area,
            crop_target_size=d.crop_target_size,
            tracker_iou_threshold=d.tracker_iou_threshold,
            tracker_max_distance=d.tracker_max_distance,
            tracker_max_lost_frames=d.tracker_max_lost_frames,
            detection_weight=d.detection_weight,
            classification_weight=d.classification_weight,
            classify_every_n_frames=d.classify_every_n_frames,
        )

        assert pipe_cfg.detector_weights == "models/detection/detector/weights/best.pt"
        assert pipe_cfg.confidence_threshold == 0.45

    def test_yolo_detector_loads_weight_name(self):
        """YOLODetector should accept weight name from config."""
        from src.detection.detector import YOLODetector
        from src.detection.base_detector import DetectorConfig

        cfg = DetectorConfig(model_path="yolo11n.pt")
        detector = YOLODetector(cfg, weight_name="yolo11n.pt")
        assert detector.weight_name == "yolo11n.pt"

    def test_pipeline_unsupported_fruit_does_not_fabricate_freshness(self):
        """An unsupported fruit should explicitly report unknown freshness."""
        pipe = DetectionPipeline(
            DetectionPipelineConfig(detector_name="mock"),
            predictor=None,
        )
        pipe.initialize()
        
        frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        frame[100:300, 100:300] = 128
        
                # Kiwi has NO validated freshness model (data_not_available per the
        # availability registry) — freshness must never be fabricated.
        det = Detection(
            label="kiwi",
            confidence=0.9,
            bbox=BoundingBox(100, 100, 300, 300),
            timestamp=0.0,
        )
        pipe.detector._detections = [det]
        
        result = pipe.process_frame(frame)
        assert len(result.fruits) == 1
        fruit_res = result.fruits[0]
        
        assert fruit_res.detection.label == "kiwi"
        assert fruit_res.freshness_class == "data_not_available"

    def test_pipeline_supported_fruit_missing_metadata(self):
        """A supported fruit with no metadata should flag shelf-life as unavailable or heuristic."""
        # Note: metadata is loaded from fruit_database.json which we created,
        # but let's clear the metadata db to simulate missing
        pipe = DetectionPipeline(
            DetectionPipelineConfig(detector_name="mock"),
            predictor=None,
        )
        pipe.initialize()
        pipe.shelf_life.metadata_db._metadata.clear()  # Force empty metadata
        
        frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        frame[100:300, 100:300] = 128
        
        # Apple is supported by freshness classifier
        det = Detection(
            label="apple",
            confidence=0.9,
            bbox=BoundingBox(100, 100, 300, 300),
            timestamp=0.0,
        )
        pipe.detector._detections = [det]
        
        result = pipe.process_frame(frame)
        assert len(result.fruits) == 1
        fruit_res = result.fruits[0]
        
        assert fruit_res.detection.label == "apple"
        assert fruit_res.shelf_life.shelf_life_status == "unsupported"



# ============================================================================
# Storage-condition pass-through, multi-fruit isolation, tracking (Phases 5/13/14)
# ============================================================================

from types import SimpleNamespace

from src.inference.shelf_life import ShelfLifeEstimate


class _StubPredictor:
    """Deterministic freshness classifier stand-in (no torch required)."""

    def __init__(self, freshness="fresh", confidence=0.85):
        self.freshness = freshness
        self.confidence = confidence
        self.calls = 0

    def predict(self, crop):
        self.calls += 1
        return SimpleNamespace(
            freshness_class=self.freshness,
            confidence=self.confidence,
            latency_ms=0.5,
            model_version="stub",
        )


def _make_pipeline():
    pipe = DetectionPipeline(DetectionPipelineConfig(detector_name="mock"), predictor=None)
    pipe.initialize()
    return pipe


def test_pipeline_passes_storage_condition_to_estimator():
    pipe = _make_pipeline()
    pipe.predictor = _StubPredictor("fresh", 0.85)
    frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)

    det = Detection(label="apple", confidence=0.9,
                    bbox=BoundingBox(100, 100, 300, 300), timestamp=0.0)
    pipe.detector._detections = [det]

    result = pipe.process_frame(frame, storage_condition="refrigerated")
    assert len(result.fruits) == 1
    shelf = result.fruits[0].shelf_life
    assert isinstance(shelf, ShelfLifeEstimate)
    assert shelf.storage_condition == "refrigerated"
    assert "assumes refrigerated storage" in shelf.explanation

    result2 = pipe.process_frame(frame, storage_condition=None)
    assert result2.fruits[0].shelf_life.storage_condition == "ambient"
    pipe.shutdown()


def test_pipeline_rejects_invalid_storage_condition_before_inference():
    pipe = _make_pipeline()
    stub = _StubPredictor()
    pipe.predictor = stub
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pipe.detector._detections = [
        Detection(label="apple", confidence=0.9,
                  bbox=BoundingBox(100, 100, 300, 300), timestamp=0.0)
    ]
    with pytest.raises(ValueError):
        pipe.process_frame(frame, storage_condition="freezer")
    assert stub.calls == 0, "no classification work may happen for invalid input"
    pipe.shutdown()


def test_pipeline_multi_fruit_shelf_life_isolated_per_fruit():
    pipe = _make_pipeline()
    pipe.predictor = _StubPredictor("fresh", 0.85)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    pipe.detector._detections = [
        Detection(label="apple", confidence=0.9,
                  bbox=BoundingBox(20, 20, 200, 200), timestamp=0.0),
        Detection(label="kiwi", confidence=0.88,
                  bbox=BoundingBox(300, 300, 480, 480), timestamp=0.0),
    ]
    result = pipe.process_frame(frame)
    by_label = {f.detection.label: f for f in result.fruits}
    assert set(by_label) == {"apple", "kiwi"}

    apple, kiwi = by_label["apple"], by_label["kiwi"]
    assert apple.freshness_class == "fresh"
    assert apple.shelf_life.shelf_life_status == "estimated"
    assert apple.shelf_life.remaining_days >= 1

    # Kiwi has NO freshness model: data_not_available, never fabricated.
    assert kiwi.freshness_class == "data_not_available"
    assert kiwi.shelf_life.shelf_life_status == "data_not_available"
    assert kiwi.shelf_life.remaining_days is None
    pipe.shutdown()


def test_pipeline_repeat_tracking_id_stabilizes_not_duplicates():
    """Same physical fruit across frames: ONE record per track id, the
    classification is REUSED between classify ticks, and shelf-life stays
    deterministic for identical fused confidence."""
    pipe = _make_pipeline()
    stub = _StubPredictor("fresh", 0.85)
    pipe.predictor = stub
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    det = Detection(label="banana", confidence=0.9,
                    bbox=BoundingBox(50, 50, 220, 220), timestamp=0.0)

    pipe.detector._detections = [det]
    first = pipe.process_frame(frame)
    assert len(first.fruits) == 1
    assert stub.calls == 1  # classified on first sight

    pipe.detector._detections = [det]  # same object -> same track id
    second = pipe.process_frame(frame)
    assert len(second.fruits) == 1
    assert second.fruits[0].detection.tracking_id == \
        first.fruits[0].detection.tracking_id
    assert stub.calls == 1, "classification must be reused within the cadence window"

    d1 = first.fruits[0].shelf_life.to_dict()
    d2 = second.fruits[0].shelf_life.to_dict()
    assert d1["remaining_days"] == d2["remaining_days"]
    assert d1["storage_condition"] == d2["storage_condition"]
    pipe.shutdown()

