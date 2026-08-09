"""Tests for the Phase 4 detection + classification pipeline
(src/inference/detection_pipeline.py).
"""

import numpy as np
import pytest

from src.detection import BoundingBox, Detection
from src.inference.detection_pipeline import (
    DetectionPipeline,
    DetectionPipelineConfig,
)


def make_frame(width=640, height=480):
    """A textured frame so crops pass size/blur gates."""
    frame = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
    frame[100:300, 100:300] = 128
    return frame


def fruit_det(bbox, label="apple", confidence=0.85):
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        timestamp=0.0,
    )


class TestDetectionPipelineConfig:
    def test_defaults(self):
        cfg = DetectionPipelineConfig()
        assert cfg.detector_name == "yolo"
        assert cfg.confidence_threshold == 0.45
        assert cfg.classify_every_n_frames == 3
        assert cfg.quality is not None  # auto-populated
        assert 0.0 <= cfg.detection_weight + cfg.classification_weight

    def test_custom_quality(self):
        from src.inference.quality import QualityConfig
        q = QualityConfig(brightness_min=50, brightness_max=200)
        cfg = DetectionPipelineConfig(quality=q)
        assert cfg.quality is q


class TestDetectionPipeline:
    def make_pipe(self, **cfg_kwargs):
        pipe = DetectionPipeline(
            DetectionPipelineConfig(detector_name="mock", **cfg_kwargs),
            predictor=None,
        )
        pipe.initialize()
        return pipe

    def test_no_detections_returns_empty(self):
        pipe = self.make_pipe()
        pipe.detector._detections = []
        result = pipe.process_frame(make_frame())
        assert len(result.fruits) == 0
        assert result.unidentified_count == 0
        assert result.frame_width == 640
        assert result.frame_height == 480
        pipe.shutdown()

    def test_detects_and_tracks_single_fruit(self):
        pipe = self.make_pipe()
        det = fruit_det(BoundingBox(100, 100, 300, 300))
        pipe.detector._detections = [det]

        first = pipe.process_frame(make_frame())
        assert len(first.fruits) == 1
        fr = first.fruits[0]
        assert fr.detection.tracking_id == 0
        assert fr.detection.label == "apple"

        # Same box on the next frame keeps the same tracking id.
        second = pipe.process_frame(make_frame())
        assert second.fruits[0].detection.tracking_id == 0
        pipe.shutdown()

    def test_multiple_fruits_get_distinct_ids(self):
        pipe = self.make_pipe()
        pipe.detector._detections = [
            fruit_det(BoundingBox(100, 100, 300, 300), label="apple"),
            fruit_det(BoundingBox(360, 150, 520, 400), label="banana"),
        ]
        result = pipe.process_frame(make_frame())
        ids = {f.detection.tracking_id for f in result.fruits}
        assert len(ids) == 2
        pipe.shutdown()

    def test_tiny_crop_is_rejected(self):
        # Box too small -> crop gate rejects it -> no fruit result.
        pipe = self.make_pipe()
        pipe.detector._detections = [
            fruit_det(BoundingBox(200, 200, 210, 210), confidence=0.9)
        ]
        result = pipe.process_frame(make_frame())
        assert len(result.fruits) == 0
        pipe.shutdown()

    def test_stabilizes_across_frames(self):
        pipe = self.make_pipe()
        det = fruit_det(BoundingBox(100, 100, 300, 300))
        pipe.detector._detections = [det]
        for _ in range(5):
            result = pipe.process_frame(make_frame())
        assert result.fruits[0].detection.tracking_id == 0
        assert pipe.get_stats()["total_fruits_tracked"] == 1
        pipe.shutdown()

    def test_str_and_stats(self):
        pipe = self.make_pipe()
        assert "MockDetector" in str(pipe)
        stats = pipe.get_stats()
        assert stats["detector_loaded"] is True
        assert stats["detector_backend"] == "MockDetector"
        pipe.shutdown()

    def test_shutdown_resets_state(self):
        pipe = self.make_pipe()
        pipe.detector._detections = [fruit_det(BoundingBox(100, 100, 300, 300))]
        pipe.process_frame(make_frame())
        pipe.shutdown()
        assert pipe.detector.is_loaded is False
        assert len(pipe.tracker.get_active_tracks()) == 0
