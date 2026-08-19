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
        assert cfg.detector_weights == "yolo11n.pt"
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
            detector_weights=ds.detector_model,
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

        assert pipe_cfg.detector_weights == "yolo11n.pt"
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
        
        # Grape is not supported by freshness classifier
        det = Detection(
            label="grape",
            confidence=0.9,
            bbox=BoundingBox(100, 100, 300, 300),
            timestamp=0.0,
        )
        pipe.detector._detections = [det]
        
        result = pipe.process_frame(frame)
        assert len(result.fruits) == 1
        fruit_res = result.fruits[0]
        
        assert fruit_res.detection.label == "grape"
        assert fruit_res.freshness_class == "unknown"

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
        assert fruit_res.shelf_life.basis_type in ("unavailable", "heuristic")

