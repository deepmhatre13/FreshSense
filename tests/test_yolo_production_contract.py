"""Production YOLODetector and pipeline contract tests (Phase 10).

Tests:
    1. model loading
    2. model path configuration
    3. valid image inference
    4. multiple detections
    5. class mapping (10-class frozen taxonomy)
    6. confidence filtering
    7. bounding-box format (x1, y1, x2, y2)
    8. empty/no-detection result
    9. missing model handling
    10. pipeline integration
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from configs.config import Config
from src.detection import (
    BaseDetector,
    BoundingBox,
    Detection,
    DetectionResult,
    DetectorConfig,
    DetectorFactory,
    SUPPORTED_CLASSES,
    YOLODetector,
)
from src.inference.detection_pipeline import (
    DetectionPipeline,
    DetectionPipelineConfig,
)


class TestYOLOProductionContract:
    """Verify all 10 required production detection contract behaviors."""

    def test_1_model_loading(self):
        """Model should load once via load() and set is_loaded=True."""
        cfg = DetectorConfig(model_path="yolo11n.pt")
        detector = YOLODetector(cfg, weight_name="yolo11n.pt")
        assert not detector.is_loaded

        mock_yolo_cls = MagicMock()
        mock_model_inst = MagicMock()
        mock_yolo_cls.return_value = mock_model_inst

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
            detector.load()

        assert detector.is_loaded
        assert detector.model is mock_model_inst
        mock_model_inst.to.assert_called_once()

    def test_2_model_path_configuration(self):
        """Model path should be configurable and default to baseline best.pt."""
        default_cfg = DetectorConfig()
        assert default_cfg.model_path == "models/detection/detector/weights/best.pt"

        custom_cfg = DetectorConfig(model_path="custom/path/model.pt")
        detector = YOLODetector(custom_cfg)
        assert detector.weight_name == "custom/path/model.pt"

    def test_3_valid_image_inference(self):
        """Valid image should be processed producing structured DetectionResult."""
        cfg = DetectorConfig(model_path="yolo11n.pt")
        detector = YOLODetector(cfg, weight_name="yolo11n.pt")
        detector.is_loaded = True
        detector.model = MagicMock()

        # Mock YOLO prediction result box
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.conf.item.return_value = 0.92
        mock_box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 150.0])]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "Apple"}
        detector.model.predict.return_value = [mock_result]

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        res = detector.detect(frame)

        assert isinstance(res, DetectionResult)
        assert res.count == 1
        det = res.detections[0]
        assert det.class_id == 0
        assert det.class_name == "Apple"
        assert det.label == "Apple"
        assert det.confidence == 0.92
        assert det.bbox == BoundingBox(10, 20, 100, 150)

    def test_4_multiple_detections(self):
        """Detector should return multiple independent detections."""
        cfg = DetectorConfig(model_path="yolo11n.pt")
        detector = YOLODetector(cfg, weight_name="yolo11n.pt")
        detector.is_loaded = True
        detector.model = MagicMock()

        b1 = MagicMock()
        b1.cls.item.return_value = 0
        b1.conf.item.return_value = 0.90
        b1.xyxy = [MagicMock(tolist=lambda: [10.0, 10.0, 50.0, 50.0])]

        b2 = MagicMock()
        b2.cls.item.return_value = 4
        b2.conf.item.return_value = 0.85
        b2.xyxy = [MagicMock(tolist=lambda: [60.0, 60.0, 120.0, 120.0])]

        mock_result = MagicMock()
        mock_result.boxes = [b1, b2]
        mock_result.names = {0: "Apple", 4: "Orange"}
        detector.model.predict.return_value = [mock_result]

        frame = np.ones((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)

        assert res.count == 2
        assert res.detections[0].class_name == "Apple"
        assert res.detections[1].class_name == "Orange"

    def test_5_class_mapping(self):
        """Supported 10-class taxonomy should match canonical index order."""
        expected_classes = [
            "Apple",
            "Grape",
            "Kiwi",
            "Mango",
            "Orange",
            "Strawberry",
            "banana",
            "cherry",
            "chickoo",
            "guava",
        ]
        assert SUPPORTED_CLASSES == expected_classes

    def test_6_confidence_filtering(self):
        """Confidence threshold should be passed to model predict call."""
        cfg = DetectorConfig(model_path="yolo11n.pt", confidence_threshold=0.65, iou_threshold=0.40)
        detector = YOLODetector(cfg, weight_name="yolo11n.pt")
        detector.is_loaded = True
        detector.model = MagicMock()
        detector.model.predict.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame)

        detector.model.predict.assert_called_once_with(
            frame,
            conf=0.65,
            iou=0.40,
            imgsz=640,
            verbose=False,
        )

    def test_7_bounding_box_format(self):
        """Detection objects must expose x1, y1, x2, y2 and bounding_box in to_dict()."""
        det = Detection(
            label="Apple",
            confidence=0.91,
            bbox=BoundingBox(15, 25, 115, 125),
            class_id=0,
        )
        assert det.x1 == 15
        assert det.y1 == 25
        assert det.x2 == 115
        assert det.y2 == 125
        assert det.class_name == "Apple"

        d_dict = det.to_dict()
        assert d_dict["class_id"] == 0
        assert d_dict["class_name"] == "Apple"
        assert d_dict["confidence"] == 0.91
        assert d_dict["x1"] == 15
        assert d_dict["y1"] == 25
        assert d_dict["x2"] == 115
        assert d_dict["y2"] == 125

    def test_8_empty_no_detection_result(self):
        """Detector should return empty count=0 result without crashing."""
        cfg = DetectorConfig(model_path="yolo11n.pt")
        detector = YOLODetector(cfg, weight_name="yolo11n.pt")
        detector.is_loaded = True
        detector.model = MagicMock()
        detector.model.predict.return_value = [MagicMock(boxes=[])]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)

        assert isinstance(res, DetectionResult)
        assert res.count == 0
        assert res.detections == []

    def test_9_missing_model_handling(self):
        """Missing model checkpoint path should raise FileNotFoundError on load()."""
        cfg = DetectorConfig(model_path="non_existent_weights/best.pt")
        detector = YOLODetector(cfg)

        with pytest.raises(FileNotFoundError):
            detector.load()

    def test_10_pipeline_integration(self):
        """Pipeline should initialize and integrate with YOLO detector configuration."""
        pipe_cfg = DetectionPipelineConfig(
            detector_name="mock",
            detector_weights="models/detection/detector/weights/best.pt",
        )
        pipe = DetectionPipeline(pipe_cfg)
        pipe.initialize()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipe.process_frame(frame)

        assert res.frame_width == 640
        assert res.frame_height == 480
        pipe.shutdown()
