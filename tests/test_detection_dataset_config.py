"""Tests for DetectionDatasetConfig."""

import pytest
from configs.config import DetectionDatasetConfig, Config


class TestDetectionDatasetConfig:
    def test_defaults(self):
        config = DetectionDatasetConfig()
        assert config.roboflow_workspace == "deepam-mhatre"
        assert config.roboflow_project == "fruits-test-ajvf8-duncc"
        assert config.roboflow_version == 1
        assert config.detector_model == "yolo11n.pt"
        assert config.detector_epochs == 50
        assert config.detector_batch == 16
        assert config.detector_imgsz == 640

    def test_invalid_version(self):
        with pytest.raises(ValueError):
            DetectionDatasetConfig(roboflow_version=0)

    def test_invalid_epochs(self):
        with pytest.raises(ValueError):
            DetectionDatasetConfig(detector_epochs=0)

    def test_invalid_batch(self):
        with pytest.raises(ValueError):
            DetectionDatasetConfig(detector_batch=0)

    def test_invalid_imgsz(self):
        with pytest.raises(ValueError):
            DetectionDatasetConfig(detector_imgsz=0)


class TestConfigIntegration:
    def test_detection_dataset_in_config(self):
        config = Config()
        assert hasattr(config, "detection_dataset")
        assert isinstance(config.detection_dataset, DetectionDatasetConfig)

    def test_config_from_yaml_with_detection_dataset(self):
        config = Config.from_yaml("configs/settings.yaml")
        assert config.detection_dataset.roboflow_workspace == "deepam-mhatre"
        assert config.detection_dataset.roboflow_project == "fruits-test-ajvf8-duncc"
