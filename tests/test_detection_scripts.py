"""Tests for YOLO detection training and evaluation scripts."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip tests that require ultralytics if it's not installed
try:
    import ultralytics  # noqa: F401
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

pytestmark = pytest.mark.skipif(
    not HAS_ULTRALYTICS,
    reason="ultralytics not installed",
)


class TestTrainDetectorScript:
    """Tests for scripts/train_detector.py"""

    def test_imports(self):
        """Script should be importable."""
        import scripts.train_detector as train_mod

        assert hasattr(train_mod, "train_detector")
        assert hasattr(train_mod, "main")

    def test_train_detector_missing_data_yaml(self, tmp_path):
        """Should exit if data.yaml is missing."""
        import scripts.train_detector as train_mod

        with pytest.raises(SystemExit):
            train_mod.train_detector(
                data_yaml=tmp_path / "nonexistent.yaml",
                model_name="yolo11n.pt",
                epochs=1,
                batch=1,
                imgsz=640,
                output_dir=tmp_path,
            )

    @patch("ultralytics.YOLO")
    def test_train_detector_success(self, mock_yolo_cls, tmp_path):
        """Should train and return path to best weights."""
        import scripts.train_detector as train_mod

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 2\nnames: [apple, banana]\n")

        run_dir = tmp_path / "detector"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_text("fake weights")

        mock_model = MagicMock()
        mock_model.train.return_value = MagicMock()
        mock_yolo_cls.return_value = mock_model

        result = train_mod.train_detector(
            data_yaml=data_yaml,
            model_name="yolo11n.pt",
            epochs=1,
            batch=1,
            imgsz=640,
            output_dir=tmp_path,
            device="cpu",
        )

        assert result == weights_dir / "best.pt"
        mock_model.train.assert_called_once()


class TestEvaluateDetectorScript:
    """Tests for scripts/evaluate_detector.py"""

    def test_imports(self):
        """Script should be importable."""
        import scripts.evaluate_detector as eval_mod

        assert hasattr(eval_mod, "evaluate_detector")
        assert hasattr(eval_mod, "main")

    def test_evaluate_detector_missing_model(self, tmp_path):
        """Should exit if model is missing."""
        import scripts.evaluate_detector as eval_mod

        with pytest.raises(SystemExit):
            eval_mod.evaluate_detector(
                model_path=tmp_path / "nonexistent.pt",
                data_yaml=tmp_path / "data.yaml",
            )

    @patch("ultralytics.YOLO")
    def test_evaluate_detector_success(self, mock_yolo_cls, tmp_path):
        """Should evaluate and return metrics dict."""
        import scripts.evaluate_detector as eval_mod

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 2\nnames: [apple, banana]\n")

        model_path = tmp_path / "best.pt"
        model_path.write_text("fake model")

        mock_model = MagicMock()
        mock_model.names = {0: "apple", 1: "banana"}
        mock_metrics = MagicMock()
        mock_metrics.box.mp = 0.85
        mock_metrics.box.mr = 0.80
        mock_metrics.box.map50 = 0.87
        mock_metrics.box.map = 0.72
        mock_metrics.box.maps = np.array([0.9, 0.84])
        mock_metrics.box.p = np.array([0.9, 0.8])
        mock_metrics.box.r = np.array([0.85, 0.75])
        mock_model.val.return_value = mock_metrics
        mock_yolo_cls.return_value = mock_model

        results = eval_mod.evaluate_detector(
            model_path=model_path,
            data_yaml=data_yaml,
        )

        assert results["status"] == "available"
        assert results["precision"] == 0.85
        assert results["recall"] == 0.80
        assert results["map50"] == 0.87
        assert "apple" in results["per_class"]
        assert "banana" in results["per_class"]

"""Tests for YOLO detection training and evaluation scripts."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestTrainDetectorScript:
    """Tests for scripts/train_detector.py"""

    def test_imports(self):
        """Script should be importable."""
        import scripts.train_detector as train_mod

        assert hasattr(train_mod, "train_detector")
        assert hasattr(train_mod, "main")

    def test_train_detector_missing_data_yaml(self, tmp_path):
        """Should exit if data.yaml is missing."""
        import scripts.train_detector as train_mod

        with pytest.raises(SystemExit):
            train_mod.train_detector(
                data_yaml=tmp_path / "nonexistent.yaml",
                model_name="yolo11n.pt",
                epochs=1,
                batch=1,
                imgsz=640,
                output_dir=tmp_path,
            )

    @patch("ultralytics.YOLO")
    def test_train_detector_success(self, mock_yolo_cls, tmp_path):
        """Should train and return path to best weights."""
        import scripts.train_detector as train_mod

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 2\nnames: [apple, banana]\n")

        run_dir = tmp_path / "detector"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_text("fake weights")

        mock_model = MagicMock()
        mock_model.train.return_value = MagicMock()
        mock_yolo_cls.return_value = mock_model

        result = train_mod.train_detector(
            data_yaml=data_yaml,
            model_name="yolo11n.pt",
            epochs=1,
            batch=1,
            imgsz=640,
            output_dir=tmp_path,
            device="cpu",
        )

        assert result == weights_dir / "best.pt"
        mock_model.train.assert_called_once()


class TestEvaluateDetectorScript:
    """Tests for scripts/evaluate_detector.py"""

    def test_imports(self):
        """Script should be importable."""
        import scripts.evaluate_detector as eval_mod

        assert hasattr(eval_mod, "evaluate_detector")
        assert hasattr(eval_mod, "main")

    def test_evaluate_detector_missing_model(self, tmp_path):
        """Should exit if model is missing."""
        import scripts.evaluate_detector as eval_mod

        with pytest.raises(SystemExit):
            eval_mod.evaluate_detector(
                model_path=tmp_path / "nonexistent.pt",
                data_yaml=tmp_path / "data.yaml",
            )

    @patch("ultralytics.YOLO")
    def test_evaluate_detector_success(self, mock_yolo_cls, tmp_path):
        """Should evaluate and return metrics dict."""
        import scripts.evaluate_detector as eval_mod

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 2\nnames: [apple, banana]\n")

        model_path = tmp_path / "best.pt"
        model_path.write_text("fake model")

        mock_model = MagicMock()
        mock_model.names = {0: "apple", 1: "banana"}
        mock_metrics = MagicMock()
        mock_metrics.box.mp = 0.85
        mock_metrics.box.mr = 0.80
        mock_metrics.box.map50 = 0.87
        mock_metrics.box.map = 0.72
        mock_metrics.box.maps = np.array([0.9, 0.84])
        mock_metrics.box.p = np.array([0.9, 0.8])
        mock_metrics.box.r = np.array([0.85, 0.75])
        mock_model.val.return_value = mock_metrics
        mock_yolo_cls.return_value = mock_model

        results = eval_mod.evaluate_detector(
            model_path=model_path,
            data_yaml=data_yaml,
        )

        assert results["status"] == "available"
        assert results["precision"] == 0.85
        assert results["recall"] == 0.80
        assert results["map50"] == 0.87
        assert "apple" in results["per_class"]
        assert "banana" in results["per_class"]

