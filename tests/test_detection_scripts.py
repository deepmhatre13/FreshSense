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

class TestDownloadDetectionDatasetScript:
    """Regression tests for scripts/download_detection_dataset.py.

    These validate the str -> Path normalization fix for ``output_dir`` so the
    script's filesystem operations (.mkdir / .is_file / .rglob / / joins) work
    whether a string or a Path is supplied, without hitting the real Roboflow
    API or downloading data.
    """

    def test_imports(self):
        """Script should be importable and expose download_dataset/main."""
        import scripts.download_detection_dataset as dl_mod

        assert hasattr(dl_mod, "download_dataset")
        assert hasattr(dl_mod, "main")

    def test_download_dataset_accepts_string_output_dir(self, tmp_path):
        """A str output path must be normalized and used to create the dir."""
        import scripts.download_detection_dataset as dl_mod

        output_dir = tmp_path / "detection"
        output_str = str(output_dir)

        mock_dataset = MagicMock()
        mock_dataset.location = str(output_dir)
        mock_dataset.classes = ["apple", "banana"]

        with patch("scripts.download_detection_dataset.Roboflow") as mock_rf_cls:
            mock_rf = mock_rf_cls.return_value
            mock_version = MagicMock()
            mock_version.download.return_value = mock_dataset
            mock_rf.workspace.return_value.project.return_value.version.return_value = (
                mock_version
            )

            # Pre-create data.yaml so find_data_yaml() resolves and normalization
            # runs against a real file (no network involved).
            output_dir.mkdir(parents=True)
            (output_dir / "data.yaml").write_text("nc: 2\nnames: [apple, banana]\n")
            (output_dir / "train" / "images").mkdir(parents=True)

            data_yaml, label_audit = dl_mod.download_dataset(
                workspace="ws",
                project="proj",
                version=1,
                output_dir=output_str,
                api_key="fake-key",
                overwrite=False,
            )

        assert output_dir.is_dir()
        assert data_yaml == output_dir / "data.yaml"
        assert label_audit == {"converted_rows": 0, "files_changed": 0}
        mock_version.download.assert_called_once_with(
            "yolov8", location=str(output_dir), overwrite=False
        )

    def test_download_dataset_accepts_path_output_dir(self, tmp_path):
        """A Path output argument must also be accepted unchanged."""
        import scripts.download_detection_dataset as dl_mod

        output_dir = tmp_path / "detection"

        mock_dataset = MagicMock()
        mock_dataset.location = str(output_dir)
        mock_dataset.classes = ["apple", "banana"]

        with patch("scripts.download_detection_dataset.Roboflow") as mock_rf_cls:
            mock_rf = mock_rf_cls.return_value
            mock_version = MagicMock()
            mock_version.download.return_value = mock_dataset
            mock_rf.workspace.return_value.project.return_value.version.return_value = (
                mock_version
            )

            output_dir.mkdir(parents=True)
            (output_dir / "data.yaml").write_text("nc: 2\nnames: [apple, banana]\n")
            (output_dir / "train" / "images").mkdir(parents=True)

            data_yaml, _ = dl_mod.download_dataset(
                workspace="ws",
                project="proj",
                version=1,
                output_dir=output_dir,
                api_key="fake-key",
                format="yolov8",
                overwrite=True,
            )

        assert isinstance(output_dir, Path)
        assert data_yaml == output_dir / "data.yaml"

    def test_download_dataset_creates_output_directory(self, tmp_path):
        """The output directory must actually be created for a fresh path."""
        import scripts.download_detection_dataset as dl_mod

        output_dir = tmp_path / "brand" / "new" / "detection"
        assert not output_dir.exists()

        mock_dataset = MagicMock()
        mock_dataset.location = str(output_dir)
        mock_dataset.classes = []

        with patch("scripts.download_detection_dataset.Roboflow") as mock_rf_cls:
            mock_rf = mock_rf_cls.return_value
            mock_version = MagicMock()
            mock_version.download.return_value = mock_dataset
            mock_rf.workspace.return_value.project.return_value.version.return_value = (
                mock_version
            )

            output_dir.mkdir(parents=True)
            (output_dir / "data.yaml").write_text("nc: 0\nnames: []\n")

            data_yaml, _ = dl_mod.download_dataset(
                workspace="ws",
                project="proj",
                version=1,
                output_dir=str(output_dir),
                api_key="fake-key",
            )

        assert output_dir.is_dir()
        assert data_yaml == output_dir / "data.yaml"

    def test_find_data_yaml_works_with_str_and_path(self, tmp_path):
        """find_data_yaml must resolve data.yaml under a str or Path root."""
        import scripts.download_detection_dataset as dl_mod

        root = tmp_path / "dataset"
        nested = root / "Sub" / "v1"
        nested.mkdir(parents=True)
        (nested / "data.yaml").write_text("nc: 1\n")

        assert dl_mod.find_data_yaml(str(root)) == nested / "data.yaml"
        assert dl_mod.find_data_yaml(root) == nested / "data.yaml"

class TestValidateDetectionDatasetScript:
    """Regression tests for scripts/validate_detection_dataset.py.

    These validate the str -> Path normalization fix for ``data_dir`` so the
    script's filesystem operations (.exists / .is_dir / .glob / .iterdir) work
    whether a string or a Path is supplied, without hitting the real Roboflow
    API or depending on a downloaded dataset.
    """

    @staticmethod
    def _make_valid_dataset(root: Path) -> None:
        """Build a minimal but structurally *valid* YOLO dataset.

        Each split gets one genuine PNG image and one matching, valid YOLO label
        (``class_id cx cy w h``). This avoids triggering the legitimate
        "class has 0 boxes" quality check, which is out of scope for the
        str-vs-Path behaviour these tests target.
        """
        from PIL import Image

        # Ensure the target directory exists before writing data.yaml into it.
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        (root / "data.yaml").write_text(
            "train: train/images\nval: valid/images\ntest: test/images\n"
            "nc: 1\nnames: [apple]\n"
        )
        for split in ("train", "valid", "test"):
            images_dir = root / split / "images"
            labels_dir = root / split / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            # Genuine minimal image (not fake/invalid bytes).
            Image.new("RGB", (64, 64), color="white").save(images_dir / "img0.jpg")
            # Valid YOLO row: class 0, centered box (cx, cy, w, h all in (0,1)).
            (labels_dir / "img0.txt").write_text("0 0.5 0.5 0.5 0.5\n")

    def test_imports(self):
        """Script should be importable and expose validate_dataset/main."""
        import scripts.validate_detection_dataset as val_mod

        assert hasattr(val_mod, "validate_dataset")
        assert hasattr(val_mod, "main")

    def test_validate_dataset_accepts_string_data_dir(self, tmp_path):
        """A str data_dir must be normalised and used to locate data.yaml."""
        import scripts.validate_detection_dataset as val_mod

        dataset_dir = tmp_path / "detection"
        self._make_valid_dataset(dataset_dir)

        report = val_mod.validate_dataset(data_dir=str(dataset_dir))

        assert report["status"] == "pass"
        assert report["dataset_root"] == str(dataset_dir)
        assert report["summary"]["total_images"] == 3
        assert report["summary"]["total_labels"] == 3

    def test_validate_dataset_accepts_path_data_dir(self, tmp_path):
        """A Path data_dir must also be accepted unchanged."""
        import scripts.validate_detection_dataset as val_mod

        dataset_dir = tmp_path / "detection"
        self._make_valid_dataset(dataset_dir)

        report = val_mod.validate_dataset(data_dir=dataset_dir)

        assert report["status"] == "pass"
        assert report["dataset_root"] == str(dataset_dir)
        assert report["summary"]["total_images"] == 3
        assert report["summary"]["total_labels"] == 3

    def test_find_dataset_root_with_str_and_path(self, tmp_path):
        """_find_dataset_root must resolve the root for str and Path inputs."""
        import scripts.validate_detection_dataset as val_mod

        root = tmp_path / "detection"
        root.mkdir(parents=True)
        (root / "data.yaml").write_text("nc: 1\nnames: [apple]\n")

        result_str = val_mod._find_dataset_root(str(root))
        result_path = val_mod._find_dataset_root(root)

        assert result_str == root
        assert result_path == root

    def test_find_dataset_root_nonexistent(self, tmp_path):
        """_find_dataset_root must return None for a missing directory."""
        import scripts.validate_detection_dataset as val_mod

        result = val_mod._find_dataset_root(tmp_path / "nonexistent")
        assert result is None

    def test_find_dataset_root_subfolder(self, tmp_path):
        """_find_dataset_root must find data.yaml inside a version sub-folder."""
        import scripts.validate_detection_dataset as val_mod

        parent = tmp_path / "detection"
        nested = parent / "v1"
        nested.mkdir(parents=True)
        (nested / "data.yaml").write_text("nc: 1\nnames: [apple]\n")

        result = val_mod._find_dataset_root(str(parent))
        assert result == nested

    def test_validate_dataset_missing_data_yaml(self, tmp_path):
        """validate_dataset must return a fail report when data.yaml is absent."""
        import scripts.validate_detection_dataset as val_mod

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        report = val_mod.validate_dataset(data_dir=str(empty_dir))

        assert report["status"] == "fail"
        assert len(report["errors"]) > 0

    def test_validate_dataset_flags_image_without_label_as_warning(self, tmp_path):
        """A split with an image that has no label must produce a warning."""
        import scripts.validate_detection_dataset as val_mod

        dataset_dir = tmp_path / "detection"
        self._make_valid_dataset(dataset_dir)

        # Add an image that has no matching label in the train split.
        from PIL import Image
        Image.new("RGB", (64, 64), color="blue").save(
            dataset_dir / "train" / "images" / "orphan.jpg"
        )

        report = val_mod.validate_dataset(data_dir=dataset_dir)

        # Orphan image -> non-fatal quality concern, dataset is still structurally valid.
        assert report["status"] == "warning"
        assert any("image(s) have no label file" in w for w in report["warnings"])
        assert report["errors"] == []

    def test_validate_dataset_flags_class_with_zero_boxes_as_warning(self, tmp_path):
        """A declared class with zero boxes must produce a warning, not a fail."""
        import scripts.validate_detection_dataset as val_mod

        dataset_dir = tmp_path / "detection"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Two classes, but only class 0 is annotated -> class 1 has 0 boxes.
        (dataset_dir / "data.yaml").write_text(
            "train: train/images\nval: valid/images\ntest: test/images\n"
            "nc: 2\nnames: [apple, grape]\n"
        )
        for split in ("train", "valid", "test"):
            (dataset_dir / split / "images").mkdir(parents=True)
            (dataset_dir / split / "labels").mkdir(parents=True)
            from PIL import Image
            Image.new("RGB", (64, 64), color="white").save(
                dataset_dir / split / "images" / "img0.jpg"
            )
            (dataset_dir / split / "labels" / "img0.txt").write_text("0 0.5 0.5 0.5 0.5\n")

        report = val_mod.validate_dataset(data_dir=dataset_dir)

        assert report["status"] == "warning"
        assert any("class 1 (grape) has 0 boxes" in w for w in report["warnings"])
        assert report["errors"] == []

    def test_validate_dataset_flags_malformed_label_as_fail(self, tmp_path):
        """A label with an out-of-range class id must be a hard error -> fail."""
        import scripts.validate_detection_dataset as val_mod

        dataset_dir = tmp_path / "detection"
        self._make_valid_dataset(dataset_dir)

        # Corrupt one label: class id 5 is out of range for nc=1.
        (dataset_dir / "train" / "labels" / "img0.txt").write_text("5 0.5 0.5 0.5 0.5\n")

        report = val_mod.validate_dataset(data_dir=dataset_dir)

        assert report["status"] == "fail"
        assert any("Class id out of range" in e for e in report["errors"])
class TestEvaluatePerClassMetrics:
    """Per-class AP50-95 reporting for scripts/evaluate_detector.py.

    The detection baseline report requires per-class AP50-95 in addition to
    precision/recall/AP50. These tests verify the extraction and its graceful
    fallback when Ultralytics does not expose the per-class array.
    """

    @patch("ultralytics.YOLO")
    def test_evaluate_per_class_includes_ap50_95(self, mock_yolo_cls, tmp_path):
        """per_class dicts must include ap50_95 when Ultralytics exposes it."""
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
        mock_metrics.box.maps = np.array([0.90, 0.84])
        mock_metrics.box.p = np.array([0.90, 0.80])
        mock_metrics.box.r = np.array([0.85, 0.75])
        mock_metrics.box.ap = np.array([0.70, 0.62])  # per-class AP@[0.5:0.95]
        mock_model.val.return_value = mock_metrics
        mock_yolo_cls.return_value = mock_model

        results = eval_mod.evaluate_detector(
            model_path=model_path,
            data_yaml=data_yaml,
        )

        assert results["per_class"]["apple"]["ap50_95"] == 0.70
        assert results["per_class"]["banana"]["ap50_95"] == 0.62

    @patch("ultralytics.YOLO")
    def test_evaluate_per_class_falls_back_when_ap_missing(self, mock_yolo_cls, tmp_path):
        """ap50_95 must be None (never fabricated) when ``box.ap`` is absent."""
        import scripts.evaluate_detector as eval_mod

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 1\nnames: [apple]\n")
        model_path = tmp_path / "best.pt"
        model_path.write_text("fake model")

        mock_model = MagicMock()
        mock_model.names = {0: "apple"}
        mock_metrics = MagicMock()
        mock_metrics.box.mp = 0.85
        mock_metrics.box.mr = 0.80
        mock_metrics.box.map50 = 0.87
        mock_metrics.box.map = 0.72
        mock_metrics.box.maps = np.array([0.90])
        mock_metrics.box.p = np.array([0.90])
        mock_metrics.box.r = np.array([0.85])
        del mock_metrics.box.ap
        mock_model.val.return_value = mock_metrics
        mock_yolo_cls.return_value = mock_model

        results = eval_mod.evaluate_detector(
            model_path=model_path,
            data_yaml=data_yaml,
        )

        assert results["per_class"]["apple"]["ap50"] == 0.90
        assert results["per_class"]["apple"]["ap50_95"] is None


class TestDetectionTaxonomyMismatch:
    """Verify the 10-class (detector) vs 6-class (freshness) taxonomy handling.

    The YOLO detector recognises 10 fruit classes, but the EfficientNet freshness
    classifier only grades apple/banana/orange. Unsupported fruits must be
    explicitly reported as not-supported, never silently mapped to a wrong
    freshness class.
    """

    def test_freshness_supported_for_known_fruits(self):
        """apple/banana/orange must be freshness-supported."""
        from src.inference.detection_pipeline import freshness_supported

        assert freshness_supported("apple")
        assert freshness_supported("banana")
        assert freshness_supported("orange")
        assert freshness_supported("APPLE")

    def test_freshness_not_supported_for_new_fruits(self):
        """grape/kiwi/mango/etc. must NOT be silently mapped to a wrong class."""
        from src.inference.detection_pipeline import freshness_supported

        for fruit in ("grape", "kiwi", "mango", "strawberry", "cherry", "chickoo", "guava"):
            assert not freshness_supported(fruit), fruit

    def test_freshness_supported_empty_input(self):
        """Empty/blank input must be treated as not-supported."""
        from src.inference.detection_pipeline import freshness_supported

        assert not freshness_supported("")
        assert not freshness_supported(None)