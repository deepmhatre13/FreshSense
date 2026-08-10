"""Tests for the Phase 5 baseline evaluator output."""
import json

import pytest

import scripts.baseline_evaluation as BE
from scripts.baseline_evaluation import (
    BaselineResult,
    NOT_AVAILABLE,
    _compute_metrics,
    _discover_dataset,
    _scan_class_folder,
)


class TestBaselineDefaults:
    def test_all_metrics_default_to_not_available(self):
        r = BaselineResult()
        assert r.status == NOT_AVAILABLE
        assert r.accuracy == NOT_AVAILABLE
        assert r.macro_precision == NOT_AVAILABLE
        assert r.macro_recall == NOT_AVAILABLE
        assert r.macro_f1 == NOT_AVAILABLE
        assert r.weighted_f1 == NOT_AVAILABLE
        assert r.balanced_accuracy == NOT_AVAILABLE
        assert r.confusion_matrix == NOT_AVAILABLE
        assert r.total_samples == 0
        assert r.class_distribution == {}

    def test_to_dict_json_serializable(self):
        assert isinstance(json.dumps(BaselineResult().to_dict()), str)


class TestComputeMetrics:
    def _args(self):
        return (
            ["freshapples", "freshapples", "freshbanana", "rottenapples"],
            ["freshapples", "freshapples", "freshapples", "rottenapples"],
            ["freshapples", "freshbanana", "rottenapples"],
        )

    def test_accuracy_and_global_metrics(self):
        true_names, pred_names, class_names = self._args()
        r = _compute_metrics(true_names, pred_names, class_names, total_available=4)
        assert r.status == "OK"
        assert r.total_samples == 4
        assert r.class_distribution == {
            "freshapples": 2,
            "freshbanana": 1,
            "rottenapples": 1,
        }
        assert r.accuracy == 0.75
        assert r.macro_precision == pytest.approx(0.555556, abs=1e-5)
        assert r.macro_recall == pytest.approx(0.666667, abs=1e-5)
        assert r.macro_f1 == pytest.approx(0.6, abs=1e-6)
        assert r.weighted_f1 == pytest.approx(0.65, abs=1e-6)
        assert r.balanced_accuracy == pytest.approx(0.666667, abs=1e-5)

    def test_per_class_metrics(self):
        true_names, pred_names, class_names = self._args()
        r = _compute_metrics(true_names, pred_names, class_names, total_available=4)
        assert r.per_class_precision["freshapples"] == pytest.approx(0.666667, abs=1e-5)
        assert r.per_class_precision["freshbanana"] == 0.0
        assert r.per_class_precision["rottenapples"] == 1.0
        assert r.per_class_recall["freshapples"] == 1.0
        assert r.per_class_recall["freshbanana"] == 0.0
        assert r.per_class_f1["rottenapples"] == 1.0

    def test_confusion_matrix_shape_and_values(self):
        true_names, pred_names, class_names = self._args()
        r = _compute_metrics(true_names, pred_names, class_names, total_available=4)
        assert isinstance(r.confusion_matrix, list)
        assert r.confusion_matrix == [[2, 0, 0], [1, 0, 0], [0, 0, 1]]

    def test_confusion_matrix_json_serializable(self):
        true_names, pred_names, class_names = self._args()
        r = _compute_metrics(true_names, pred_names, class_names, total_available=4)
        payload = json.dumps(r.to_dict())
        assert "[[" in payload
        assert "NOT_AVAILABLE" not in payload

    def test_empty_inputs_stay_not_available(self):
        r = _compute_metrics([], [], ["a", "b"], total_available=0)
        assert r.status == NOT_AVAILABLE
        assert r.accuracy == NOT_AVAILABLE
        assert r.total_samples == 0
class TestScanClassFolder:
    def test_scan_detects_classes(self, tmp_path):
        (tmp_path / "freshapples").mkdir()
        (tmp_path / "rottenapples").mkdir()
        (tmp_path / "freshapples" / "a.png").write_bytes(b"x")
        (tmp_path / "freshapples" / "b.png").write_bytes(b"x")
        (tmp_path / "rottenapples" / "c.png").write_bytes(b"x")
        paths, labels, class_names = _scan_class_folder(tmp_path)
        assert class_names == ["freshapples", "rottenapples"]
        assert len(paths) == 3
        assert labels == [0, 0, 1]


class TestDiscover:
    def test_class_folder_source(self, tmp_path):
        (tmp_path / "freshapples").mkdir()
        (tmp_path / "freshapples" / "a.png").write_bytes(b"x")
        paths, labels, class_names, source, notes = _discover_dataset(
            data_root=tmp_path,
            manifest=None,
            default_root=tmp_path / "missing",
        )
        assert len(paths) == 1
        assert labels == ["freshapples"]
        assert class_names == ["freshapples"]
        assert source.startswith("class-folder:")
        assert notes == []

    def test_manifest_source(self, tmp_path):
        from src.data.real_world_schema import CanonicalRecord, write_manifest_csv

        img = tmp_path / "images" / "i1.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"placeholder bytes")
        rec = CanonicalRecord(
            {
                "image_id": "i1",
                "image_path": "images/i1.png",
                "fruit_type": "apples",
                "freshness_label": "fresh",
                "physical_fruit_id": "F1",
                "capture_session_id": "S1",
                "capture_timestamp": "2026-08-01T00:00:00Z",
                "camera_id": "c1",
            }
        )
        mpath = write_manifest_csv([rec], tmp_path / "manifest.csv")
        paths, labels, class_names, source, notes = _discover_dataset(
            data_root=tmp_path,
            manifest=mpath,
            default_root=tmp_path / "missing",
        )
        assert len(paths) == 1
        assert labels == ["freshapples"]
        assert class_names == ["freshapples"]
        assert source.startswith("manifest:")

    def test_no_data_returns_not_available(self, tmp_path):
        paths, labels, class_names, source, notes = _discover_dataset(
            data_root=tmp_path / "empty",
            manifest=tmp_path / "no.csv",
            default_root=tmp_path / "missing_root",
        )
        assert paths == []
        assert NOT_AVAILABLE in notes


class TestMainNotAvailable:
    def test_missing_checkpoint_returns_not_available(self, tmp_path, monkeypatch):
        out = tmp_path / "na.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "baseline_evaluation.py",
                "--checkpoint", "no_such_checkpoint.pth",
                "--manifest", str(tmp_path / "no.csv"),
                "--output", str(out),
            ],
        )
        rc = BE.main()
        assert rc == 1
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["status"] == NOT_AVAILABLE
        assert payload["accuracy"] == NOT_AVAILABLE
