"""Integration smoke tests for the Phase 5 dataset scripts (CLI entry points)."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import scripts.create_dataset_split as split_script
import scripts.validate_real_world_dataset as validate_script


def _make_fixture(tmp_path, images=24) -> Path:
    """Create a small realistic fixture: images + canonical manifest."""
    root = tmp_path / "fixture"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "accepted").mkdir(exist_ok=True)
    (root / "metadata").mkdir(exist_ok=True)

    rows = []
    for i in range(images):
        iid = f"IMG_{i:03d}"
        path = root / "images" / f"{iid}.png"
        seed = int.from_bytes(
            hashlib.md5(str(path).encode("utf-8")).digest()[:4], "little"
        )
        img = np.random.default_rng(seed).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)
        rows.append(
            {
                "image_id": iid,
                "image_path": f"images/{iid}.png",
                "fruit_type": "apples" if i % 2 == 0 else "banana",
                "freshness_label": "fresh" if i % 3 else "rotten",
                "physical_fruit_id": f"F{i:03d}",
                "capture_session_id": f"S{i % 4}",
                "capture_timestamp": f"2026-08-{(i % 9) + 1:02d}T10:00:00Z",
                "camera_id": "cam1",
            }
        )
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return root


class TestCreateDatasetSplitScript:
    def test_success_generates_splits_and_report(self, tmp_path, monkeypatch):
        root = _make_fixture(tmp_path)
        out = root / "splits"
        monkeypatch.setattr(
            "sys.argv",
            [
                "create_dataset_split.py",
                "--manifest", str(root / "manifest.csv"),
                "--data-dir", str(root),
                "--out-dir", str(out),
                "--seed", "42",
            ],
        )
        rc = split_script.main()
        assert rc == 0, "splitter should succeed on a valid fixture"
        for name in ("train.csv", "val.csv", "test.csv", "split_report.json"):
            assert (out / name).exists(), name
        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        assert report["no_fruit_leakage"] is True
        assert report["full_partition"] is True
        assert report["total_images"] == 24
        assert set(report["splits_by_class"]) == {
            "freshapples", "freshbanana", "rottenapples", "rottenbanana",
        }

    def test_invalid_manifest_aborts(self, tmp_path, monkeypatch):
        root = tmp_path / "bad"
        root.mkdir()
        rows = [
            {
                "image_id": "i1",
                "image_path": "images/i1.png",  # file does not exist
                "fruit_type": "apples",
                "freshness_label": "fresh",
                # missing physical_fruit_id intentionally
                "capture_session_id": "S1",
                "capture_timestamp": "2026-08-01T00:00:00Z",
                "camera_id": "c1",
            }
        ]
        with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        monkeypatch.setattr(
            "sys.argv",
            [
                "create_dataset_split.py",
                "--manifest", str(root / "manifest.csv"),
                "--data-dir", str(root),
                "--out-dir", str(root / "splits"),
            ],
        )
        rc = split_script.main()
        assert rc == 1  # fails loudly on invalid manifest
        assert not (root / "splits" / "train.csv").exists()

    def test_missing_manifest_aborts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "create_dataset_split.py",
                "--manifest", str(tmp_path / "missing.csv"),
                "--data-dir", str(tmp_path),
                "--out-dir", str(tmp_path / "splits"),
            ],
        )
        assert split_script.main() == 1
class TestValidateScript:
    def test_writes_machine_readable_reports(self, tmp_path, monkeypatch):
        root = _make_fixture(tmp_path)
        # First produce a valid split so the validation has split files to audit.
        monkeypatch.setattr(
            "sys.argv",
            [
                "create_dataset_split.py",
                "--manifest", str(root / "manifest.csv"),
                "--data-dir", str(root),
                "--out-dir", str(root / "splits"),
                "--seed", "42",
            ],
        )
        assert split_script.main() == 0

        report_dir = tmp_path / "reports"
        monkeypatch.setattr(
            "sys.argv",
            [
                "validate_real_world_dataset.py",
                "--data-dir", str(root),
                "--report-dir", str(report_dir),
            ],
        )
        rc = validate_script.main()  # NOT READY is fine: no accepted/ images
        assert rc in (0, 1)

        for name in ("dataset_validation.json", "dataset_validation.md"):
            assert (report_dir / name).exists(), name

        payload = json.loads(
            (report_dir / "dataset_validation.json").read_text(encoding="utf-8")
        )
        canonical = payload["canonical_manifest"]
        assert canonical["manifest_present"] is True
        assert canonical["error_count"] == 0
        assert canonical["physical_fruit_leakage"] == []
        assert canonical["total_rows"] == 24
        assert canonical["class_distribution"]["freshapples"] > 0
