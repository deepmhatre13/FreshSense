"""Tests for the read-only detection-dataset audit (Scripts/audit).

These tests use small **synthetic** datasets created in a temp dir so they do
not depend on the full ``data/detection`` export and never mutate real data.
They cover the audit primitives and the end-to-end report generation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import sys
from pathlib import Path as _P

_REPO_ROOT = str(_P(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.audit_detection_dataset import (  # noqa: E402
    _find_dataset_root,
    _read_boxes,
    audit_dataset,
    build_metadata,
    collect_split_data,
    compute_bbox_stats,
    compute_imbalance,
    load_data_config,
)


def _make_image(path: Path, color: int = 128, size: int = 80) -> Path:
    """Write a small solid-colour image (readable by PIL + cv2)."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _write(data_dir: Path, split: str, rows: dict) -> Path:
    """Build a tiny YOLO split with given ``stem -> label_text`` mappings.

    ``rows`` maps an image stem to a label-file body (or ``None`` = no label).
    """
    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for stem, label in rows.items():
        _make_image(img_dir / f"{stem}.jpg")
        if label is not None:
            (lbl_dir / f"{stem}.txt").write_text(label, encoding="utf-8")
    return data_dir


@pytest.fixture()
def synthetic_dataset(tmp_path: Path):
    """A minimal 3-class dataset with one deliberate label fault per split."""
    (tmp_path / "data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 3\n"
        "names:\n- n0\n- n1\n- n2\n",
        encoding="utf-8",
    )
    # train: 3 images; n0 appears twice, n1 once; one image has NO label.
    _write(tmp_path, "train", {
        "a": "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n",
        "b": "0 0.5 0.5 0.3 0.3\n",
        "c": None,  # image with no label (background)
    })
    # valid: one valid label + one malformed label (wrong field count).
    _write(tmp_path, "valid", {
        "v1": "2 0.5 0.5 0.4 0.4\n",
        "v2": "0 0.5 0.5\n",  # only 3 fields -> malformed
    })
    # test: one invalid class id and one out-of-range coordinate.
    _write(tmp_path, "test", {
        "t1": "5 0.5 0.5 0.2 0.2\n",     # class id 5 >= nc(3)
        "t2": "1 1.5 0.5 0.2 0.2\n",      # cx out of [0,1]
        "t3": "2 0.5 0.5 0.5 0.5\n",
    })
    return tmp_path
class TestDataYamlAndDiscovery:
    def test_load_data_config(self, synthetic_dataset):
        raw, nc, names = load_data_config(synthetic_dataset)
        assert nc == 3
        assert names == ["n0", "n1", "n2"]

    def test_find_dataset_root(self, synthetic_dataset):
        assert _find_dataset_root(synthetic_dataset) == synthetic_dataset
        assert _find_dataset_root(synthetic_dataset / "nope") is None


class TestLabelParsing:
    def test_parse_valid_rows(self, synthetic_dataset):
        lbl = synthetic_dataset / "train/labels/a.txt"
        boxes, issues = _read_boxes(lbl, nc=3)
        assert len(boxes) == 2
        assert issues == []

    def test_invalid_class_detected(self, synthetic_dataset):
        lbl = synthetic_dataset / "test/labels/t1.txt"
        boxes, issues = _read_boxes(lbl, nc=3)
        assert any(i["type"] == "invalid_class" for i in issues)

    def test_malformed_row_detected(self, synthetic_dataset):
        lbl = synthetic_dataset / "valid/labels/v2.txt"
        boxes, issues = _read_boxes(lbl, nc=3)
        assert any(i["type"] == "field_count" for i in issues)

    def test_out_of_range_center_detected(self, synthetic_dataset):
        lbl = synthetic_dataset / "test/labels/t2.txt"
        boxes, issues = _read_boxes(lbl, nc=3)
        assert any(i["type"] == "center_out_of_range" for i in issues)

    def test_bbox_range_width_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            lbl = Path(tmp) / "l.txt"
            lbl.write_text("0 0.5 0.5 -0.2 0.5\n0 0.5 0.5 0.2 1.5\n0 0.5 0.5 1.2 0.5\n",
                           encoding="utf-8")
            boxes, issues = _read_boxes(lbl, nc=3)
            sizes = [i["type"] for i in issues]
            assert sizes.count("size_out_of_range") >= 1


class TestClassCountingAndStats:
    def test_collect_split_class_counts(self, synthetic_dataset):
        metadata = build_metadata(
            [p for p in (synthetic_dataset / "train/images").iterdir()])
        data = collect_split_data("train", synthetic_dataset / "train/images",
                                  synthetic_dataset / "train/labels", 3,
                                  ["n0", "n1", "n2"], metadata)
        assert data["per_class_instances"] == {"0": 2, "1": 1, "2": 0}
        # the image with no label must be counted as 0 objects but present
        assert data["images"] == 3
        assert data["labels"] == 2
        assert len(data["images_without_label"]) == 1

    def test_collect_missing_and_orphan_split(self, synthetic_dataset):
        metadata = build_metadata(
            [p for p in (synthetic_dataset / "valid/images").iterdir()])
        data = collect_split_data("valid", synthetic_dataset / "valid/images",
                                  synthetic_dataset / "valid/labels", 3,
                                  ["n0", "n1", "n2"], metadata)
        # malformed row is still counted? No - skipped because field count wrong
        assert data["total_objects"] == 1  # only v1 valid
        assert any(i["type"] == "field_count" for i in data["issues"])

    def test_compute_imbalance_orders(self):
        per_split = {
            "train": {"0": 10, "1": 2, "2": 5},
            "valid": {"0": 1, "1": 0, "2": 1},
            "test": {"0": 1, "1": 0, "2": 0},
        }
        res = compute_imbalance(per_split, ["n0", "n1", "n2"])
        assert res["most_represented_class"] == "n0"
        assert res["least_represented_class"] == "n1"
        assert res["total_annotated_objects"] == 20

    def test_compute_bbox_stats_categories(self):
        boxes = [
            (0, 0.5, 0.5, 0.05, 0.05, 10, 10, 0.0025),   # small (<0.01)
            (0, 0.5, 0.5, 0.6, 0.6, 50, 50, 0.36),       # large (>0.25)
            (0, 0.5, 0.5, 0.2, 0.2, 20, 20, 0.04),       # medium
        ]
        stats = compute_bbox_stats(boxes, 1, ["n0"])
        cats = stats["0"]["size_categories"]
        assert cats == {"small": 1, "medium": 1, "large": 1}
        assert stats["0"]["count"] == 3


class TestReportGeneration:
    def test_full_audit_writes_valid_json_and_md(self, synthetic_dataset, tmp_path):
        json_out = tmp_path / "audit.json"
        md_out = tmp_path / "audit.md"
        report = audit_dataset(synthetic_dataset, json_out, md_out)

        assert json_out.exists()
        assert md_out.exists()
        # JSON must be valid
        with open(json_out, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["split_summary"]["total_images"] == 8
        # total objects = train a(2 rows)+b(1) + valid v1(1) + test t1,t2,t3 (3) = 7
        assert loaded["split_summary"]["total_objects"] == 7
        assert loaded["label_validation"]["summary"]["num_affected_files"] >= 1
        assert md_out.read_text(encoding="utf-8").startswith("# SmartFreshAI")

    def test_audit_does_not_modify_dataset(self, synthetic_dataset, tmp_path):
        before = {str(p): p.read_bytes()
                  for p in synthetic_dataset.rglob("*") if p.is_file()}
        json_out = tmp_path / "reports" / "audit.json"
        md_out = tmp_path / "reports" / "audit.md"
        audit_dataset(synthetic_dataset, json_out, md_out)
        # Every pre-existing dataset file must still exist with identical bytes.
        for path, content in before.items():
            assert Path(path).exists(), f"dataset file removed: {path}"
            assert Path(path).read_bytes() == content, f"dataset file modified: {path}"