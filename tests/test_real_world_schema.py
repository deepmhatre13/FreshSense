"""Tests for the Phase 5 canonical real-world schema and validation."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.real_world_schema import (
    CanonicalRecord,
    find_physical_fruit_leakage,
    find_session_leakage,
    load_canonical_manifest,
    split_manifest_by_physical_fruit,
    validate_canonical_manifest,
    write_manifest_csv,
)


def _make_image(path: Path, color: int = 128, size: int = 32) -> Path:
    """Create a tiny image that is byte-distinct for each distinct path.

    The image content is drawn from a PRNG seeded by the path's full digest,
    so two different ``path`` values produce different bytes (a "valid"
    fixture manifest must not trip the exact-duplicate-file detector).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = int.from_bytes(
        hashlib.md5(str(path).encode("utf-8")).digest()[:4], "little"
    )
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _row(image_id, index=0, **overrides):
    row = {
        "image_id": image_id,
        "image_path": f"images/{image_id}.png",
        "fruit_type": "apples",
        "freshness_label": "fresh",
        "physical_fruit_id": f"F{index % 8}",
        "capture_session_id": f"SESS_{index % 3}",
        "capture_timestamp": "2026-08-01T10:00:00Z",
        "camera_id": "cam1",
    }
    row.update(overrides)
    return row


def _records_from_rows(rows):
    return [CanonicalRecord(dict(r)) for r in rows]


class TestLoadManifest:
    def test_csv_roundtrip(self, tmp_path):
        records = _records_from_rows([_row("IMG_1", 0), _row("IMG_2", 1)])
        out = write_manifest_csv(records, tmp_path / "m.csv")
        loaded = load_canonical_manifest(out)
        assert [r.image_id for r in loaded] == ["IMG_1", "IMG_2"]
        assert loaded[0].fruit_type == "apples"

    def test_load_json_array(self, tmp_path):
        rows = [_row("IMG_1", 0), _row("IMG_2", 1)]
        p = tmp_path / "m.json"
        p.write_text(json.dumps(rows), encoding="utf-8")
        assert [r.image_id for r in load_canonical_manifest(p)] == ["IMG_1", "IMG_2"]

    def test_load_json_records_key(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"records": [_row("IMG_1", 0)]}), encoding="utf-8")
        assert len(load_canonical_manifest(p)) == 1

    def test_load_csv_with_bom(self, tmp_path):
        rows = [_row("IMG_1", 0)]
        fieldnames = sorted(rows[0].keys())
        p = tmp_path / "bom.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        assert load_canonical_manifest(p)[0].image_id == "IMG_1"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_canonical_manifest(tmp_path / "nope.csv")

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "m.txt"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            load_canonical_manifest(p)
class TestValidation:
    def _valid_rows(self, count=24):
        return [_row(f"IMG_{i}", i) for i in range(count)]

    def test_valid_manifest_passes(self, tmp_path):
        rows = self._valid_rows(24)
        for row in rows:
            _make_image(tmp_path / row["image_path"])
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert report.is_pass()
        assert report.error_count == 0
        assert report.total_rows == 24
        assert report.physical_fruit_count == 8
        assert report.capture_session_count == 3
        assert report.class_distribution == {"freshapples": 24}

    def test_missing_required_fields(self, tmp_path):
        rows = self._valid_rows(3)
        del rows[0]["physical_fruit_id"]
        rows[1]["capture_timestamp"] = ""
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert not report.is_pass()
        missing_fields = {f for _, f in report.missing_required}
        assert "physical_fruit_id" in missing_fields
        assert "capture_timestamp" in missing_fields

    def test_invalid_labels(self, tmp_path):
        rows = [_row("IMG_1", 0, fruit_type="durian")]
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert any("durian" in detail for _, detail in report.invalid_labels)

    def test_invalid_numeric_values(self, tmp_path):
        rows = [
            _row("IMG_1", 0, days_since_purchase=-1),
            _row("IMG_2", 1, annotation_confidence=1.7),
            _row("IMG_3", 2, occlusion_level=2.0),
            _row("IMG_4", 3, capture_timestamp="not-a-date"),
        ]
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert report.invalid_values, "expected invalid values to be reported"
        text = json.dumps(report.invalid_values)
        assert "days_since_purchase" in text
        assert "annotation_confidence" in text
        assert "occlusion_level" in text
        assert "capture_timestamp" in text

    def test_duplicate_image_ids(self, tmp_path):
        rows = [_row("IMG_1", 0), _row("IMG_1", 1)]
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert "IMG_1" in report.duplicate_image_ids

    def test_duplicate_paths(self, tmp_path):
        rows = [_row("IMG_1", 0), _row("IMG_2", 1, image_path="images/IMG_1.png")]
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert "images/IMG_1.png" in report.duplicate_paths

    def test_exact_duplicate_files(self, tmp_path):
        rows = [_row("IMG_1", 0), _row("IMG_2", 1, image_path="images/IMG_2.png")]
        _make_image(tmp_path / rows[0]["image_path"], color=7)
        (tmp_path / "images/IMG_2.png").write_bytes(
            (tmp_path / "images/IMG_1.png").read_bytes()
        )
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert len(report.exact_duplicate_files) >= 1

    def test_missing_image_file(self, tmp_path):
        rows = [_row("IMG_1", 0)]
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert report.missing_image_files == ["images/IMG_1.png"]

    def test_fruit_type_conflict_impossible(self, tmp_path):
        rows = [
            _row("IMG_1", 0, physical_fruit_id="X1", fruit_type="apples"),
            _row("IMG_2", 1, physical_fruit_id="X1", fruit_type="banana"),
        ]
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert report.fruit_type_conflicts
        assert any(
            "X1" in detail or "fruit_type" in detail
            for _, detail in report.impossible_combinations
        )

    def test_same_session_fresh_and_rotten_flagged(self, tmp_path):
        rows = [
            _row("IMG_1", 0, physical_fruit_id="X1", capture_session_id="S1", freshness_label="fresh"),
            _row("IMG_2", 1, physical_fruit_id="X1", capture_session_id="S1", freshness_label="rotten"),
        ]
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert any(
            "same capture session" in detail
            for _, detail in report.impossible_combinations
        )

    def test_class_imbalance_flagged(self, tmp_path):
        rows = []
        for i in range(30):
            fl = "fresh" if i < 25 else "rotten"
            rows.append(_row(f"IMG_{i}", i, freshness_label=fl))
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=tmp_path)
        assert report.imbalance_ratio >= 4.0
        assert not report.is_balanced

    def test_no_data_root_skips_file_checks(self, tmp_path):
        rows = self._valid_rows(4)
        report = validate_canonical_manifest(_records_from_rows(rows), data_root=None)
        assert report.missing_image_files == []
        assert report.is_pass()
class TestLeakageDetection:
    def _rec(self, image_id, fruit_id, session_id):
        return CanonicalRecord(
            _row(image_id, 0, physical_fruit_id=fruit_id, capture_session_id=session_id)
        )

    def test_no_fruit_leakage_empty(self):
        splits = {
            "train": [self._rec("a1", "F1", "S1")],
            "val": [self._rec("b1", "F2", "S2")],
            "test": [self._rec("c1", "F3", "S3")],
        }
        assert find_physical_fruit_leakage(splits) == []

    def test_fruit_leakage_detected(self):
        splits = {
            "train": [self._rec("a1", "F1", "S1")],
            "val": [self._rec("b1", "F1", "S2")],
        }
        assert find_physical_fruit_leakage(splits) == [("F1", "train", "val")]

    def test_session_leakage_detected(self):
        splits = {
            "train": [self._rec("a1", "F1", "S1")],
            "val": [self._rec("b1", "F2", "S1")],
        }
        assert find_session_leakage(splits) == [("S1", "train", "val")]


class TestSplitBasics:
    def _rows(self, fruits_per_class=10):
        rows = []
        idx = 0
        for ft, fl in (
            ("apples", "fresh"),
            ("apples", "rotten"),
            ("banana", "fresh"),
            ("banana", "rotten"),
        ):
            for f in range(fruits_per_class):
                idx += 1
                for day in range(1, 2):  # keep one image per fruit for simplicity
                    rows.append(
                        _row(
                            f"IMG_{idx}_{day}",
                            idx,
                            physical_fruit_id=f"F{idx:03d}",
                            fruit_type=ft,
                            freshness_label=fl,
                            capture_session_id=f"S{idx}_{day}",
                        )
                    )
        return rows

    def test_physical_fruit_never_crosses_splits(self):
        rows = self._rows(10)
        result = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=42)
        assert result.verify_no_fruit_leakage()
        train = {r.physical_fruit_id for r in result.train}
        val = {r.physical_fruit_id for r in result.val}
        test = {r.physical_fruit_id for r in result.test}
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)

    def test_full_partition(self):
        rows = self._rows(10)
        result = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=0)
        assert result.verify_full_partition()
        assert result.total == 40
        assert len(result.train) + len(result.val) + len(result.test) == 40

    def test_deterministic_splitting(self):
        rows = self._rows(10)
        a = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=7)
        b = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=7)
        assert [r.image_id for r in a.train] == [r.image_id for r in b.train]
        assert [r.image_id for r in a.val] == [r.image_id for r in b.val]
        assert [r.image_id for r in a.test] == [r.image_id for r in b.test]
        assert a.to_dict()["no_fruit_leakage"] is True

    def test_different_seed_changes_assignment(self):
        rows = self._rows(10)
        a = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=1)
        b = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=2)
        assert sorted({r.physical_fruit_id for r in a.train}) != \
            sorted({r.physical_fruit_id for r in b.train})

    def test_class_balance_report_contains_all_classes(self):
        rows = self._rows(10)
        result = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=42)
        # splits_by_class returns a dict {class: {split: count}}.
        by_class = result.splits_by_class()
        for cls in ("freshapples", "freshbanana", "rottenapples", "rottenbanana"):
            assert cls in by_class
            assert by_class[cls]["train"] > 0
        # class_balance_report returns a human-readable string that mentions classes.
        report_text = result.class_balance_report()
        assert isinstance(report_text, str)
        assert "freshapples" in report_text

    def test_missing_physical_fruit_id_raises(self):
        rows = self._rows(2)
        rows[0]["physical_fruit_id"] = ""
        with pytest.raises(ValueError, match="physical_fruit_id"):
            split_manifest_by_physical_fruit(_records_from_rows(rows))

    def test_missing_labels_raise(self):
        rows = self._rows(2)
        rows[0]["freshness_label"] = ""
        with pytest.raises(ValueError, match="freshness_label"):
            split_manifest_by_physical_fruit(_records_from_rows(rows))

    def test_bad_ratios_raise(self):
        rows = self._rows(2)
        with pytest.raises(ValueError, match="ratios"):
            split_manifest_by_physical_fruit(_records_from_rows(rows), train=0.0)

    def test_ratios_approximately_respected_for_many_fruits(self):
        rows = self._rows(40)
        result = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=42)
        images = result.counts
        total = sum(images.values())
        assert abs(images["train"] / total - 0.7) < 0.1
        assert abs(images["val"] / total - 0.15) < 0.1
        assert abs(images["test"] / total - 0.15) < 0.1

    def test_split_csv_roundtrip(self, tmp_path):
        rows = self._rows(10)
        result = split_manifest_by_physical_fruit(_records_from_rows(rows), seed=42)
        for split_name, records in (
            ("train", result.train),
            ("val", result.val),
            ("test", result.test),
        ):
            out = write_manifest_csv(records, tmp_path / f"{split_name}.csv")
            assert [r.image_id for r in load_canonical_manifest(out)] == \
                [r.image_id for r in records]
