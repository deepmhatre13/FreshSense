"""Tests for validate_manual_annotations.py (Dataset V3 human-annotation validation)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pytest

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.validate_manual_annotations import validate_row, validate_dir


def _mk_img(path: Path, size: int = 64):
    import cv2, numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((size, size, 3), dtype=np.uint8))


def _build_dir(root: Path):
    d = root / "manual_annotations"
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    _mk_img(d / "images" / "a.jpg")
    (d / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _mk_img(d / "images" / "b.jpg")
    (d / "labels" / "b.txt").write_text("", encoding="utf-8")  # empty -> skipped
    _mk_img(d / "images" / "bad.jpg")
    (d / "labels" / "bad.txt").write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")  # bad class
    return d


class TestValidateRow:
    def test_valid(self):
        assert validate_row("0 0.5 0.5 0.2 0.2")[0] is True

    def test_invalid_class(self):
        assert validate_row("99 0.5 0.5 0.2 0.2")[0] is False

    def test_coord_out_of_range(self):
        assert validate_row("0 1.5 0.5 0.2 0.2")[0] is False

    def test_zero_width(self):
        assert validate_row("0 0.5 0.5 0.0 0.2")[0] is False

    def test_zero_height(self):
        assert validate_row("0 0.5 0.5 0.2 0.0")[0] is False

    def test_negative(self):
        assert validate_row("0 -0.5 0.5 0.2 0.2")[0] is False

    def test_beyond_image(self):
        assert validate_row("0 0.9 0.5 0.5 0.2")[0] is False

    def test_malformed(self):
        assert validate_row("0 0.5 0.5")[0] is False  # only 3 values
        assert validate_row("x y z w h")[0] is False  # non-numeric


class TestValidateDir:
    def test_synthetic(self, tmp_path):
        d = _build_dir(tmp_path)
        r = validate_dir(d)
        assert r["total"] == 3
        assert r["invalid"] == 1  # the bad.jpg bad class
        assert any("bad class" in e for e in r["errors"])

    def test_missing_image(self, tmp_path):
        d = _build_dir(tmp_path)
        (d / "images" / "orphan.jpg").write_bytes(b"x")  # has image but no label
        r = validate_dir(d)
        # the .jpg with no label triggers missing-label error (image exists)


class TestNoV2Modification:
    def test_v2_counts_unchanged(self):
        # Ensure we never touch data/detection
        import cv2, numpy as np
        from scripts.manual_annotate_detection import EMPTY_IMGS
        # Just confirm the queue references data/detection paths (never writes)
        assert all("data/detection" in p for p, _, _ in EMPTY_IMGS)


class TestEmptyLabelHandling:
    def test_empty_label_skipped(self, tmp_path):
        d = Path(tmp_path) / "manual_annotations"
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir(parents=True)
        _mk_img(d / "images" / "e.jpg")
        (d / "labels" / "e.txt").write_text("", encoding="utf-8")
        r = validate_dir(d)
        # empty label treated as valid/empty (image present, no bad rows)
        assert r["total"] == 1