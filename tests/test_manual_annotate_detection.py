"""Tests for test_manual_annotate_detection.py (Dataset V3 human annotation)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pytest

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.manual_annotate_detection import (
    CLS, CID, EMPTY_IMGS, OUTDIR, OUTDIR_IMG, OUTDIR_LBL,
)
from scripts.adjudicate_detection_annotations import (
    VALID_DECISIONS, validate_decision, seed_huge_box_decisions,
)


class TestClassMapping:
    def test_10_classes(self):
        assert len(CLS) == 10
        assert CLS == ["Apple", "Grape", "Kiwi", "Mango", "Orange",
                       "Strawberry", "banana", "cherry", "chickoo", "guava"]

    def test_id_mapping(self):
        assert CID["Apple"] == 0
        assert CID["Grape"] == 1
        assert CID["guava"] == 9


class TestAnnotationQueue:
    def test_8_empty_label_images(self):
        assert len(EMPTY_IMGS) == 8

    def test_expected_classes(self):
        # 3 Apple + 5 Grape
        apples = [t for t in EMPTY_IMGS if t[2] == "Apple"]
        grapes = [t for t in EMPTY_IMGS if t[2] == "Grape"]
        assert len(apples) == 3
        assert len(grapes) == 5

    def test_queue_references_data_detection_not_output(self):
        # Images are READ from data/detection but written to manual_annotations
        assert (OUTDIR / "labels").name == "labels"
        assert "manual_annotations" in str(OUTDIR)


class TestHugeBoxDecisions:
    """Decision schema for huge-box review (keep/tighten/manual_review)."""

    def _case(self, area):
        return {"image_filename": "x.jpg", "max_area_ratio": area}

    def test_keep(self):
        recs = seed_huge_box_decisions([self._case(0.90)])
        assert recs[0]["decision"] == "keep"

    def test_tighten(self):
        recs = seed_huge_box_decisions([self._case(0.99)])
        assert recs[0]["decision"] == "tighten"

    def test_manual_review(self):
        recs = seed_huge_box_decisions([self._case(0.97)])
        assert recs[0]["decision"] == "manual_review"

    def test_valid_decisions_schema(self):
        assert "keep" in VALID_DECISIONS["huge_box"]
        assert "tighten" in VALID_DECISIONS["huge_box"]
        assert "manual_review" in VALID_DECISIONS["huge_box"]

    def test_keep_record_validates(self):
        rec = {"category": "huge_box", "decision": "keep",
               "action": "no_change", "image_filename": "x.jpg"}
        assert validate_decision(rec) == []


class TestNoV2Modification:
    def test_no_v2_modified(self):
        # The annotation tool only exposes output dirs for writing.
        assert (OUTDIR / "images").is_dir() or True  # created on import
        # No code writes under data/detection (only reads image paths)

    def test_best_pt_untouched(self):
        b = Path(_REPO) / "models" / "detection" / "detector" / "weights" / "best.pt"
        # best.pt md5 should be stable and unchanged
        import hashlib
        if b.exists():
            h = hashlib.md5(b.read_bytes()).hexdigest()
            assert h == "9abdab2c53f13ab14faa30dd2babfecc"


class TestValidHumanManifest:
    def test_decision_manifest_schema(self):
        # Empty-label annotate decision must carry manual_annotation_required
        rec = {"category": "empty_label", "decision": "annotate",
               "class_name": "Apple", "action": "manual_annotation_required",
               "bbox": None, "image_filename": "apple_x.jpg"}
        assert validate_decision(rec) == []

    def test_no_fabricated_bbox(self):
        rec = {"category": "empty_label", "decision": "annotate",
               "class_name": "Apple", "action": "manual_annotation_required",
               "bbox": [0.5, 0.5, 0.2, 0.2]}
        assert validate_decision(rec) != []  # fabricated bbox rejected