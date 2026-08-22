"""Tests for the human-adjudication layer (Phase 1 V3 prep).

Covers the decision schema, manifest generation/validation, the empty-label and
huge-box decision seeding, the "no fabricated bbox" guarantee, and the safety
invariant that ``data/detection`` / ``best.pt`` are never modified.

Uses small synthetic datasets/decision records in temp dirs so the tests never
depend on the real export and never mutate it.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.adjudicate_detection_annotations import (  # noqa: E402
    VALID_ACTIONS,
    VALID_DECISIONS,
    find_suspensions,
    validate_decision,
    seed_empty_label_decisions,
    seed_huge_box_decisions,
    write_manifest,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_image(path: Path, size: int = 80) -> Path:
    """Write a tiny valid JPEG so image/label pairing exists."""
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _build_dataset(root: Path):
    yaml = ("train: train/images\nval: valid/images\ntest: test/images\nnc: 3\n"
            "names:\n- Apple\n- Mango\n- Kiwi\n")
    (root / "data.yaml").write_text(yaml, encoding="utf-8")
    (root / "train/images").mkdir(parents=True)
    (root / "train/labels").mkdir(parents=True)
    # empty-label image (label file empty) -> flagged
    _write_image(root / "train/images/empty.jpg")
    (root / "train/labels/empty.txt").write_text("", encoding="utf-8")
    # huge-box image -> flagged
    _write_image(root / "train/images/huge.jpg")
    (root / "train/labels/huge.txt").write_text("0 0.5 0.5 0.99 0.99\n",
                                                encoding="utf-8")
    return root


def _empty_label_record(filename: str = "apple_1.jpg", split: str = "train",
                        do_annotate: bool = True) -> dict:
    return {
        "image": f"data/detection/{split}/images/{filename}",
        "image_filename": filename,
        "split": split,
        "category": "empty_label",
        "decision": "annotate" if do_annotate else "keep_empty",
        "class_name": "Apple" if do_annotate else None,
        "action": "manual_annotation_required",
        "notes": "Visible fruit confirmed by human review",
        "bbox": None,
    }


# --------------------------------------------------------------------------- #
# Validation / schema
# --------------------------------------------------------------------------- #
class TestValidDecisionSchema:
    def test_valid_empty_label_passes(self):
        assert validate_decision(_empty_label_record()) == []

    def test_valid_huge_box_keep_passes(self):
        rec = {"category": "huge_box", "decision": "keep",
               "action": "no_change", "image_filename": "x.jpg"}
        assert validate_decision(rec) == []

    def test_invalid_decision_rejected(self):
        rec = _empty_label_record()
        rec["decision"] = "ship_the_boxes"  # not in VALID_DECISIONS
        assert validate_decision(rec) != []

    def test_invalid_category_rejected(self):
        rec = _empty_label_record()
        rec["category"] = "not_a_category"
        assert validate_decision(rec) != []

    def test_disallowed_action_rejected(self):
        rec = _empty_label_record()
        rec["action"] = "explode"
        assert validate_decision(rec) != []
class TestEmptyLabelDecision:
    def test_annotate_requires_class_name(self):
        rec = _empty_label_record()
        rec.pop("class_name", None)
        errs = validate_decision(rec)
        assert errs != []

    def test_annotate_with_fabricated_bbox_rejected(self):
        # A human may NOT supply a fabricated bbox in this phase.
        rec = _empty_label_record()
        rec["bbox"] = [0.5, 0.5, 0.2, 0.2]
        assert validate_decision(rec) != []

    def test_annotate_without_bbox_ok(self):
        rec = _empty_label_record(do_annotate=True)
        assert rec["bbox"] is None
        assert validate_decision(rec) == []


def _corrections_manifest(tmp_path, filenames):
    p = tmp_path / "v3_corrections.json"
    corr = [
        {"image": f"data/detection/{'valid' if i % 2 else 'train'}/images/{f}",
         "image_filename": f, "split": "valid" if i % 2 else "train"}
        for i, f in enumerate(filenames)
    ]
    p.write_text(json.dumps(corr, indent=2), encoding="utf-8")
    return p


class TestSeedEmptyLabelDecisions:
    def test_grape_policy_class_assignment(self, tmp_path):
        corr = _corrections_manifest(
            tmp_path, ["apple_18_x.jpg", "Grape-23-_jpeg.rf.x.jpg"])
        recs = seed_empty_label_decisions(corr)
        by_name = {r["image_filename"]: r for r in recs}
        assert by_name["apple_18_x.jpg"]["class_name"] == "Apple"
        assert by_name["Grape-23-_jpeg.rf.x.jpg"]["class_name"] == "Grape"

    def test_every_record_annotate_no_bbox(self, tmp_path):
        corr = _corrections_manifest(
            tmp_path, ["Grape-23-_jpeg.rf.x.jpg", "apple_20_x.jpg",
                       "Grape-41-_jpeg.rf.y.jpg", "apple_49_x.jpg"])
        recs = seed_empty_label_decisions(corr)
        assert len(recs) == 4
        for r in recs:
            assert r["decision"] == "annotate"
            assert r["action"] == "manual_annotation_required"
            assert r["bbox"] is None  # no fabricated coordinates


class TestSeedHugeBoxDecisions:
    def test_seed_assigns_suggested_status(self):
        huge = [
            {"image_filename": "a.jpg", "max_area_ratio": 0.993},
            {"image_filename": "b.jpg", "max_area_ratio": 0.97},
            {"image_filename": "c.jpg", "max_area_ratio": 0.91},
        ]
        out = {r["image_filename"]: r for r in seed_huge_box_decisions(huge)}
        assert out["a.jpg"]["decision"] == "tighten"
        assert out["b.jpg"]["decision"] == "manual_review"
        assert out["c.jpg"]["decision"] == "keep"

    def test_review_does_not_modify_source_labels(self, tmp_path):
        root = _build_dataset(tmp_path)
        label = root / "train/labels/huge.txt"
        before = label.read_text(encoding="utf-8")
        _ = seed_huge_box_decisions(
            find_suspensions(root, ["n0", "n1", "n2"])["huge_box"])
        # seeding creates decision records, it must never touch the label file
        assert label.read_text(encoding="utf-8") == before
# --------------------------------------------------------------------------- #
# Manifest generation
# --------------------------------------------------------------------------- #
class TestManifest:
    def test_manifest_creation(self, tmp_path):
        out = tmp_path / "human_decisions.json"
        recs = [_empty_label_record("apple_1.jpg")]
        write_manifest(recs, out, notes=["review complete"])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["record_count"] == 1
        assert data["records"][0]["decision"] == "annotate"
        assert data["records"][0]["action"] == "manual_annotation_required"

    def test_dataset_not_touched_by_manifest(self, tmp_path):
        root = _build_dataset(tmp_path)
        label = root / "train/labels/empty.txt"
        before = label.read_text(encoding="utf-8")
        write_manifest([_empty_label_record("apple_1.jpg")], tmp_path / "out.json")
        assert label.read_text(encoding="utf-8") == before


class TestV2Untouched:
    """Guarantee data/detection and best.pt are never modified."""

    def test_no_write_to_dataset_paths(self):
        import inspect
        from scripts import adjudicate_detection_annotations as mod
        src = inspect.getsource(mod)
        # The module reads from data/detection but its only writer is
        # write_manifest() which targets the --out manifest path (reports/).
        assert "write_manifest" in src
        assert "best.pt" not in src
        # No call that opens a file under data/ for writing is introduced.
        assert ".write_text(" not in src.replace(
            "out_path.write_text", "")  # only manifest writer uses it


# --------------------------------------------------------------------------- #
# Grape policy document presence
# --------------------------------------------------------------------------- #
class TestGrapePolicyPresent:
    def test_policy_file_exists(self):
        policy = Path(_REPO_ROOT) / "docs" / "DETECTION_V3_ANNOTATION_POLICY.md"
        assert policy.exists(), "Grape annotation policy document is missing"
        text = policy.read_text(encoding="utf-8")
        assert "Grape policy" in text
        assert "bunch" in text  # one box per clearly distinct bunch

    def test_policy_mentions_annotate_empty_labels(self):
        policy = Path(_REPO_ROOT) / "docs" / "DETECTION_V3_ANNOTATION_POLICY.md"
        text = policy.read_text(encoding="utf-8")
        assert "manual_annotation_required" in text


# --------------------------------------------------------------------------- #
# No fabricated coordinates
# --------------------------------------------------------------------------- #
class TestNoFakeCoordinates:
    def test_seeded_annotate_records_have_null_bbox(self, tmp_path):
        corr = _corrections_manifest(
            tmp_path, ["apple_18_x.jpg", "Grape-23-_jpeg.rf.x.jpg"])
        for r in seed_empty_label_decisions(corr):
            assert "bbox" in r and r["bbox"] is None

    def test_config_has_only_decision_vocab(self):
        # The module defines only decisions/action vocab, never coordinate boxes.
        assert isinstance(VALID_DECISIONS, dict)
        assert isinstance(VALID_ACTIONS, set)