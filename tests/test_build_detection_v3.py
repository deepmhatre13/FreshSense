"""Tests for the V3 dataset builder (scripts/build_detection_v3.py).

Uses small synthetic datasets in temp dirs so tests never touch the real
``data/detection`` export or ``best.pt``, and never construct a real V3.
"""
from __future__ import annotations

import json
import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.build_detection_v3 import (  # noqa: E402
    check_gate,
    construct_v3,
    build_image_v3,
    dedup_rows,
    load_manual_annotations,
    write_data_yaml,
    build_manifest,
    GateResult,
)


def _make_image(path: Path, size: int = 128, color: int = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        seed = int(hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    else:
        img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


NAMES = ["Apple", "Mango", "Kiwi"]


def _build_v2(root: Path) -> Path:
    """Create a minimal frozen V2 dataset (3 classes, one image per split)."""
    (root / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnc: 3\n"
        "names:\n- Apple\n- Mango\n- Kiwi\n", encoding="utf-8")
    # train/a.jpg: one Apple (normalized YOLO)
    (root / "train/images").mkdir(parents=True)
    (root / "train/labels").mkdir(parents=True)
    _make_image(root / "train/images/a.jpg")
    (root / "train/labels/a.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    # valid/v.jpg: one Mango
    (root / "valid/images").mkdir(parents=True)
    (root / "valid/labels").mkdir(parents=True)
    _make_image(root / "valid/images/v.jpg")
    (root / "valid/labels/v.txt").write_text("1 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    # test/t.jpg: one Kiwi
    (root / "test/images").mkdir(parents=True)
    (root / "test/labels").mkdir(parents=True)
    _make_image(root / "test/images/t.jpg")
    (root / "test/labels/t.txt").write_text("2 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    return root


def _make_decisions(records) -> dict:
    return {"schema_version": 1, "source": "human-adjudication", "record_count": len(records),
            "records": records}


def _proposal_record(image, split, hd, cls_id=0, cls_name="Apple", box=None):
    box = box or [0.2, 0.2, 0.5, 0.5]
    prop = {"proposal_id": "p1", "image": str(image), "class_id": cls_id,
            "class_name": cls_name, "x1": box[0], "y1": box[1], "x2": box[2],
            "y2": box[3], "confidence": 0.9}
    return {"image": str(image), "image_filename": Path(str(image)).name, "split": split,
            "review_category": "ambiguous_classes", "ai_proposals": [prop],
            "human_decision": hd, "final_class": cls_name,
            "final_boxes": [list(box)], "reviewer": "test", "timestamp": "now"}


class TestGateBlocks:
    def test_refuses_unresolved_uncertain(self, tmp_path):
        root = _build_v2(tmp_path)
        rec = _proposal_record(root / "train/images/a.jpg", "train", "uncertain")
        dec = _make_decisions([rec])
        gate = check_gate(root, dec, [], (None, {}), NAMES, Path(tmp_path / "no-policy.md"))
        assert not gate.passed
        assert gate.unresolved_proposal_records == 1
        assert any("unresolved" in r for r in gate.reasons)

    def test_refuses_malformed_decision(self, tmp_path):
        root = _build_v2(tmp_path)
        dec = _make_decisions([{"image": "x.jpg", "human_decision": "nonsense"}])
        gate = check_gate(root, dec, [], (None, {}), NAMES, None)
        assert not gate.passed
        assert gate.malformed >= 1

    def test_refuses_invalid_box(self, tmp_path):
        root = _build_v2(tmp_path)
        # box extends beyond the 128x128 image frame
        rec = _proposal_record(root / "train/images/a.jpg", "train", "accepted",
                               box=[0, 0, 200, 200])
        gate = check_gate(root, _make_decisions([rec]), [], (None, {}), NAMES, None)
        assert not gate.passed
        assert any("invalid box" in r for r in gate.reasons)

    def test_refuses_invalid_class_id(self, tmp_path):
        root = _build_v2(tmp_path)
        rec = _proposal_record(root / "train/images/a.jpg", "train", "accepted", cls_id=99)
        gate = check_gate(root, _make_decisions([rec]), [], (None, {}), NAMES, None)
        assert not gate.passed
        assert any("invalid class id" in r for r in gate.reasons)

    def test_refuses_invalid_class_name(self, tmp_path):
        root = _build_v2(tmp_path)
        rec = _proposal_record(root / "train/images/a.jpg", "train", "accepted",
                               cls_name="notafruit")
        gate = check_gate(root, _make_decisions([rec]), [], (None, {}), NAMES, None)
        assert not gate.passed
        assert any("invalid class name" in r for r in gate.reasons)


class TestConstruction:
    def _build_with_decisions(self, root, decisions, out, manual=(None, {})):
        return construct_v3(root, out, _make_decisions(decisions), manual, NAMES,
                            manifest_meta={})

    def test_preserves_original_v2_labels(self, tmp_path):
        root = _build_v2(tmp_path)
        out = tmp_path / "v3"
        self._build_with_decisions(root, [], out)
        lbl = out / "train/labels/a.txt"
        lines = [l.split() for l in lbl.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert [int(lines[0][0])] + [float(x) for x in lines[0][1:]] == \
            [0, 0.5, 0.5, 0.3, 0.3]

    def test_incorporates_accepted_proposal(self, tmp_path):
        root = _build_v2(tmp_path)
        out = tmp_path / "v3"
        rec = _proposal_record(root / "train/images/a.jpg", "train", "accepted",
                               cls_id=0, cls_name="Apple",
                               box=[10, 10, 40, 40])
        self._build_with_decisions(root, [rec], out)
        lines = [l.split() for l in (out / "train/labels/a.txt").read_text().strip().splitlines()]
        assert len(lines) == 2  # original + accepted
        classes = [int(l[0]) for l in lines]
        assert classes.count(0) == 2

    def test_rejects_rejected_proposal(self, tmp_path):
        root = _build_v2(tmp_path)
        out = tmp_path / "v3"
        rec = _proposal_record(root / "train/images/a.jpg", "train", "rejected")
        self._build_with_decisions(root, [rec], out)
        lines = (out / "train/labels/a.txt").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_preserves_keep_original(self, tmp_path):
        root = _build_v2(tmp_path)
        out = tmp_path / "v3"
        rec = _proposal_record(root / "train/images/a.jpg", "train", "kept")
        self._build_with_decisions(root, [rec], out)
        lines = [l.split() for l in (out / "train/labels/a.txt").read_text().strip().splitlines()]
        assert len(lines) == 1
        assert [int(lines[0][0])] + [float(x) for x in lines[0][1:]] == \
            [0, 0.5, 0.5, 0.3, 0.3]

    def test_handles_manual_annotations(self, tmp_path):
        root = _build_v2(tmp_path)
        out = tmp_path / "v3"
        _make_image(root / "train/images/empty.jpg")
        (root / "train/labels/empty.txt").write_text("", encoding="utf-8")
        manual_labels = {"empty": [[0, 0.5, 0.5, 0.2, 0.2]]}
        self._build_with_decisions(root, [], out, manual=(None, manual_labels))
        lines = (out / "train/labels/empty.txt").read_text().strip().splitlines()
        assert len(lines) == 1
        assert int(lines[0].split()[0]) == 0

    def test_creates_valid_data_yaml(self, tmp_path):
        root = _build_v2(tmp_path)
        out = tmp_path / "v3"
        self._build_with_decisions(root, [], out)
        import yaml
        cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        assert cfg["nc"] == 3
        assert cfg["names"] == NAMES
        assert cfg["train"] == "train/images"

    def test_creates_valid_manifest(self, tmp_path):
        root = _build_v2(tmp_path)
        out = tmp_path / "v3"
        self._build_with_decisions(root, [], out)
        man = json.loads((out / "v3_manifest.json").read_text(encoding="utf-8"))
        for key in ["builder_version", "source_dataset", "creation_timestamp",
                    "split_counts", "class_counts", "validation_result",
                    "original_annotations_retained", "ai_annotations_accepted"]:
            assert key in man
        assert man["validation_result"] == "passed"
        assert man["builder_version"] == "1.0.0"

    def test_preserves_train_valid_test_split(self, tmp_path):
        root = _build_v2(tmp_path)
        out = tmp_path / "v3"
        self._build_with_decisions(root, [], out)
        assert (out / "train/images/a.jpg").exists()
        assert (out / "valid/images/v.jpg").exists()
        assert (out / "test/images/t.jpg").exists()

    def test_detects_duplicate_annotations(self):
        rows = [[0, 0.5, 0.5, 0.3, 0.3], [0, 0.5, 0.5, 0.3, 0.3]]
        dedup = dedup_rows(rows, iou_threshold=0.9)
        assert len(dedup) == 1

    def test_never_writes_to_detection(self, tmp_path):
        root = _build_v2(tmp_path)
        before = {str(p): p.read_bytes() for p in root.rglob("*.jpg")}
        out = tmp_path.parent / f"{tmp_path.name}_v3out"  # outside source root
        self._build_with_decisions(root, [], out)
        for p in root.rglob("*.jpg"):
            assert before[str(p)] == p.read_bytes()
        shutil.rmtree(out, ignore_errors=True)
