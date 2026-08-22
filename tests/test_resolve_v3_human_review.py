"""Tests for the FINAL V3 human-review resolver.

Covers queue generation, unresolved-blocker detection, empty-label handling,
manual-review/tighten handling, uncertain-proposal handling, coordinate/class
validation, Grape policy, backup/atomic persistence, duplicate prevention,
V2/best.pt immutability, no-V3-creation during resolution, and dry-run
non-modification.

Everything uses synthetic temporary data and never touches the real
``data/detection/``, ``best.pt``, or the real ``human_decisions.json``.
"""
from __future__ import annotations

import copy
import json
import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.resolve_v3_human_review as resolver  # noqa: E402
from scripts.build_detection_v3 import (  # noqa: E402
    load_human_decisions,
)

NAMES = ["Apple", "Grape", "Kiwi", "Mango", "Orange"]
NC = len(NAMES)


# --------------------------------------------------------------------------- #
# Synthetic dataset + decision helpers
# --------------------------------------------------------------------------- #
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


def _build_v2(root: Path) -> Path:
    """Minimal frozen V2 dataset (5 classes) with one image per split."""
    (root / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnc: 5\n"
        "names:\n- Apple\n- Grape\n- Kiwi\n- Mango\n- Orange\n",
        encoding="utf-8")
    (root / "train/images").mkdir(parents=True)
    (root / "train/labels").mkdir(parents=True)
    _make_image(root / "train/images/a.jpg")
    (root / "train/labels/a.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    _make_image(root / "train/images/empty.jpg")
    (root / "train/labels/empty.txt").write_text("", encoding="utf-8")
    _make_image(root / "train/images/huge.jpg")
    (root / "train/labels/huge.txt").write_text("1 0.5 0.5 0.98 0.98\n", encoding="utf-8")
    (root / "valid/images").mkdir(parents=True)
    (root / "valid/labels").mkdir(parents=True)
    _make_image(root / "valid/images/v.jpg")
    (root / "valid/labels/v.txt").write_text("2 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (root / "test/images").mkdir(parents=True)
    (root / "test/labels").mkdir(parents=True)
    _make_image(root / "test/images/t.jpg")
    (root / "test/labels/t.txt").write_text("3 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    return root


def _make_decisions(records) -> dict:
    return {"schema_version": 1, "source": "human-adjudication",
            "record_count": len(records), "records": records}


def _empty_label_image_record(image="empty.jpg", row=None) -> dict:
    """An empty-label adjudication record (decision=annotate, bbox optional)."""
    return {"image": f"data/detection/train/images/{image}",
            "image_filename": Path(image).name,
            "split": "train", "category": "empty_label", "decision": "annotate",
            "class_name": "Apple", "action": "manual_annotation_required",
            "bbox": row}


def _hugebox_image_record(image="huge.jpg", decision="manual_review") -> dict:
    """A huge-box adjudication record."""
    return {"image": f"data/detection/train/images/{image}",
            "image_filename": Path(image).name, "split": "train",
            "category": "huge_box", "decision": decision, "class_name": "Grape",
            "action": "manual_review_required", "max_area_ratio": 0.98,
            "bbox": None}


def _proposal(image, pid="p1", cls_id=0, cls_name="Apple", conf=0.9) -> dict:
    return {"proposal_id": pid, "image": f"data/detection/train/images/{image}",
            "split": "train", "class_id": cls_id, "class_name": cls_name,
            "x1": 20, "y1": 20, "x2": 60, "y2": 60, "confidence": conf,
            "proposal_status": "pending_human_review"}


def _uncertain_record(image, pid="p1", cls_id=0, cls_name="Apple") -> dict:
    return {"image": f"data/detection/train/images/{image}",
            "image_filename": Path(image).name, "split": "train",
            "review_category": "ambiguous_classes",
            "original_annotation": None,
            "ai_proposal": _proposal(image, pid, cls_id, cls_name),
            "human_decision": "uncertain", "final_class": None,
            "final_boxes": [], "reviewer": "cli_batch_reviewer"}


def _state(root: Path, records: list, proposals: list = None,
           manual_dir: Path = None) -> Dict[str, Any]:
    """Load dataset config, decisions, proposals, manual annotations.

    Pure read of the frozen inputs. If ``data.yaml`` is not present under the
    supplied root, a minimal one is auto-created from the module's NAMES constant
    so the helper works with ``tmp_path`` directories.
    """
    proposals = proposals if proposals is not None else []
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            f"train: train/images\nval: valid/images\ntest: test/images\n"
            f"nc: {NC}\n"
            f"names:\n- Apple\n- Grape\n- Kiwi\n- Mango\n- Orange\n",
            encoding="utf-8")
    _, _, class_names = resolver.load_data_config(root)
    return {
        "data_root": root,
        "class_names": class_names,
        "decisions": _make_decisions(records),
        "proposals": proposals,
        "manual": resolver.load_manual_annotations(
            manual_dir or (root.parent / "no_manual")),
    }
# --------------------------------------------------------------------------- #
# 1. Queue generation
# --------------------------------------------------------------------------- #
class TestQueue:
    def test_queue_includes_all_blockers(self, tmp_path):
        root = _build_v2(tmp_path)
        records = [
            _empty_label_image_record(),
            _hugebox_image_record("huge.jpg", "manual_review"),
            _hugebox_image_record("a.jpg", "tighten"),
            _uncertain_record("a.jpg", "p1"),
            _uncertain_record("v.jpg", "p2", cls_name="Mango"),
        ]
        st = _state(root, records,
                    proposals=[_proposal("a.jpg", "p1"),
                               _proposal("v.jpg", "p2")])
        q = resolver.build_review_queue(st)
        cats = {}
        for it in q["items"]:
            cats[it["category"]] = cats.get(it["category"], 0) + 1
        assert cats[resolver.CAT_EMPTY] == 1
        assert cats[resolver.CAT_MANUAL] == 1
        assert cats[resolver.CAT_TIGHTEN] == 1
        assert cats[resolver.CAT_UNCERTAIN] == 2

    def test_queue_excludes_resolved(self, tmp_path):
        root = _build_v2(tmp_path)
        resolved = _uncertain_record("a.jpg", "p1")
        resolved["human_decision"] = "accepted"
        resolved["final_boxes"] = [[20, 20, 60, 60]]
        keep = _hugebox_image_record("huge.jpg", "keep")
        st = _state(root, [resolved, keep], proposals=[_proposal("a.jpg", "p1")])
        q = resolver.build_review_queue(st)
        by_frame = {it["image_filename"] for it in q["items"]}
        assert "a.jpg" not in by_frame
        assert "huge.jpg" not in by_frame

    def test_uncertain_grouped_by_image(self, tmp_path):
        root = _build_v2(tmp_path)
        recs = [_uncertain_record("a.jpg", "p1"),
                _uncertain_record("a.jpg", "p2", cls_name="Mango"),
                _uncertain_record("v.jpg", "p3")]
        st = _state(root, recs,
                    proposals=[_proposal("a.jpg", "p1"),
                               _proposal("a.jpg", "p2"),
                               _proposal("v.jpg", "p3")])
        q = resolver.build_review_queue(st)
        unc = [it for it in q["items"] if it["category"] == resolver.CAT_UNCERTAIN]
        assert len(unc) == 2
        aq = next(it for it in unc if it["image_filename"] == "a.jpg")
        assert aq["proposal_count"] == 2
        assert aq["gt_count"] >= 1
        assert len(aq["proposal_ids"]) == 2


# --------------------------------------------------------------------------- #
# 2. Unresolved blocker detection
# --------------------------------------------------------------------------- #
class TestBlockers:
    def test_collect_adjudication_blockers(self, tmp_path):
        root = _build_v2(tmp_path)
        recs = [_empty_label_image_record(),
                _hugebox_image_record("huge.jpg", "manual_review"),
                _hugebox_image_record("a.jpg", "tighten"),
                _hugebox_image_record("v.jpg", "keep")]
        blocks = resolver.collect_adjudication_blockers(recs)
        blocked = {r["image_filename"] for r in blocks}
        assert blocked == {"huge.jpg", "a.jpg"}

    def test_collect_uncertain_proposals(self, tmp_path):
        root = _build_v2(tmp_path)
        acc = _uncertain_record("a.jpg", "p1")
        acc["human_decision"] = "accepted"
        unc = _uncertain_record("v.jpg", "p2")
        kept = _uncertain_record("t.jpg", "p3")
        kept["human_decision"] = "kept"
        recs = [acc, unc, kept]
        st = _state(root, recs)
        got = resolver.collect_uncertain_proposals(recs, st["proposals"])
        assert len(got) == 1
        assert got[0]["image_filename"] == "v.jpg"

    def test_gate_blocked_when_unresolved(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_uncertain_record("a.jpg", "p1")],
                    proposals=[_proposal("a.jpg", "p1")])
        v = resolver.validate_resolved_state(st)
        assert not v["passed"]

# --------------------------------------------------------------------------- #
# 3. Empty-label handling
# --------------------------------------------------------------------------- #
class TestEmptyLabel:
    def test_manually_annotate_supplies_validated_bbox(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_empty_label_image_record()])
        res = {"items": [
            {"category": resolver.CAT_EMPTY, "image_filename": "empty.jpg",
             "action": "MANUALLY_ANNOTATE",
             "coordinates": [0, 0.5, 0.5, 0.3, 0.3], "notes": "human box"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        st2 = copy.deepcopy(st)
        st2["decisions"]["records"] = result["proposed_records"]
        assert resolver.validate_resolved_state(st2)["passed"]

    def test_confirm_background(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_empty_label_image_record()])
        res = {"items": [
            {"category": resolver.CAT_EMPTY, "image_filename": "empty.jpg",
             "action": "CONFIRM_BACKGROUND", "notes": "no fruit"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        rec = next(r for r in result["proposed_records"]
                   if r.get("image_filename") == "empty.jpg")
        assert rec["decision"] == "keep_empty"
        assert rec["bbox"] is None

    def test_mark_uncertain_stays_blocked(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_empty_label_image_record()])
        res = {"items": [
            {"category": resolver.CAT_EMPTY, "image_filename": "empty.jpg",
             "action": "MARK_UNCERTAIN"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        assert result["applied"] == 0
        assert result["skipped"] == 1

    def test_rejects_invalid_coordinates(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_empty_label_image_record()])
        res = {"items": [
            {"category": resolver.CAT_EMPTY, "image_filename": "empty.jpg",
             "action": "MANUALLY_ANNOTATE",
             "coordinates": [99, 0.5, 0.5, 0.3, 0.3], "notes": "bad class"}]}
        with pytest.raises(ValueError):
            resolver.apply_resolutions(st, res, dry_run=True)


# --------------------------------------------------------------------------- #
# 4. Manual-review / tighten handling
# --------------------------------------------------------------------------- #
class TestHugeBox:
    def test_manual_review_keep(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_hugebox_image_record("huge.jpg", "manual_review")])
        res = {"items": [
            {"category": resolver.CAT_MANUAL, "image_filename": "huge.jpg",
             "action": "KEEP", "notes": "object fills frame"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        rec = next(r for r in result["proposed_records"]
                   if r.get("image_filename") == "huge.jpg")
        assert rec["decision"] == "keep"
        st2 = copy.deepcopy(st)
        st2["decisions"]["records"] = result["proposed_records"]
        assert resolver.validate_resolved_state(st2)["passed"]

    def test_tighten_requires_validated_box(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_hugebox_image_record("a.jpg", "tighten")])
        res = {"items": [
            {"category": resolver.CAT_TIGHTEN, "image_filename": "a.jpg",
             "action": "TIGHTEN",
             "coordinates": [0, 0.5, 0.5, 0.4, 0.4], "notes": "tighter box"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        rec = next(r for r in result["proposed_records"]
                   if r.get("image_filename") == "a.jpg")
        assert rec["decision"] == "tighten"
        assert rec["bbox"] == [0.0, 0.5, 0.5, 0.4, 0.4]

    def test_tighten_rejects_missing_box(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_hugebox_image_record("a.jpg", "tighten")])
        res = {"items": [
            {"category": resolver.CAT_TIGHTEN, "image_filename": "a.jpg",
             "action": "TIGHTEN", "notes": "no coords"}]}
        with pytest.raises(ValueError):
            resolver.apply_resolutions(st, res, dry_run=True)

# --------------------------------------------------------------------------- #
# 5. Uncertain-proposal handling
# --------------------------------------------------------------------------- #
class TestUncertain:
    def test_accept_all_explicit(self, tmp_path):
        root = _build_v2(tmp_path)
        recs = [_uncertain_record("a.jpg", "p1"),
                _uncertain_record("a.jpg", "p2", cls_name="Mango")]
        st = _state(root, recs,
                    proposals=[_proposal("a.jpg", "p1"),
                               _proposal("a.jpg", "p2")])
        res = {"items": [
            {"category": resolver.CAT_UNCERTAIN, "image_filename": "a.jpg",
             "action": "ACCEPT_ALL", "notes": "explicit bulk accept"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        acc = [r for r in result["proposed_records"]
               if isinstance(r.get("human_decision"), str) and
               r.get("image_filename") == Path("a.jpg").name]
        assert len(acc) == 2
        assert all(r["human_decision"] == "accepted" for r in acc)

    def test_reject_selected(self, tmp_path):
        root = _build_v2(tmp_path)
        recs = [_uncertain_record("a.jpg", "p1"),
                _uncertain_record("a.jpg", "p2", cls_name="Mango")]
        st = _state(root, recs,
                    proposals=[_proposal("a.jpg", "p1"),
                               _proposal("a.jpg", "p2")])
        res = {"items": [
            {"category": resolver.CAT_UNCERTAIN, "image_filename": "a.jpg",
             "action": "REJECT_SELECTED", "proposal_ids": ["p1"]}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        rejected = [r for r in result["proposed_records"]
                    if isinstance(r.get("human_decision"), str) and
                    r["human_decision"] == "rejected"]
        assert len(rejected) == 1
        assert rejected[0]["ai_proposal"]["proposal_id"] == "p1"

    def test_keep_original_respects_proposals(self, tmp_path):
        root = _build_v2(tmp_path)
        recs = [_uncertain_record("a.jpg", "p1"),
                _uncertain_record("a.jpg", "p2", cls_name="Mango")]
        st = _state(root, recs,
                    proposals=[_proposal("a.jpg", "p1"),
                               _proposal("a.jpg", "p2")])
        res = {"items": [
            {"category": resolver.CAT_UNCERTAIN, "image_filename": "a.jpg",
             "action": "KEEP_ORIGINAL", "notes": "keep GT"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        kept = [r for r in result["proposed_records"]
                if isinstance(r.get("human_decision"), str) and
                r["human_decision"] == "kept"]
        assert len(kept) == 2

    def test_uncertain_stays_blocked(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_uncertain_record("a.jpg", "p1")],
                    proposals=[_proposal("a.jpg", "p1")])
        res = {"items": [
            {"category": resolver.CAT_UNCERTAIN, "image_filename": "a.jpg",
             "action": "UNCERTAIN"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        assert result["applied"] == 0
        assert result["skipped"] == 1


# --------------------------------------------------------------------------- #
# 6/7. Coordinate & class validation
# --------------------------------------------------------------------------- #
class TestCoordinateClassValidation:
    def test_valid_yolo_row_accepted(self):
        row = resolver._validated_yolo_row([0, 0.5, 0.5, 0.3, 0.3], NC)
        assert row[0] == 0.0

    def test_invalid_class_rejected(self):
        with pytest.raises(ValueError):
            resolver._validated_yolo_row([99, 0.5, 0.5, 0.3, 0.3], NC)

    def test_out_of_frame_rejected(self):
        with pytest.raises(ValueError):
            resolver._validated_yolo_row([0, 0.9, 0.5, 0.3, 0.3], NC)

    def test_zero_area_rejected(self):
        with pytest.raises(ValueError):
            resolver._validated_yolo_row([0, 0.5, 0.5, 0.0, 0.3], NC)

    def test_malformed_row_rejected(self):
        with pytest.raises(ValueError):
            resolver._validated_yolo_row([0, 0.5, 0.5, 0.3], NC)  # 4 fields


# --------------------------------------------------------------------------- #
# 8. Grape policy
# --------------------------------------------------------------------------- #
class TestGrapePolicy:
    def test_grape_queue_has_warning(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_hugebox_image_record("huge.jpg", "manual_review")])
        q = resolver.build_review_queue(st)
        it = next(i for i in q["items"] if i["image_filename"] == "huge.jpg")
        assert it.get("grape_warning") is True

    def test_per_berry_grape_box_flagged(self, tmp_path):
        root = _build_v2(tmp_path)
        rec = _hugebox_image_record("a.jpg", "tighten")
        rec["class_name"] = "Grape"
        rec["bbox"] = [1, 0.01, 0.01, 0.01, 0.01]
        st = _state(root, [rec])
        v = resolver.validate_resolved_state(st)
        assert any("Grape" in e and "implausibly small" in e for e in v["errors"])
# --------------------------------------------------------------------------- #
# 9/10/11. Backup + atomic persistence + duplicates
# --------------------------------------------------------------------------- #
class TestPersistence:
    def test_backup_created_before_write(self, tmp_path):
        src = tmp_path / "backup_src.json"
        src.write_text('{"a": 1}', encoding="utf-8")
        bak_out = tmp_path / "out"
        bak_out.mkdir()
        backup = resolver.backup_decisions(src, out_dir=bak_out,
                                           timestamp="20200101_000000")
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == '{"a": 1}'

    def test_atomic_write(self, tmp_path):
        target = tmp_path / "decisions.json"
        original = {"records": [{"decision": "keep"}]}
        resolver.write_json_atomic(original, target)
        assert json.loads(target.read_text(encoding="utf-8")) == original
        new = {"records": [{"decision": "annotate",
                            "bbox": [0, 0.5, 0.5, 0.2, 0.2]}]}
        resolver.write_json_atomic(new, target)
        assert json.loads(target.read_text(encoding="utf-8"))["records"][0] \
            ["decision"] == "annotate"
        left_over = list(tmp_path.glob(".decisions.tmp_*"))
        assert left_over == []  # temp file cleaned up

    def test_duplicate_prevention_validation(self, tmp_path):
        root = _build_v2(tmp_path)
        r1 = _uncertain_record("a.jpg", "p1")
        r2 = copy.deepcopy(r1)
        st = _state(root, [r1, r2], proposals=[_proposal("a.jpg", "p1")])
        v = resolver.validate_resolved_state(st)
        assert any("duplicate" in e for e in v["errors"])

    def test_backup_files_distinct(self, tmp_path):
        src = tmp_path / "back_src.json"
        src.write_text("{}", encoding="utf-8")
        bak_out = tmp_path / "baks"
        bak_out.mkdir()
        paths = [resolver.backup_decisions(src, out_dir=bak_out,
                                           timestamp=f"2020010{i}")
                 for i in range(3)]
        assert len({p.name for p in paths}) == 3


# --------------------------------------------------------------------------- #
# 12/13/14/15/16/17. Immutability + dry-run no writes
# --------------------------------------------------------------------------- #
class TestImmutability:
    def test_dry_run_makes_no_changes(self, tmp_path):
        root = _build_v2(tmp_path)
        v2_files = {}
        for d in (root / "train", root / "valid", root / "test"):
            for p in d.rglob("*"):
                if p.is_file():
                    v2_files[str(p)] = p.read_bytes()
        st = _state(root, [_empty_label_image_record()])
        res = {"items": [
            {"category": resolver.CAT_EMPTY, "image_filename": "empty.jpg",
             "action": "MANUALLY_ANNOTATE",
             "coordinates": [0, 0.5, 0.5, 0.3, 0.3]}]}
        resolver.apply_resolutions(st, res, dry_run=True)
        for d in (root / "train", root / "valid", root / "test"):
            for p in d.rglob("*"):
                if p.is_file():
                    assert v2_files[str(p)] == p.read_bytes()
        # No backup / tmp artifacts created under the tree.
        assert list(tmp_path.rglob("*.backup_*")) == []

    def test_no_v3_created(self, tmp_path):
        root = _build_v2(tmp_path)
        st = _state(root, [_empty_label_image_record()])
        res = {"items": [
            {"category": resolver.CAT_EMPTY, "image_filename": "empty.jpg",
             "action": "MANUALLY_ANNOTATE",
             "coordinates": [0, 0.5, 0.5, 0.3, 0.3]}]}
        resolver.apply_resolutions(st, res, dry_run=True)
        # resolver never creates a V3 directory.
        assert list(root.rglob("detection_v3")) == []
        assert list(tmp_path.rglob("v3_manifest.json")) == []

    def test_best_pt_untouched(self, tmp_path):
        root = _build_v2(tmp_path)
        # Place a fake best.pt and capture its bytes.
        fake_best = tmp_path / "best.pt"
        fake_best.write_bytes(b"fake-weights")
        before = fake_best.read_bytes()
        st = _state(root, [_empty_label_image_record()])
        res = {"items": [
            {"category": resolver.CAT_EMPTY, "image_filename": "empty.jpg",
             "action": "CONFIRM_BACKGROUND"}]}
        resolver.apply_resolutions(st, res, dry_run=True)
        assert fake_best.read_bytes() == before
# --------------------------------------------------------------------------- #
# End-to-end apply with atomic write to a temp decisions file
# --------------------------------------------------------------------------- #
class TestEndToEnd:
    def test_full_resolution_atomic_write(self, tmp_path):
        root = _build_v2(tmp_path)
        records = [
            _empty_label_image_record(),
            _hugebox_image_record("huge.jpg", "manual_review"),
            _hugebox_image_record("a.jpg", "tighten"),
            _uncertain_record("v.jpg", "p1"),
        ]
        res = {"items": [
            {"category": resolver.CAT_EMPTY, "image_filename": "empty.jpg",
             "action": "CONFIRM_BACKGROUND", "notes": "no fruit"},
            {"category": resolver.CAT_MANUAL, "image_filename": "huge.jpg",
             "action": "KEEP", "notes": "fills frame"},
            {"category": resolver.CAT_TIGHTEN, "image_filename": "a.jpg",
             "action": "TIGHTEN", "coordinates": [1, 0.5, 0.5, 0.4, 0.4]},
            {"category": resolver.CAT_UNCERTAIN, "image_filename": "v.jpg",
             "action": "ACCEPT_ALL", "notes": "explicit"},
        ]}
        st = _state(root, records,
                    proposals=[_proposal("v.jpg", "p1")])
        dec_path = tmp_path / "human_decisions.json"
        result = resolver.apply_resolutions(st, res,
                                            decisions_path=dec_path,
                                            dry_run=False)
        # Backup was created next to the decision file.
        backups = list(dec_path.parent.glob("human_decisions.backup_*.json"))
        assert len(backups) == 1
        # Atomic file written and parseable.
        persisted = load_human_decisions(dec_path)
        assert persisted["record_count"] == sum(1 for _ in persisted["records"])
        # Resolved records carry human provenance.
        for r in persisted["records"]:
            if r.get("resolved_at"):
                assert r.get("reviewer") == "human"

    def test_validate_passes_after_resolution(self, tmp_path):
        root = _build_v2(tmp_path)
        recs = [_uncertain_record("a.jpg", "p1"),
                _uncertain_record("a.jpg", "p2", cls_name="Mango")]
        st = _state(root, recs,
                    proposals=[_proposal("a.jpg", "p1"),
                               _proposal("a.jpg", "p2")])
        res = {"items": [
            {"category": resolver.CAT_UNCERTAIN, "image_filename": "a.jpg",
             "action": "ACCEPT_ALL", "notes": "explicit"}]}
        result = resolver.apply_resolutions(st, res, dry_run=True)
        st2 = copy.deepcopy(st)
        st2["decisions"]["records"] = result["proposed_records"]
        v = resolver.validate_resolved_state(st2)
        assert v["passed"]


def test_dry_run_status_printer(tmp_path, capsys):
    st = _state(tmp_path, [_uncertain_record("a.jpg", "p1")],
                proposals=[_proposal("a.jpg", "p1")])
    rc = resolver.print_status(st)
    assert rc == 3
    out = capsys.readouterr().out
    assert "V3 GATE: BLOCKED" in out
    assert "Total blocker images" in out
