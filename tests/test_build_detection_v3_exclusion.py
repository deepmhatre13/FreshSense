"""Tests for the V3 exclusion (V2-copy) builder."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from unittest import mock
import cv2, numpy as np, pytest, yaml

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.build_detection_v3 import (
    EXPECTED_CLASS_NAMES, _compute_exclusion, exclusion_build, load_blocker_queue,
    verify_v3_build,
)

NAMES = list(EXPECTED_CLASS_NAMES)


def _img(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    return path


def _build_v2(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    y = ["train: train/images", "val: valid/images", "test: test/images", "nc: 10", "names:"]
    y += [f"- {n}" for n in NAMES]
    (root / "data.yaml").write_text("\n".join(y) + "\n", encoding="utf-8")
    for sp in ("train", "valid", "test"):
        (root / sp / "images").mkdir(parents=True)
        (root / sp / "labels").mkdir(parents=True)
    files = {"train": ["t0", "t1", "t2"], "valid": ["v0", "v1", "v2"], "test": ["e0", "e1", "e2"]}
    for sp, stems in files.items():
        for cls, stem in enumerate(stems):
            _img(root / sp / "images" / f"{stem}.jpg")
            (root / sp / "labels" / f"{stem}.txt").write_text(f"{cls % 10} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return root


def _queue(path: Path, blockers) -> None:
    items = [{"image_filename": fn, "split": sp, "category": "A_EMPTY_LABEL", "current_decision": "annotate"} for sp, fn in blockers]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"counts": {"total_blocker_images": len(blockers)}, "items": items}), encoding="utf-8")


@pytest.fixture
def exclude_queue(tmp_path):
    q = tmp_path / "queue.json"
    blockers = [("train", "t0.jpg"), ("valid", "v0.jpg"), ("valid", "v1.jpg")]
    _queue(q, blockers)
    with mock.patch("scripts.build_detection_v3.DEFAULT_QUEUE_FILE", q):
        yield q, blockers

class TestHelpers:
    def test_queue_filters_test(self, tmp_path):
        q = tmp_path / "q.json"
        _queue(q, [("train", "a.jpg"), ("test", "b.jpg")])
        with mock.patch("scripts.build_detection_v3.DEFAULT_QUEUE_FILE", q):
            blockers = load_blocker_queue()
        assert [b["image_filename"] for b in blockers] == ["a.jpg"]


class TestExclusionBuild:
    @pytest.fixture(autouse=True)
    def _isolate_reports(self, tmp_path, monkeypatch):
        """Redirect _REPO_ROOT to a temp dir so exclusion_build never mutates
        the real repo reports (reports/detection_v3_exclusion_report*)."""
        monkeypatch.setattr("scripts.build_detection_v3._REPO_ROOT", tmp_path)

    def test_dry_run_writes_nothing(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        out = tmp_path / "v3"
        report = exclusion_build(root, out, dry_run=True)
        assert report["dry_run"] is True
        assert report["excluded_count"] == 3
        assert report["excluded_by_split"] == {"train": 1, "valid": 2, "test": 0}
        assert not out.exists()

    def test_build_excludes_right_splits(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        out = tmp_path / "v3"
        exclusion_build(root, out)
        assert (out / "data.yaml").is_file()
        assert not (out / "train/images/t0.jpg").exists()
        assert not (out / "train/labels/t0.txt").exists()
        assert not (out / "valid/images/v0.jpg").exists()
        assert not (out / "valid/images/v1.jpg").exists()
        assert (out / "train/images/t1.jpg").exists()
        assert (out / "valid/images/v2.jpg").exists()
        assert len(list((out / "train/images").glob("*.jpg"))) == 2
        assert len(list((out / "valid/images").glob("*.jpg"))) == 1
        assert len(list((out / "test/images").glob("*.jpg"))) == 3
        assert len(list((out / "train/labels").glob("*.txt"))) == 2

    def test_does_not_modify_v2(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        before = {str(p): p.read_bytes() for p in root.rglob("*.jpg")}
        before_lbl = {str(p): p.read_bytes() for p in root.rglob("*.txt")}
        out = tmp_path / "v3"
        exclusion_build(root, out)
        after = {str(p): p.read_bytes() for p in root.rglob("*.jpg")}
        after_lbl = {str(p): p.read_bytes() for p in root.rglob("*.txt")}
        assert before == after
        assert before_lbl == after_lbl

    def test_test_split_byte_identical(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        out = tmp_path / "v3"
        exclusion_build(root, out)
        for n in ("e0", "e1", "e2"):
            assert (root / "test/images" / f"{n}.jpg").read_bytes() == \
                (out / "test/images" / f"{n}.jpg").read_bytes()
            assert (root / "test/labels" / f"{n}.txt").read_bytes() == \
                (out / "test/labels" / f"{n}.txt").read_bytes()

    def test_class_mapping_unchanged(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        out = tmp_path / "v3"
        exclusion_build(root, out)
        cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        assert cfg["nc"] == 10
        assert cfg["names"] == NAMES
        assert cfg["train"] == "train/images"
        assert cfg["val"] == "valid/images"
        assert cfg["test"] == "test/images"

    def test_deterministic(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        o1, o2 = tmp_path / "v3a", tmp_path / "v3b"
        exclusion_build(root, o1)
        exclusion_build(root, o2)
        h1 = {str(p.relative_to(o1)): p.read_bytes() for p in o1.rglob("*") if p.is_file()}
        h2 = {str(p.relative_to(o2)): p.read_bytes() for p in o2.rglob("*") if p.is_file()}
        assert h1 == h2

    def test_refuses_without_force_when_exists(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        out = tmp_path / "v3"
        exclusion_build(root, out)
        with pytest.raises(RuntimeError):
            exclusion_build(root, out)
        exclusion_build(root, out, force=True)

    def test_duplicate_blocker_refused(self, tmp_path):
        q = tmp_path / "queue.json"
        _queue(q, [("train", "dup.jpg"), ("train", "dup.jpg")])
        with mock.patch("scripts.build_detection_v3.DEFAULT_QUEUE_FILE", q):
            root = _build_v2(tmp_path / "v2")
            with pytest.raises(RuntimeError):
                _compute_exclusion(root, NAMES)

    def test_blocker_in_test_ignored(self, tmp_path):
        q = tmp_path / "queue.json"
        _queue(q, [("test", "e0.jpg")])
        with mock.patch("scripts.build_detection_v3.DEFAULT_QUEUE_FILE", q):
            root = _build_v2(tmp_path / "v2")
            out = tmp_path / "v3"
            exclusion_build(root, out)
            assert (out / "test/images/e0.jpg").exists()

    def test_verify_passes(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        out = tmp_path / "v3"
        exclusion_build(root, out)
        plan = _compute_exclusion(root, NAMES)
        built = verify_v3_build(root, out, plan)
        assert built["verification"]["passed"] is True
        assert built["destination_counts"] == {"train": 2, "valid": 1, "test": 3}

    def test_verify_detects_missing_image(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        out = tmp_path / "v3"
        exclusion_build(root, out)
        (out / "test/images/e0.jpg").unlink()
        plan = _compute_exclusion(root, NAMES)
        built = verify_v3_build(root, out, plan)
        assert built["verification"]["passed"] is False

    def test_report_required_fields(self, tmp_path, exclude_queue):
        root = _build_v2(tmp_path / "v2")
        out = tmp_path / "v3"
        report = exclusion_build(root, out)
        for key in ("source_dataset", "destination_dataset", "timestamp",
                    "excluded_count", "excluded_by_split", "excluded_files",
                    "reason", "source_test_count", "destination_test_count",
                    "test_set_unchanged", "destination_counts"):
            assert key in report
        assert report["test_set_unchanged"] is True


class TestRealQueueSafety:
    def test_no_test_blockers_in_authoritative_queue(self):
        p = Path("reports/audit_review/v3_human_review_queue.json")
        if not p.exists():
            pytest.skip("real review queue not present")
        items = json.loads(p.read_text(encoding="utf-8"))["items"]
        test_blocks = [i for i in items if i.get("split") == "test"]
        assert test_blocks == [], "No blockers may ever be in the test split"
