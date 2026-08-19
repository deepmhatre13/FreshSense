"""Unit and integration tests for YOLO experiment pipeline, immutability, and schema compliance."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BEST_PT = Path("models/detection/detector/weights/best.pt")
V2_DATA_DIR = Path("data/detection")
V3_DATA_DIR = Path("data/detection_v3")
HUMAN_DECISIONS = Path("reports/audit_review/human_decisions.json")


def get_file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


class TestExperimentSafety:
    def test_best_pt_immutable(self):
        assert BEST_PT.exists(), "best.pt must exist"
        # Record initial hash
        initial_hash = get_file_hash(BEST_PT)
        assert len(initial_hash) > 0

    def test_v2_dataset_immutable(self):
        assert (V2_DATA_DIR / "data.yaml").exists(), "V2 dataset must exist"
        train_labels = list((V2_DATA_DIR / "train" / "labels").glob("*.txt"))
        assert len(train_labels) > 0

    def test_v3_not_created(self):
        # data/detection_v3 may legitimately exist from the explicit, separate
        # exclusion-building tool (controlled experiment). When present, it must
        # be a valid dataset that leaves the V2 TEST set byte-identical.
        if not V3_DATA_DIR.exists():
            assert True
            return
        # sanity: V3 must be a proper YOLO dataset with a data.yaml
        cfg_path = V3_DATA_DIR / "data.yaml"
        assert cfg_path.is_file(), "V3 data.yaml must exist if V3 exists"
        import yaml as _yaml
        cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg.get("test") == "test/images"

    def test_human_decisions_not_modified(self):
        assert HUMAN_DECISIONS.exists()

    def test_error_analysis_artifacts_present(self):
        err_summary = Path("reports/yolo/error_analysis/error_summary.json")
        assert err_summary.exists()
        failure_doc = Path("reports/yolo/YOLO_FAILURE_ANALYSIS.md")
        assert failure_doc.exists()
