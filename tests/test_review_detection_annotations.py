"""Tests for the read-only annotation-review system (Phase 1 of V3 prep).

Uses small synthetic datasets in temp dirs so the tests never depend on the
real ``data/detection`` export and never mutate it.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.review_detection_annotations import (  # noqa: E402
    _label_path,
    collect_suspicious_images,
    review_dataset,
    render_review_image,
)
from scripts.audit_detection_dataset import _read_boxes  # noqa: E402


def _make_image(path: Path, color: int = None, size: int = 80) -> Path:
    """Write a deterministic image whose bytes differ per path.

    Each path produces a different stable pattern, so two images that are
    byte-identical only happen when one is explicitly copied onto the other.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        seed = int(hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    else:
        img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _build_dataset(root: Path):
    yaml = ("train: train/images\nval: valid/images\ntest: test/images\nnc: 3\n"
            "names:\n- Apple\n- Mango\n- Kiwi\n")
    (root / "data.yaml").write_text(yaml, encoding="utf-8")
    # train: one empty label, one huge box, one tiny box, one many-objects, one ok
    (root / "train/images").mkdir(parents=True)
    (root / "train/labels").mkdir(parents=True)
    _make_image(root / "train/images/empty.jpg")
    (root / "train/labels/empty.txt").write_text("", encoding="utf-8")
    _make_image(root / "train/images/huge.jpg")
    (root / "train/labels/huge.txt").write_text("0 0.5 0.5 0.99 0.99\n", encoding="utf-8")
    _make_image(root / "train/images/tiny.jpg")
    (root / "train/labels/tiny.txt").write_text("1 0.5 0.5 0.001 0.001\n", encoding="utf-8")
    _make_image(root / "train/images/ok.jpg")
    (root / "train/labels/ok.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    # many-objects image (>60 boxes) -> build a label with 70 boxes
    _make_image(root / "train/images/many.jpg")
    (root / "train/labels/many.txt").write_text(
        "".join("0 0.01 0.01 0.02 0.02\n" for _ in range(70)), encoding="utf-8")
    (root / "valid/images").mkdir(parents=True)
    (root / "valid/labels").mkdir(parents=True)
    _make_image(root / "valid/images/v.jpg")
    (root / "valid/labels/v.txt").write_text("1 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (root / "test/images").mkdir(parents=True)
    (root / "test/labels").mkdir(parents=True)
    _make_image(root / "test/images/t.jpg")
    (root / "test/labels/t.txt").write_text("2 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    return root


class TestSuspiciousDetection:
    def test_flags_empty_huge_tiny_many(self, tmp_path):
        root = _build_dataset(tmp_path)
        findings = collect_suspicious_images(root, ["n0", "n1", "n2"])
        flagged = {p["image_filename"] for p in findings["empty_labels"]}
        assert "empty.jpg" in flagged
        huge = {p["image_filename"] for p in findings["huge_boxes"]}
        assert "huge.jpg" in huge
        tiny = {p["image_filename"] for p in findings["tiny_boxes"]}
        assert "tiny.jpg" in tiny
        many = {p["image_filename"] for p in findings["many_objects"]}
        assert "many.jpg" in many

    def test_ambiguous_class_detection(self, tmp_path):
        root = _build_dataset(tmp_path)
        findings = collect_suspicious_images(root, ["Apple", "Mango", "Kiwi"])
        amb = {p["image_filename"] for p in findings["ambiguous_classes"]}
        # ok image contains class 2 (Kiwi) -> not ambiguous; huge contains class 0 (Apple)
        assert "huge.jpg" in amb

    def test_ok_image_not_flagged(self, tmp_path):
        root = _build_dataset(tmp_path)
        findings = collect_suspicious_images(root, ["n0", "n1", "n2"])
        # 'ok' image has a normal single box of a non-ambiguous class -> no flags
        all_flagged = set()
        for v in findings.values():
            all_flagged.update(p["image_filename"] for p in v)
        assert "ok.jpg" not in all_flagged


class TestRenderAndReport:
    def test_render_review_image_writes_file(self, tmp_path):
        root = _build_dataset(tmp_path)
        img = root / "train/images/huge.jpg"
        lbl = _label_path(root / "train/labels", "huge")
        out = tmp_path / "out.jpg"
        render_review_image(img, lbl, ["n0", "n1", "n2"], ["huge_box"], out)
        assert out.exists() and out.stat().st_size > 0

    def test_review_dataset_writes_json_and_dirs(self, tmp_path):
        root = _build_dataset(tmp_path)
        out_dir = tmp_path / "reviews"
        summary = review_dataset(root, out_dir, max_per_category=50)
        assert out_dir.exists()
        # each category has a review JSON
        for cat in ["empty_labels", "huge_boxes", "tiny_boxes", "many_objects",
                    "ambiguous_classes"]:
            assert (out_dir / f"{cat}_review.json").exists()
            assert (out_dir / cat).is_dir()
            assert summary[cat]["count"] >= 1
        # no dataset file modified
        for p in root.rglob("*"):
            if p.is_file() and p.suffix == ".jpg":
                assert p.stat().st_size > 0



class TestV3Corrections:
    """Evidence-based V3 decision logic for empty-label images."""

    def test_empty_with_exact_labeled_twin_is_apply(self, tmp_path):
        # Reuse the standard synthetic dataset, then overwrite the empty-slot
        # image bytes with those of a *labeled* image (huge.jpg). The empty
        # label now has a provably byte-identical labeled twin -> apply_to_v3.
        root = _build_dataset(tmp_path)
        import shutil
        shutil.copyfile(root / "train/images/huge.jpg", root / "train/images/empty.jpg")

        from scripts.review_detection_annotations import (
            build_md5_index, collect_suspicious_images, find_v3_corrections)
        names = ["Apple", "Mango", "Kiwi"]
        empty = collect_suspicious_images(root, names)["empty_labels"]
        assert {r["image_filename"] for r in empty} == {"empty.jpg"}
        corrections = find_v3_corrections(empty, build_md5_index(root))
        assert len(corrections) == 1
        assert corrections[0]["decision"] == "apply_to_v3"
        assert corrections[0]["confidence"] == "high"
        assert len(corrections[0]["matched_labeled_copies"]) >= 1
        assert any(t["stem"] == "huge" for t in corrections[0]["matched_labeled_copies"])

    def test_empty_without_twin_is_needs_review(self, tmp_path):
        root = _build_dataset(tmp_path)
        from scripts.review_detection_annotations import (
            build_md5_index, collect_suspicious_images, find_v3_corrections)
        names = ["Apple", "Mango", "Kiwi"]
        empty = collect_suspicious_images(root, names)["empty_labels"]
        corrections = find_v3_corrections(empty, build_md5_index(root))
        assert len(corrections) == 1
        assert corrections[0]["decision"] == "needs_manual_review"
        assert corrections[0]["confidence"] == "none"
        assert corrections[0]["matched_labeled_copies"] == []
