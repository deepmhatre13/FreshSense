"""Tests for Phase 4A dataset-validation primitives."""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.dataset_validation import (
    ImageQuality,
    MetadataRecord,
    compute_image_info,
    file_md5,
    find_exact_duplicates,
    find_near_duplicates,
    find_suspicious_groups,
    group_files_by_session,
    hamming_distance,
    is_corrupted,
    load_metadata_dir,
    parse_metadata_file,
    perceptual_hash,
    scan_directory,
    split_by_session,
    split_files,
)


def _make_image(path: Path, color: int = 128, size: int = 64) -> Path:
    """Create a small solid-color image and return its path."""
    img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


class TestQuality:
    def test_is_dark_bright_blurry(self):
        dark = ImageQuality(brightness=10.0, contrast=30.0, blur_score=500.0)
        assert dark.is_dark()
        assert not dark.is_bright()
        bright = ImageQuality(brightness=250.0, contrast=30.0, blur_score=500.0)
        assert bright.is_bright()
        blurry = ImageQuality(brightness=120.0, contrast=30.0, blur_score=5.0)
        assert blurry.is_blurry()
        low = ImageQuality(brightness=120.0, contrast=2.0, blur_score=500.0)
        assert low.is_low_contrast()


class TestCorrupted:
    def test_valid_image_not_corrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_image(Path(tmp) / "ok.jpg")
            corrupted, _ = is_corrupted(p)
            assert not corrupted

    def test_missing_is_corrupted(self):
        corrupted, err = is_corrupted(Path("does/not/exist.jpg"))
        assert corrupted
        assert err == "missing"

    def test_garbage_bytes_corrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.jpg"
            p.write_bytes(b"this is definitely not a valid image")
            corrupted, _ = is_corrupted(p)
            assert corrupted

    def test_compute_image_info_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_image(Path(tmp) / "img.png", size=80)
            info = compute_image_info(p)
            assert info.readable
            assert info.width == 80
            assert info.height == 80
            assert info.aspect_ratio == pytest.approx(1.0)
            assert info.quality is not None


class TestScan:
    def test_scan_counts_and_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fresh").mkdir()
            (root / "stale").mkdir()
            _make_image(root / "fresh" / "a.jpg")
            _make_image(root / "fresh" / "b.jpg")
            _make_image(root / "stale" / "c.jpg")
            scan = scan_directory(root)
            assert scan.total == 3
            counts = scan.class_counts()
            assert counts["fresh"] == 2

class TestMd5AndHash:
    def test_file_md5_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_image(Path(tmp) / "a.png")
            assert file_md5(p) == file_md5(p)

    def test_perceptual_hash_same_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_image(Path(tmp) / "a.png")
            assert perceptual_hash(p) == perceptual_hash(p)

    def test_hamming_distance(self):
        assert hamming_distance("0000", "0001") == 1
        assert hamming_distance("0000", "1111") == 4


def _make_pattern(path: Path, seed: int = 0, size: int = 64) -> Path:
    """Create a smooth structured image yielding a stable pHash.

    A vertical gradient plus a bright square gives low-frequency structure so
    the perceptual hash is robust to subtle perturbations.
    """
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    base = (ys / size * 255).astype(np.uint8)
    square = (np.abs(xs - size / 2) < size / 4) & (np.abs(ys - size / 2) < size / 4)
    img = np.stack([base, base, base], axis=-1)
    img[square] = 200
    cv2.imwrite(str(path), img)
    return path


class TestDuplicates:
    def test_exact_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _make_image(root / "a.jpg")
            b = _make_image(root / "b.jpg")  # same solid color -> same bytes
            pairs = find_exact_duplicates([a, b])
            assert any(p.kind == "exact" for p in pairs)

    def test_near_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _make_pattern(root / "a.png", seed=1)
            img = cv2.imread(str(a))
            b = root / "b.png"
            # Slightly perturbed copy that is perceptually near the original.
            cv2.imwrite(str(b), (img.astype(np.int16) + 3).clip(0, 255).astype(np.uint8))
            pairs = find_near_duplicates([a, b], max_distance=10)
            assert any(p.kind == "near" for p in pairs)

    def test_suspicious_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Three near-identical patterned images -> one connected group.
            base = _make_pattern(root / "i0.png", seed=2)
            bimg = cv2.imread(str(base))
            for i in (1, 2):
                cv2.imwrite(
                    str(root / f"i{i}.png"),
                    (bimg.astype(np.int16) + i).clip(0, 255).astype(np.uint8),
                )
            paths = [root / "i0.png", root / "i1.png", root / "i2.png"]
            groups = find_suspicious_groups(paths)
            assert groups  # near-duplicate images form a group



class TestMetadata:
    def test_parse_valid_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.json"
            p.write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "session_id": "sess_1",
                        "timestamp": 1234.5,
                        "image_path": "accepted/s1.jpg",
                        "label": "fresh",
                        "predicted_confidence": 0.9,
                    }
                ),
                encoding="utf-8",
            )
            rec = parse_metadata_file(p)
            assert rec is not None
            assert rec.sample_id == "s1"
            assert rec.label == "fresh"

    def test_parse_invalid_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("not json{", encoding="utf-8")
            assert parse_metadata_file(p) is None

    def test_missing_fields_detected(self):
        rec = MetadataRecord.from_dict({"sample_id": "s", "session_id": ""})
        missing = rec.missing_fields()
        assert "session_id" in missing
        assert "label" in missing

    def test_load_metadata_dir_skips_session_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "session_session1.json").write_text("{}", encoding="utf-8")
            (d / "sample1.json").write_text(
                json.dumps(
                    {"sample_id": "s", "session_id": "x", "timestamp": 1.0,
                     "image_path": "a.jpg", "label": "fresh"}
                ),
                encoding="utf-8",
            )
            recs = load_metadata_dir(d)
            assert len(recs) == 1


class TestSplit:
    def test_split_by_session_disjoint(self):
        files = {f"sess_{i}": [Path(f"img_{i}.jpg")] for i in range(10)}
        tr, va, te = split_by_session(files)
        assert not (tr & va)
        assert not (tr & te)
        assert not (va & te)
        assert len(tr) + len(va) + len(te) == 10

    def test_split_files_no_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fresh").mkdir()
            (root / "stale").mkdir()
            files = []
            for i in range(20):
                files.append(_make_image(root / "fresh" / f"f{i}.jpg"))
            for i in range(20):
                files.append(_make_image(root / "stale" / f"s{i}.jpg"))
            split = split_files(files)
            assert not split.by_session
            sets = [set(split.train_files), set(split.val_files), set(split.test_files)]
            assert not (sets[0] & sets[1])
            assert not (sets[0] & sets[2])
            assert not (sets[1] & sets[2])
            assert split.counts["train"] + split.counts["val"] + split.counts["test"] == 40

    def test_group_files_by_session(self):
        images = [Path("img1.jpg"), Path("img2.jpg"), Path("img3.jpg")]
        lookup = lambda p: "s1" if "1" in p.name else None
        grouped = group_files_by_session(images, lookup)
        assert grouped["s1"] == [Path("img1.jpg")]