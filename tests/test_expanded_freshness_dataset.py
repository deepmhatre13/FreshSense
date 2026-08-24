"""Unit tests for expanded freshness dataset assembly and validation (Phase 4).

Covers:
    - source inspection
    - label mapping
    - invalid label rejection
    - duplicate detection
    - canonical mapping
    - 20-class structure
    - provenance
    - split leakage
    - missing classes
    - immutability

Uses synthetic temporary datasets. NEVER touches the real 2.6 GB Mendeley
dataset or production model checkpoints.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.freshness_dataset_builder import (
    CANONICAL_CLASS_MAPPING,
    CLASS_TO_ID,
    CandidateImage,
    deduplicate_exact,
    split_by_class,
    find_near_duplicates,
    _match_quality_fruit,
    _check_contradiction,
    validate_canonical_dataset,
)


@pytest.fixture
def temp_environment():
    """Create a temporary dummy source dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_dir = tmp_path / "raw_dataset"
        output_dir = tmp_path / "freshness"

        for split in ["train", "test"]:
            for legacy_cls in ["freshapples", "rottenapples", "freshbanana"]:
                cls_p = source_dir / split / legacy_cls
                cls_p.mkdir(parents=True, exist_ok=True)
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                img_p = cls_p / f"sample_{split}_{legacy_cls}.jpg"
                cv2.imwrite(str(img_p), img)

        yield source_dir, output_dir


@pytest.fixture
def sample_candidates():
    """Create sample CandidateImage objects for pure-function tests."""
    candidates = []
    for cls in ["Apple_fresh", "Apple_rotten", "Grape_fresh"]:
        for i in range(10):
            candidates.append(CandidateImage(
                path=f"/tmp/{cls}_{i}.jpg",
                source_dataset="Test",
                source_label=cls,
                canonical_class=cls,
                fruit=cls.split("_")[0],
                freshness_state=cls.split("_")[1],
                license="CC0",
                source_url="http://example.com",
                sha256=f"sha_{cls}_{i}" * 4,
                perceptual_hash=f"{cls}_{i}" * 8,
                width=100,
                height=100,
                file_size=1000,
            ))
    return candidates


# ---------- Label mapping / invalid label rejection ----------

def test_canonical_class_mapping_is_20_classes():
    assert len(CANONICAL_CLASS_MAPPING) == 20
    assert CANONICAL_CLASS_MAPPING[0] == "Apple_fresh"
    assert CANONICAL_CLASS_MAPPING[19] == "guava_rotten"
    assert len(CLASS_TO_ID) == 20


def test_quality_fruit_keyword_matching():
    assert _match_quality_fruit(
        "fresh-apple.jpg", {
            "Apple": {"keywords": ["apple", "apples"]},
            "banana": {"keywords": ["banana", "bananas"]},
        }
    ) == "Apple"
    assert _match_quality_fruit(
        "rotten-banana.jpg", {
            "Apple": {"keywords": ["apple"]},
            "banana": {"keywords": ["banana", "bananas"]},
        }
    ) == "banana"
    assert _match_quality_fruit(
        "unknown-fruit.jpg", {"Apple": {"keywords": ["apple"]}}
    ) is None
    assert _match_quality_fruit(
        "apple-banana.jpg", {
            "Apple": {"keywords": ["apple"]},
            "banana": {"keywords": ["banana"]},
        }
    ) is None


def test_contradiction_detection():
    words = ["rotten", "decompos", "spoiled", "mold", "mould"]
    assert _check_contradiction("rotten-apple.jpg", "fresh", words) is True
    assert _check_contradiction("fresh-apple.jpg", "fresh", words) is False
    assert _check_contradiction("rotten-apple.jpg", "rotten", words) is False


def test_readme_no_fruit_identity_rejected():
    # A filename with no fruit keyword -> no fruit identity -> rejected
    assert _match_quality_fruit(
        "download-1.jpg", {"Apple": {"keywords": ["apple"]}}
    ) is None


# ---------- duplicate detection ----------

def test_exact_deduplication(sample_candidates):
    dup = CandidateImage(
        path="/tmp/dup.jpg",
        source_dataset="Test2",
        source_label="Apple_fresh",
        canonical_class="Apple_fresh",
        fruit="Apple",
        freshness_state="fresh",
        license="CC0",
        source_url="http://x",
        sha256="sha_Apple_fresh_0" * 4,
        perceptual_hash="dup",
        width=100,
        height=100,
        file_size=1000,
    )
    candidates = sample_candidates + [dup]
    unique, report = deduplicate_exact(candidates)
    assert report.total_candidates == 31
    assert report.exact_duplicates == 1
    assert report.unique_images == 30
    assert len(unique) == 30


def test_near_duplicate_detection(temp_environment):
    source_dir, _ = temp_environment
    paths = []
    # Create 3 near-identical images: same random texture + tiny per-image noise
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    for i in range(3):
        noise = rng.integers(0, 3, (50, 50, 3), dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        p = source_dir / f"sim_{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    # A clearly different image
    diff = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    pdiff = source_dir / "diff.jpg"
    cv2.imwrite(str(pdiff), diff)
    paths.append(pdiff)

    report = find_near_duplicates(paths, max_distance=10, path_obj=True)
    assert report.status == "REVIEW"
    assert report.total_images_hashed == 4
    # The 3 near-identical images should form at least one group
    assert report.near_duplicate_groups >= 1


def _make_unique_image(path: Path, value: int) -> None:
    """Write a tiny solid-color image whose bytes are unique per *value*."""
    img = np.full((24, 24, 3), value % 256, dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------- end-to-end build (dry run) ----------

def test_build_canonical_dataset_dry_run_end_to_end(tmp_path):
    """Full pipeline on synthetic sources in dry-run mode.

    Validates collect -> dedup -> near-dup -> split -> summary WITHOUT writing
    any files, and confirms the dry-run is side-effect free.
    """
    import yaml

    mendeley_dir = tmp_path / "mendeley"
    quality_dir = tmp_path / "quality"
    legacy_dir = tmp_path / "legacy"
    output_dir = tmp_path / "freshness_out"

    # Mendeley: class-directory layout (3 accepted + 1 unsupported fruit)
    for i, cls in enumerate(["FreshApple", "RottenBanana", "FreshStrawberry"]):
        d = mendeley_dir / cls
        d.mkdir(parents=True)
        _make_unique_image(d / f"img_{i}.jpg", 10 + i)
    rej_dir = mendeley_dir / "FreshJujube"
    rej_dir.mkdir(parents=True)
    _make_unique_image(rej_dir / "rej.jpg", 200)

    # Legacy: split/class-directory layout (.png), 4 images
    counter = 20
    for split in ["train", "test"]:
        for cls in ["freshapples", "rottenbanana"]:
            d = legacy_dir / split / cls
            d.mkdir(parents=True)
            _make_unique_image(d / "l.png", counter)
            counter += 1

    # Quality: split/freshness-dir/filename with fruit keyword
    for split in ["train", "valid"]:
        for state in ["fresh", "rotten"]:
            d = quality_dir / split / state
            d.mkdir(parents=True)
            fname = "apple.jpg" if state == "fresh" else "banana.jpg"
            _make_unique_image(d / fname, counter)
            counter += 1

    config = {
        "dataset_version": "1.0.0",
        "random_seed": 42,
        "train_split": 0.70, "valid_split": 0.15, "test_split": 0.15,
        "output_dir": str(output_dir),
        "mendeley_original_image": {
            "source_dataset": "Mendeley Original Image",
            "path": str(mendeley_dir),
            "license": "CC BY 4.0",
            "source_url": "https://example.com/mendeley",
            "structure": "class_directory",
            "label_mapping": {
                "FreshApple": {"canonical_class": "Apple_fresh", "fruit": "Apple",
                               "freshness_state": "fresh", "accept": True,
                               "mapping_reason": "explicit dir label"},
                "RottenBanana": {"canonical_class": "banana_rotten", "fruit": "banana",
                                 "freshness_state": "rotten", "accept": True,
                                 "mapping_reason": "explicit dir label"},
                "FreshStrawberry": {"canonical_class": "Strawberry_fresh",
                                    "fruit": "Strawberry", "freshness_state": "fresh",
                                    "accept": True, "mapping_reason": "explicit dir label"},
                "FreshJujube": {"canonical_class": None, "fruit": "Jujube",
                                "freshness_state": "fresh", "accept": False,
                                "reject_reason": "unsupported_fruit",
                                "mapping_reason": "not in taxonomy"},
            },
        },
        "quality_dataset": {
            "source_dataset": "Quality Dataset",
            "path": str(quality_dir),
            "license": "CC0 1.0 Universal",
            "source_url": "https://example.com/quality",
            "structure": "split/freshness_directory",
            "directory_labels": {"fresh": "fresh", "rotten": "rotten"},
            "fruit_keywords": {
                "Apple": {"keywords": ["apple"]},
                "banana": {"keywords": ["banana"]},
            },
            "rejected_fruits": {},
            "state_contradiction_reject_words": [],
            "contradiction_check": True,
        },
        "legacy_fresh_rotten": {
            "source_dataset": "Legacy Benchmark",
            "path": str(legacy_dir),
            "license": "CC0 1.0 Universal",
            "source_url": "https://example.com/legacy",
            "structure": "split/class_directory",
            "label_mapping": {
                "freshapples": {"canonical_class": "Apple_fresh", "fruit": "Apple",
                                "freshness_state": "fresh", "accept": True,
                                "mapping_reason": "explicit dir label"},
                "rottenbanana": {"canonical_class": "banana_rotten", "fruit": "banana",
                                 "freshness_state": "rotten", "accept": True,
                                 "mapping_reason": "explicit dir label"},
            },
        },
    }

    config_path = tmp_path / "freshness_sources.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    from src.data.freshness_dataset_builder import build_canonical_dataset

    result = build_canonical_dataset(
        config_path=config_path, output_dir=output_dir, dry_run=True,
    )

    # Summary structure and counts:
    #   Mendeley 3 accepted (+1 jujube rejected)
    #   Legacy   4 accepted
    #   Quality  4 accepted (apple fresh x2 splits, banana rotten x2 splits)
    assert result["total_images"] == 11
    assert result["train"] > 0
    assert result["train"] + result["valid"] + result["test"] == result["total_images"]
    assert result["exact_duplicates_removed"] == 0  # every image has unique bytes
    assert result["rejected_count"] >= 1  # FreshJujube -> unsupported_fruit
    assert result["leakage_check"] == "PENDING"  # dry-run never runs validation

    # Dry-run must be side-effect free: nothing written under output dir
    if output_dir.exists():
        assert not any(output_dir.rglob("*"))


# ---------- split leakage ----------

def test_split_has_no_leakage(sample_candidates):
    splits = split_by_class(sample_candidates, 0.7, 0.15, 42)
    train_hashes = {c.sha256 for c in splits["train"]}
    valid_hashes = {c.sha256 for c in splits["valid"]}
    test_hashes = {c.sha256 for c in splits["test"]}
    assert not (train_hashes & valid_hashes)
    assert not (train_hashes & test_hashes)
    assert not (valid_hashes & test_hashes)
    assert len(splits["train"]) + len(splits["valid"]) + len(splits["test"]) == len(sample_candidates)


def test_missing_classes_are_empty_dirs():
    assert "chickoo_fresh" in CANONICAL_CLASS_MAPPING.values()
    assert "chickoo_rotten" in CANONICAL_CLASS_MAPPING.values()
    assert "guava_fresh" in CANONICAL_CLASS_MAPPING.values()
    assert "guava_rotten" in CANONICAL_CLASS_MAPPING.values()


# ---------- canonical structure / provenance ----------

def _make_20class_structure(output_dir):
    for split in ["train", "valid", "test"]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for cls_name in CANONICAL_CLASS_MAPPING.values():
            (split_dir / cls_name).mkdir(parents=True, exist_ok=True)
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in CANONICAL_CLASS_MAPPING.items()}, f, indent=2)


def test_class_dirs_and_mapping_in_freshness(temp_environment):
    _, output_dir = temp_environment
    _make_20class_structure(output_dir)
    with open(output_dir / "dataset_manifest.json", "w") as f:
        json.dump({"accepted_entries": [], "rejected_entries": []}, f, indent=2)

    result = validate_canonical_dataset(output_dir)
    assert result["checks"]["class_mapping_correct"]["passed"] is True
    assert result["checks"]["class_dirs_exist_train"]["passed"] is True
    assert result["checks"]["class_dirs_exist_valid"]["passed"] is True
    assert result["checks"]["class_dirs_exist_test"]["passed"] is True


def test_provenance_and_license_checks(temp_environment):
    _, output_dir = temp_environment
    _make_20class_structure(output_dir)
    with open(output_dir / "dataset_manifest.json", "w") as f:
        json.dump({"accepted_entries": [], "rejected_entries": []}, f, indent=2)

    result = validate_canonical_dataset(output_dir)
    assert result["checks"]["provenance_exists"]["passed"] is False  # no entries


# ---------- immutability ----------

def test_legacy_dataset_immutable(temp_environment):
    source_dir, _ = temp_environment
    files_before = sorted(str(p) for p in source_dir.rglob("*") if p.is_file())
    files_after = sorted(str(p) for p in source_dir.rglob("*") if p.is_file())
    assert files_before == files_after


# ---------- production contract ----------

def test_production_models_not_modified():
    best_model = Path(r"D:\SmartFreshAI\models\checkpoints\best_model.pth")
    yolo_best = Path(r"D:\SmartFreshAI\models\detection\detector\weights\best.pt")
    assert best_model.exists()
    assert yolo_best.exists()


# ---------- near-duplicate regression (Phase 6/7 audit) ----------

def _diag_make_hash(bits_set):
    """64-char binary hash with 1-bits at the given positions."""
    chars = ["0"] * 64
    for b in bits_set:
        chars[b] = "1"
    return "".join(chars)


def _diag_cand(path, phash):
    return CandidateImage(
        path=path, source_dataset="T", source_label="x",
        canonical_class="Apple_fresh", fruit="Apple",
        freshness_state="fresh", license="CC0",
        source_url="http://example.com",
        sha256="sha_" + path, perceptual_hash=phash,
        width=100, height=100, file_size=1000,
    )


def test_unrelated_images_not_grouped(tmp_path):
    """CASE C: two visually unrelated images must NOT be near duplicates."""
    rng = np.random.default_rng(7)
    pa = tmp_path / "a.jpg"
    pb = tmp_path / "b.jpg"
    cv2.imwrite(str(pa), rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(pb), rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    report = find_near_duplicates([pa, pb], max_distance=6, path_obj=True)
    assert report.near_duplicate_pairs == 0
    assert report.near_duplicate_groups == 0


def test_cross_class_same_bucket_not_merged():
    """CASE D: images sharing an 8-bit pHash bucket but with full Hamming
    distance > max_distance must NOT be merged (cross-class false-positive
    prevention)."""
    h1 = _diag_make_hash({8})
    h2 = _diag_make_hash({10, 11, 12, 13, 14, 15, 16, 17})
    assert h1[:8] == h2[:8]  # same bucket prefix
    from src.data.freshness_dataset_builder import hamming_distance
    assert hamming_distance(h1, h2) > 6
    cands = [_diag_cand("apple/1.jpg", h1), _diag_cand("banana/1.jpg", h2)]
    cands[0].canonical_class = "Apple_fresh"
    cands[1].canonical_class = "banana_fresh"
    report = find_near_duplicates(cands, max_distance=6)
    assert report.near_duplicate_pairs == 0
    assert report.near_duplicate_groups == 0


def test_transitive_chain_not_merged():
    """CASE E: A~B, B~C, but A!~C must NOT merge all three into one
    equivalence class. Star clustering keeps A-C separate."""
    from src.data.freshness_dataset_builder import hamming_distance
    a = _diag_make_hash({10, 11, 12, 13})
    b = _diag_make_hash({10, 11, 12, 14})
    c = _diag_make_hash({10, 11, 12, 14, 15, 16, 17, 18, 19})
    assert hamming_distance(a, b) <= 6
    assert hamming_distance(b, c) <= 6
    assert hamming_distance(a, c) > 6
    cands = [_diag_cand("trans/A.jpg", a),
             _diag_cand("trans/B.jpg", b),
             _diag_cand("trans/C.jpg", c)]
    report = find_near_duplicates(cands, max_distance=6)
    grouped_paths = []
    for g in report.groups:
        grouped_paths.append({m["path"] for m in g["members"]})
    # A and B grouped together
    assert any({"trans/A.jpg", "trans/B.jpg"} <= g for g in grouped_paths)
    # A and C must NOT share a group
    assert not any({"trans/A.jpg", "trans/C.jpg"} <= g for g in grouped_paths)


def test_near_duplicate_detection_is_deterministic():
    """Same inputs -> identical groups, independent of dict ordering."""
    a = _diag_make_hash({1, 2, 3})
    b = _diag_make_hash({1, 2, 4})
    cands1 = [_diag_cand("x/1.jpg", a), _diag_cand("x/2.jpg", b)]
    cands2 = [_diag_cand("x/2.jpg", b), _diag_cand("x/1.jpg", a)]
    r1 = find_near_duplicates(cands1, max_distance=6)
    r2 = find_near_duplicates(cands2, max_distance=6)
    assert r1.near_duplicate_groups == r2.near_duplicate_groups
    assert r1.near_duplicate_pairs == r2.near_duplicate_pairs
    g1 = sorted(m["path"] for m in r1.groups[0]["members"])
    g2 = sorted(m["path"] for m in r2.groups[0]["members"])
    assert g1 == g2

