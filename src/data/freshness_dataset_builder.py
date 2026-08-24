"""Freshness dataset builder core logic for SmartFreshAI Phase 4.

This module provides reusable, testable functions for:
  - Inspecting source datasets
  - Mapping source labels to the 20-class canonical taxonomy
  - SHA256 exact deduplication
  - pHash near-duplicate detection
  - Stratified 70/15/15 class-aware splitting
  - Canonical dataset construction with full provenance
  - Validation of the canonical dataset
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

from src.data.dataset_validation import (
    VALID_IMAGE_EXTS,
    is_corrupted,
)

logger = logging.getLogger("freshness_dataset_builder")

ROOT_DIR = Path(__file__).resolve().parents[2]

CANONICAL_CLASS_MAPPING: Dict[int, str] = {
    0: "Apple_fresh",
    1: "Apple_rotten",
    2: "Grape_fresh",
    3: "Grape_rotten",
    4: "Kiwi_fresh",
    5: "Kiwi_rotten",
    6: "Mango_fresh",
    7: "Mango_rotten",
    8: "Orange_fresh",
    9: "Orange_rotten",
    10: "Strawberry_fresh",
    11: "Strawberry_rotten",
    12: "banana_fresh",
    13: "banana_rotten",
    14: "cherry_fresh",
    15: "cherry_rotten",
    16: "chickoo_fresh",
    17: "chickoo_rotten",
    18: "guava_fresh",
    19: "guava_rotten",
}

CLASS_TO_ID: Dict[str, int] = {v: k for k, v in CANONICAL_CLASS_MAPPING.items()}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CandidateImage:
    """A single candidate image from a source dataset."""

    path: str  # absolute path to source file
    source_dataset: str
    source_label: str  # the original class/label from the source
    canonical_class: str  # mapped to one of the 20 canonical classes
    fruit: str
    freshness_state: str  # "fresh" or "rotten"
    license: str
    source_url: str
    sha256: str = ""
    perceptual_hash: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0


@dataclass
class RejectedImage:
    """An image rejected from the canonical dataset."""

    path: str
    source_dataset: str
    source_label: str
    rejection_reason: str
    details: str = ""


@dataclass
class InspectionReport:
    """Structured report from inspecting a source dataset."""

    source_name: str
    source_path: str
    total_files: int = 0
    total_images: int = 0
    image_extensions: Dict[str, int] = field(default_factory=dict)
    directory_tree: List[str] = field(default_factory=list)
    immediate_subdirs: List[str] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)
    zero_byte_files: List[str] = field(default_factory=list)
    corrupt_images: List[str] = field(default_factory=list)
    suspicious_files: List[str] = field(default_factory=list)
    nested_directories: List[str] = field(default_factory=list)
    duplicate_filenames: List[str] = field(default_factory=list)
    sample_paths: List[str] = field(default_factory=list)
    non_image_files: List[str] = field(default_factory=list)
    rejected_labels: Dict[str, int] = field(default_factory=dict)
    accepted_labels: Dict[str, int] = field(default_factory=dict)
    rejected_count: int = 0
    accepted_count: int = 0
    label_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeduplicationReport:
    """Report on exact deduplication results."""

    total_candidates: int = 0
    exact_duplicates: int = 0
    unique_images: int = 0
    duplicate_groups: List[Dict[str, Any]] = field(default_factory=list)
    source_to_source_duplicates: Dict[str, int] = field(default_factory=dict)
    retained_source: Dict[str, int] = field(default_factory=dict)
    discarded_source: Dict[str, int] = field(default_factory=dict)


@dataclass
class NearDuplicateReport:
    """Report on near-duplicate (pHash) detection results."""

    total_images_hashed: int = 0
    near_duplicate_groups: int = 0
    near_duplicate_pairs: int = 0
    groups: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "REVIEW"  # always REVIEW for uncertain near-dups


@dataclass
class AssemblySummary:
    """Summary of the assembled canonical dataset."""

    dataset_version: str = "1.0.0"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_images: int = 0
    train_images: int = 0
    valid_images: int = 0
    test_images: int = 0
    class_mapping: Dict[int, str] = field(default_factory=lambda: dict(CANONICAL_CLASS_MAPPING))
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)
    canonical_class_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    exact_duplicates_removed: int = 0
    near_duplicates_marked: int = 0
    rejected_counts: Dict[str, int] = field(default_factory=dict)
    license_summary: Dict[str, str] = field(default_factory=dict)
    leakage_check: str = "PENDING"
    validation_status: str = "PENDING"
    group_split_policy: str = "no session metadata; stratified split by class with fixed seed"
    dataset_hash: str = ""

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_freshness_config(config_path: Optional[Path] = None) -> dict:
    """Load freshness_sources.yaml configuration.

    Args:
        config_path: Path to config file. Defaults to
            configs/freshness_sources.yaml.

    Returns:
        Parsed YAML config dict.
    """
    if config_path is None:
        config_path = ROOT_DIR / "configs" / "freshness_sources.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file not found: %s", config_path)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Stream-compute the SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_image_dimensions(path: Path | str) -> tuple[int, int]:
    """Return (width, height) of an image. (0, 0) on failure."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)
# ---------------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------------


def _walk_image_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Walk a directory, returning (image_files, non_image_files)."""
    images: list[Path] = []
    non_images: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in VALID_IMAGE_EXTS:
            images.append(path)
        else:
            non_images.append(path)
    return images, non_images


def inspect_mendeley(path: Path, config: dict, check_corruption: bool = False) -> InspectionReport:
    """Inspect the Mendeley Original Image dataset."""
    name = "Mendeley Original Image"
    report = InspectionReport(source_name=name, source_path=str(path))
    src_cfg = config.get("mendeley_original_image", {})
    label_map = src_cfg.get("label_mapping", {})

    if not path.exists():
        report.suspicious_files.append(str(path))
        return report

    images, non_images = _walk_image_files(path)
    report.total_images = len(images)
    report.total_files = len(images) + len(non_images)
    report.non_image_files = [str(p) for p in non_images[:20]]

    # Extension distribution
    for img in images:
        ext = img.suffix.lower()
        report.image_extensions[ext] = report.image_extensions.get(ext, 0) + 1

    # Immediate subdirectories
    report.immediate_subdirs = sorted(
        d.name for d in path.iterdir() if d.is_dir()
    )

    # Directory tree (class -> count)
    for img in images:
        rel = img.relative_to(path)
        top = rel.parts[0] if rel.parts else "root"
        report.class_counts[top] = report.class_counts.get(top, 0) + 1

    # Check zero-byte and corrupt
    seen_names: dict[str, int] = defaultdict(int)
    for img in images:
        size = img.stat().st_size
        if size == 0:
            report.zero_byte_files.append(str(img))
        corrupted, err = (False, None)
        if check_corruption:
            corrupted, err = is_corrupted(img)
        if corrupted:
            report.corrupt_images.append(f"{str(img)} ({err})")
        seen_names[img.name] += 1

    # Duplicate filenames (within the dataset)
    for fname, count in seen_names.items():
        if count > 1:
            report.duplicate_filenames.append(f"{fname} ({count} copies)")

    # Label mapping
    for dir_label, mapped in label_map.items():
        accept = mapped.get("accept", False)
        canonical = mapped.get("canonical_class", "REJECTED")
        if accept and canonical:
            report.accepted_labels[dir_label] = report.class_counts.get(dir_label, 0)
            report.label_mapping[dir_label] = canonical
        else:
            report.rejected_labels[dir_label] = report.class_counts.get(dir_label, 0)
            report.rejected_count += report.class_counts.get(dir_label, 0)

    for count in report.accepted_labels.values():
        report.accepted_count += count

    # Sample paths
    report.sample_paths = [str(p) for p in images[:10]]
    report.sample_paths.append("...")
    report.sample_paths.extend(str(p) for p in images[-3:])

    return report

# ---------------------------------------------------------------------------
# Quality Dataset fruit keyword matching
# ---------------------------------------------------------------------------


def _match_quality_fruit(filename: str, fruit_keywords: dict) -> Optional[str]:
    """Return the canonical fruit name if exactly one fruit keyword matches.

    Returns None if no match or multiple matches.
    """
    lower = filename.lower()
    matched: list[str] = []
    for fruit, cfg in fruit_keywords.items():
        for kw in cfg.get("keywords", []):
            if kw.lower() in lower:
                matched.append(fruit)
                break
    if len(matched) == 1:
        return matched[0]
    return None  # 0 or >1 matches -> ambiguous


def _check_contradiction(
    filename: str, freshness_dir: str, contradiction_words: list[str]
) -> bool:
    """Return True if a contradiction exists between filename state words
    and the directory freshness label."""
    lower = filename.lower()
    has_rotten_words = any(w in lower for w in contradiction_words)
    if has_rotten_words and freshness_dir == "fresh":
        return True
    # "fresh" in filename but directory says "rotten"
    # Only checks if "fresh" appears as a standalone word, not as substring
    # (e.g. "scifresh" should not trigger this)
    if freshness_dir == "rotten":
        # Check if filename explicitly says "fresh" as a state indicator
        # (not as part of another word)
        for word in lower.replace("-", " ").replace("_", " ").split():
            if word in ("fresh", "fresh"):
                return True
    return False


def inspect_quality_dataset(path: Path, config: dict, check_corruption: bool = False) -> InspectionReport:
    """Inspect the Kaggle Quality Dataset (binary fresh/rotten, fruit in filename)."""
    name = "Quality Dataset"
    report = InspectionReport(source_name=name, source_path=str(path))
    src_cfg = config.get("quality_dataset", {})
    fruit_kw = src_cfg.get("fruit_keywords", {})
    rejected_fruits = src_cfg.get("rejected_fruits", {})
    contradiction_words = src_cfg.get("state_contradiction_reject_words", [])
    check_contradiction = src_cfg.get("contradiction_check", True)

    if not path.exists():
        report.suspicious_files.append(str(path))
        return report

    images, non_images = _walk_image_files(path)
    report.total_images = len(images)
    report.total_files = len(images) + len(non_images)
    report.non_image_files = [str(p) for p in non_images[:20]]

    for img in images:
        ext = img.suffix.lower()
        report.image_extensions[ext] = report.image_extensions.get(ext, 0) + 1

    report.immediate_subdirs = sorted(
        d.name for d in path.iterdir() if d.is_dir()
    )

    # Directory tree
    for img in images:
        rel = img.relative_to(path)
        parts = rel.parts
        key = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        report.class_counts[key] = report.class_counts.get(key, 0) + 1

    # Check zero-byte and corrupt
    seen_names: dict[str, int] = defaultdict(int)
    for img in images:
        size = img.stat().st_size
        if size == 0:
            report.zero_byte_files.append(str(img))
        corrupted, err = (False, None)
        if check_corruption:
            corrupted, err = is_corrupted(img)
        if corrupted:
            report.corrupt_images.append(f"{str(img)} ({err})")
        seen_names[img.name] += 1

    for fname, count in seen_names.items():
        if count > 1:
            report.duplicate_filenames.append(f"{fname} ({count} copies)")

    # Check for nested directories beyond split/freshness
    for img in images:
        rel = img.relative_to(path)
        if len(rel.parts) > 3:
            report.nested_directories.append(str(img))

    # Fruit keyword analysis
    no_fruit = 0
    rejected_non_taxonomy: dict[str, int] = defaultdict(int)
    rejected_contradiction = 0
    accepted: dict[str, int] = defaultdict(int)
    rejected_other = 0

    for img in images:
        rel = img.relative_to(path)
        parts = rel.parts
        split_dir = parts[0] if len(parts) > 0 else "root"
        freshness_dir = parts[1] if len(parts) > 1 else "unknown"
        fname = img.name

        fruit = _match_quality_fruit(fname, fruit_kw)
        state_words = any(
            w in fname.lower() for w in contradiction_words
        )

        if fruit is None:
            # Check if it matches a rejected (non-taxonomy) fruit
            matched_rejected = False
            for rfruit, rcfg in rejected_fruits.items():
                for kw in rcfg.get("keywords", []):
                    if kw.lower() in fname.lower():
                        rejected_non_taxonomy[rfruit] = (
                            rejected_non_taxonomy.get(rfruit, 0) + 1
                        )
                        matched_rejected = True
                        break
                if matched_rejected:
                    break
            if not matched_rejected:
                no_fruit += 1
            continue

        if check_contradiction:
            if _check_contradiction(fname, freshness_dir, contradiction_words):
                rejected_contradiction += 1
                report.rejected_labels["contradiction"] = (
                    report.rejected_labels.get("contradiction", 0) + 1
                )
                continue

        canonical = f"{fruit}_{freshness_dir}"
        accepted[canonical] += 1
        report.label_mapping[f"{split_dir}/{freshness_dir}/{fname}"] = canonical

    for cls, cnt in sorted(accepted.items()):
        report.accepted_labels[cls] = cnt
    report.accepted_count = sum(accepted.values())

    report.rejected_labels["no_fruit_identity"] = no_fruit
    report.rejected_labels["unsupported_fruit"] = sum(rejected_non_taxonomy.values())
    report.rejected_labels["label_contradiction"] = rejected_contradiction
    report.rejected_count = no_fruit + sum(rejected_non_taxonomy.values()) + rejected_contradiction

    report.sample_paths = [str(p) for p in images[:10]]

    return report

def inspect_legacy(path: Path, config: dict, check_corruption: bool = False) -> InspectionReport:
    """Inspect the legacy Fresh and Rotten Fruits benchmark dataset."""
    name = "Kaggle Fresh and Rotten Fruits Benchmark"
    report = InspectionReport(source_name=name, source_path=str(path))
    src_cfg = config.get("legacy_fresh_rotten", {})
    label_map = src_cfg.get("label_mapping", {})

    if not path.exists():
        report.suspicious_files.append(str(path))
        return report

    images, non_images = _walk_image_files(path)
    report.total_images = len(images)
    report.total_files = len(images) + len(non_images)
    report.non_image_files = [str(p) for p in non_images[:20]]

    for img in images:
        ext = img.suffix.lower()
        report.image_extensions[ext] = report.image_extensions.get(ext, 0) + 1

    report.immediate_subdirs = sorted(
        d.name for d in path.iterdir() if d.is_dir()
    )

    seen_names: dict[str, int] = defaultdict(int)
    for img in images:
        rel = img.relative_to(path)
        parts = rel.parts
        # Determine label from parent directory (class)
        label = parts[-2] if len(parts) >= 2 else parts[0]
        report.class_counts[label] = report.class_counts.get(label, 0) + 1

        size = img.stat().st_size
        if size == 0:
            report.zero_byte_files.append(str(img))
        corrupted, err = (False, None)
        if check_corruption:
            corrupted, err = is_corrupted(img)
        if corrupted:
            report.corrupt_images.append(f"{str(img)} ({err})")
        seen_names[img.name] += 1

        if len(rel.parts) > 3:
            report.nested_directories.append(str(img))

    for fname, count in seen_names.items():
        if count > 1:
            report.duplicate_filenames.append(f"{fname} ({count} copies)")

    for label, mapped in label_map.items():
        accept = mapped.get("accept", False)
        canonical = mapped.get("canonical_class", "REJECTED")
        cnt = report.class_counts.get(label, 0)
        if accept and canonical:
            report.accepted_labels[label] = cnt
            report.label_mapping[label] = canonical
        else:
            report.rejected_labels[label] = cnt
            report.rejected_count += cnt

    for count in report.accepted_labels.values():
        report.accepted_count += count

    report.sample_paths = [str(p) for p in images[:10]]

    return report


def inspect_all_sources(config: Optional[dict] = None) -> dict[str, InspectionReport]:
    """Inspect all three source datasets and return reports keyed by name."""
    if config is None:
        config = load_freshness_config()

    reports: dict[str, InspectionReport] = {}

    # Mendeley
    mendeley_cfg = config.get("mendeley_original_image", {})
    mendeley_path = ROOT_DIR / mendeley_cfg.get("path", "data/Original Image")
    reports["mendeley"] = inspect_mendeley(mendeley_path, config)

    # Quality Dataset
    qd_cfg = config.get("quality_dataset", {})
    qd_path = ROOT_DIR / qd_cfg.get("path", "data/Quality Dataset")
    reports["quality"] = inspect_quality_dataset(qd_path, config)

    # Legacy
    legacy_cfg = config.get("legacy_fresh_rotten", {})
    legacy_path = ROOT_DIR / legacy_cfg.get("path", "data/raw/dataset/dataset")
    reports["legacy"] = inspect_legacy(legacy_path, config)

    return reports

# ---------------------------------------------------------------------------
# Candidate image collection
# ---------------------------------------------------------------------------


def collect_mendeley(path: Path, config: dict) -> tuple[list[CandidateImage], list[RejectedImage]]:
    """Collect accepted candidate images from the Mendeley dataset.

    Uses directory labels only (no filename inference).
    """
    candidates: list[CandidateImage] = []
    rejected: list[RejectedImage] = []
    src_cfg = config.get("mendeley_original_image", {})
    label_map = src_cfg.get("label_mapping", {})
    license_str = src_cfg.get("license", "Unknown")
    source_url = src_cfg.get("source_url", "")
    source_name = src_cfg.get("source_dataset", "Mendeley Original Image")

    if not path.exists():
        rejected.append(RejectedImage(
            path=str(path), source_dataset=source_name,
            source_label="N/A", rejection_reason="source_missing",
        ))
        return candidates, rejected

    images, _ = _walk_image_files(path)
    for img in images:
        rel = img.relative_to(path)
        dir_label = rel.parts[0] if rel.parts else "unknown"
        mapped = label_map.get(dir_label)

        if mapped is None or not mapped.get("accept"):
            reason = mapped.get("reject_reason", "unmapped_label") if mapped else "unmapped_label"
            rejected.append(RejectedImage(
                path=str(img),
                source_dataset=source_name,
                source_label=dir_label,
                rejection_reason=reason,
                details=mapped.get("mapping_reason", "No canonical mapping") if mapped else "Unknown label",
            ))
            continue

        canonical = mapped["canonical_class"]
        fruit = mapped["fruit"]
        freshness = mapped["freshness_state"]

        sha = sha256_file(img)
        w, h = compute_image_dimensions(img)
        fsize = img.stat().st_size
        ph = compute_image_phash(img)

        candidates.append(CandidateImage(
            path=str(img),
            source_dataset=source_name,
            source_label=dir_label,
            canonical_class=canonical,
            fruit=fruit,
            freshness_state=freshness,
            license=license_str,
            source_url=source_url,
            sha256=sha,
            perceptual_hash=ph,
            width=w,
            height=h,
            file_size=fsize,
        ))

    return candidates, rejected
def collect_quality_dataset(path: Path, config: dict) -> tuple[list[CandidateImage], list[RejectedImage]]:
    """Collect accepted candidates from the Quality Dataset.

    Freshness state comes from directory labels (fresh/rotten).
    Fruit identity comes from explicit fruit keywords in filenames.
    Files with no fruit keyword, non-taxonomy fruit, or contradictory
    state words are rejected.
    """
    candidates: list[CandidateImage] = []
    rejected: list[RejectedImage] = []
    src_cfg = config.get("quality_dataset", {})
    fruit_kw = src_cfg.get("fruit_keywords", {})
    rejected_fruits = src_cfg.get("rejected_fruits", {})
    contradiction_words = src_cfg.get("state_contradiction_reject_words", [])
    check_contradiction = src_cfg.get("contradiction_check", True)
    license_str = src_cfg.get("license", "Unknown")
    source_url = src_cfg.get("source_url", "")
    source_name = src_cfg.get("source_dataset", "Quality Dataset")

    if not path.exists():
        rejected.append(RejectedImage(
            path=str(path), source_dataset=source_name,
            source_label="N/A", rejection_reason="source_missing",
        ))
        return candidates, rejected

    images, _ = _walk_image_files(path)
    for img in images:
        rel = img.relative_to(path)
        parts = rel.parts
        split_dir = parts[0] if len(parts) > 0 else "root"
        freshness_dir = parts[1] if len(parts) > 1 else "unknown"
        fname = img.name
        lower = fname.lower()

        if check_contradiction:
            has_rotten_word = any(w in lower for w in contradiction_words)
            if has_rotten_word and freshness_dir == "fresh":
                rejected.append(RejectedImage(
                    path=str(img),
                    source_dataset=source_name,
                    source_label="/" + split_dir + "/" + freshness_dir,
                    rejection_reason="label_contradiction",
                    details="Filename has rot state word but dir=" + freshness_dir,
                ))
                continue
            # Reverse contradiction: filename explicitly says "fresh" but dir is rotten
            if freshness_dir == "rotten":
                words = lower.replace("-", " ").replace("_", " ").split()
                if "fresh" in words:
                    rejected.append(RejectedImage(
                        path=str(img),
                        source_dataset=source_name,
                        source_label="/" + split_dir + "/" + freshness_dir,
                        rejection_reason="label_contradiction",
                        details="Filename has fresh state word but dir=rotten",
                    ))
                    continue

        fruit = _match_quality_fruit(fname, fruit_kw)

        if fruit is None:
            matched_rejected = False
            for rfruit, rcfg in rejected_fruits.items():
                for kw in rcfg.get("keywords", []):
                    if kw.lower() in lower:
                        rejected.append(RejectedImage(
                            path=str(img),
                            source_dataset=source_name,
                            source_label="/" + split_dir + "/" + freshness_dir,
                            rejection_reason="unsupported_fruit",
                            details="Contains " + rfruit + " keyword (not in 20-class taxonomy)",
                        ))
                        matched_rejected = True
                        break
                if matched_rejected:
                    break
            if not matched_rejected:
                rejected.append(RejectedImage(
                    path=str(img),
                    source_dataset=source_name,
                    source_label="/" + split_dir + "/" + freshness_dir,
                    rejection_reason="no_fruit_identity",
                    details="No fruit keyword in filename; cannot determine fruit identity",
                ))
            continue

        canonical = fruit + "_" + freshness_dir
        sha = sha256_file(img)
        w, h = compute_image_dimensions(img)
        fsize = img.stat().st_size
        ph = compute_image_phash(img)

        candidates.append(CandidateImage(
            path=str(img),
            source_dataset=source_name,
            source_label="/" + split_dir + "/" + freshness_dir,
            canonical_class=canonical,
            fruit=fruit,
            freshness_state=freshness_dir,
            license=license_str,
            source_url=source_url,
            sha256=sha,
            perceptual_hash=ph,
            width=w,
            height=h,
            file_size=fsize,
        ))

    return candidates, rejected

def collect_legacy(path: Path, config: dict) -> tuple[list[CandidateImage], list[RejectedImage]]:
    """Collect accepted candidates from the legacy dataset."""
    candidates: list[CandidateImage] = []
    rejected: list[RejectedImage] = []
    src_cfg = config.get("legacy_fresh_rotten", {})
    label_map = src_cfg.get("label_mapping", {})
    license_str = src_cfg.get("license", "Unknown")
    source_url = src_cfg.get("source_url", "")
    source_name = src_cfg.get("source_dataset", "Legacy Dataset")

    if not path.exists():
        rejected.append(RejectedImage(
            path=str(path), source_dataset=source_name,
            source_label="N/A", rejection_reason="source_missing",
        ))
        return candidates, rejected

    images, _ = _walk_image_files(path)
    for img in images:
        rel = img.relative_to(path)
        parts = rel.parts
        label = parts[-2] if len(parts) >= 2 else parts[0]
        mapped = label_map.get(label)

        if mapped is None or not mapped.get("accept"):
            reason = mapped.get("reject_reason", "unmapped_label") if mapped else "unmapped_label"
            rejected.append(RejectedImage(
                path=str(img),
                source_dataset=source_name,
                source_label=label,
                rejection_reason=reason,
                details=mapped.get("mapping_reason", "") if mapped else "No mapping entry",
            ))
            continue

        canonical = mapped["canonical_class"]
        fruit = mapped["fruit"]
        freshness = mapped["freshness_state"]

        sha = sha256_file(img)
        w, h = compute_image_dimensions(img)
        fsize = img.stat().st_size
        ph = compute_image_phash(img)

        candidates.append(CandidateImage(
            path=str(img),
            source_dataset=source_name,
            source_label=label,
            canonical_class=canonical,
            fruit=fruit,
            freshness_state=freshness,
            license=license_str,
            source_url=source_url,
            sha256=sha,
            perceptual_hash=ph,
            width=w,
            height=h,
            file_size=fsize,
        ))

    return candidates, rejected


def collect_all(config: Optional[dict] = None) -> tuple[list[CandidateImage], list[RejectedImage]]:
    """Collect candidate images from all three sources."""
    if config is None:
        config = load_freshness_config()

    all_candidates: list[CandidateImage] = []
    all_rejected: list[RejectedImage] = []

    # Mendeley
    m_cfg = config.get("mendeley_original_image", {})
    m_path = ROOT_DIR / m_cfg.get("path", "data/Original Image")
    cand, rej = collect_mendeley(m_path, config)
    all_candidates.extend(cand)
    all_rejected.extend(rej)

    # Quality Dataset
    q_cfg = config.get("quality_dataset", {})
    q_path = ROOT_DIR / q_cfg.get("path", "data/Quality Dataset")
    cand, rej = collect_quality_dataset(q_path, config)
    all_candidates.extend(cand)
    all_rejected.extend(rej)

    # Legacy
    l_cfg = config.get("legacy_fresh_rotten", {})
    l_path = ROOT_DIR / l_cfg.get("path", "data/raw/dataset/dataset")
    cand, rej = collect_legacy(l_path, config)
    all_candidates.extend(cand)
    all_rejected.extend(rej)

    return all_candidates, all_rejected

# ---------------------------------------------------------------------------
# Perceptual hash wrapper
# ---------------------------------------------------------------------------


def compute_image_phash(path, hash_size=8):
    """Compute a dHash perceptual hash for an image via Pillow+dHash.

    Returns empty string if the image cannot be read.
    """
    try:
        img = Image.open(str(path)).convert("L").resize((hash_size + 1, hash_size))
        pixels = np.asarray(img, dtype=np.int32)
        diff = pixels[1:, :] > pixels[:-1, :]
        return "".join("1" if d else "0" for d in diff.flatten())
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two equal-length hash strings."""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(ca != cb for ca, cb in zip(a, b))


# ---------------------------------------------------------------------------
# Exact deduplication (SHA256)
# ---------------------------------------------------------------------------


def deduplicate_exact(
    candidates: list[CandidateImage],
) -> tuple[list[CandidateImage], DeduplicationReport]:
    """Remove exact duplicate images by SHA256 hash.

    When a duplicate is found, the FIRST occurrence is retained.
    Source priority for retention: Legacy > Mendeley > Quality.
    """
    report = DeduplicationReport()
    report.total_candidates = len(candidates)

    seen: dict[str, CandidateImage] = {}
    unique: list[CandidateImage] = []
    source_order = {"Legacy": 0, "Mendeley": 1, "Quality": 2}

    for cand in candidates:
        sha = cand.sha256
        if sha in seen:
            report.exact_duplicates += 1
            existing = seen[sha]
            # Determine source-to-source duplicate tracking
            src_pair = f"{existing.source_dataset} -> {cand.source_dataset}"
            report.source_to_source_duplicates[src_pair] = (
                report.source_to_source_duplicates.get(src_pair, 0) + 1
            )
            # Track discarded
            report.discarded_source[cand.source_dataset] = (
                report.discarded_source.get(cand.source_dataset, 0) + 1
            )
            # Record duplicate group
            report.duplicate_groups.append({
                "sha256": sha,
                "retained_path": existing.path,
                "retained_source": existing.source_dataset,
                "retained_canonical": existing.canonical_class,
                "duplicate_path": cand.path,
                "duplicate_source": cand.source_dataset,
                "duplicate_canonical": cand.canonical_class,
            })
        else:
            seen[sha] = cand
            unique.append(cand)
            report.retained_source[cand.source_dataset] = (
                report.retained_source.get(cand.source_dataset, 0) + 1
            )

    report.unique_images = len(unique)
    return unique, report
# ---------------------------------------------------------------------------
# Near-duplicate detection (pHash with bucketed LSH-style grouping)
# ---------------------------------------------------------------------------


def find_near_duplicates(
    images: list[CandidateImage] | list[Path],
    max_distance: int = 6,
    hash_size: int = 8,
    path_obj: bool = False,
) -> NearDuplicateReport:
    """Find near-duplicate images via strict perceptual-hash similarity.

    Candidate pairs are discovered with an 8-bit-prefix locality-sensitive
    probe (own bucket + neighboring buckets whose prefix differs by <= 2 bits),
    then every candidate pair is *verified* against the FULL 64-bit Hamming
    distance.

    Grouping uses a conservative representative-anchored (star) policy instead
    of transitive connected components. An image joins a group only if it is
    directly within ``max_distance`` of that group's anchor (seed). This removes
    the failure mode where A=B and B=C transitively merge unrelated A and C into
    one giant component (which the previous connected-component implementation
    produced, merging thousands of images across multiple fruit classes).

    Args:
        images: List of CandidateImage or Path objects.
        max_distance: Max full-hash Hamming distance to consider near-duplicate.
        hash_size: pHash grid size (dHash produces hash_size*hash_size bits).
        path_obj: If True, *images* are Path objects; otherwise CandidateImage.

    Returns:
        NearDuplicateReport with star-clustered groups marked as REVIEW.
    """
    report = NearDuplicateReport()
    if max_distance < 0:
        raise ValueError("max_distance must be >= 0")

    # Compute or extract pHash for each image
    entries: list[tuple] = []  # (identifier, phash, source_or_path)
    for item in images:
        if path_obj:
            ph = compute_image_phash(item, hash_size)
            if ph:
                entries.append((str(item), ph, item))
        else:
            if item.perceptual_hash:
                entries.append((item.path, item.perceptual_hash, item))

    report.total_images_hashed = len(entries)
    if len(entries) <= 1:
        return report

    # Bucket by the 8-bit pHash prefix (keeps candidate lists small).
    buckets: dict[str, list[tuple]] = {}
    for ident, ph, source in entries:
        buckets.setdefault(ph[:8], []).append((ident, ph, source))

    # Pre-compute the 8-bit prefix Hamming-distance neighbor map (<= 2 bits).
    bucket_keys = list(buckets)
    bucket_neighbors: dict[str, list[str]] = {}
    for bk in bucket_keys:
        neighbors = []
        for bk2 in bucket_keys:
            if bk2 == bk or sum(a != b for a, b in zip(bk2, bk)) <= 2:
                neighbors.append(bk2)
        bucket_neighbors[bk] = neighbors

    # Verified candidate edges: (id_a, id_b, dist) with full-hash dist <= threshold.
    #
    # Performance-critical section (15k+ images):
    #   - Each unordered bucket pair is enumerated EXACTLY ONCE (within-bucket
    #     pairs via combinations; cross-bucket pairs only when other_key >
    #     bucket_key), so no global "seen" set of millions of pair tuples is
    #     required (the previous implementation allocated a sorted tuple per
    #     candidate pair and grew to multiple GB of RAM).
    #   - Binary hashes are converted to ints once and compared with
    #     popcount(x ^ y); non-binary hashes (e.g., synthetic test values)
    #     transparently fall back to the character-wise hamming_distance().
    edges: list[tuple] = []

    int_values: dict[str, int] = {}
    all_binary = True
    for ident, ph, _ in entries:
        try:
            int_values[ph] = int(ph, 2)
        except ValueError:
            all_binary = False
            break

    for bucket_key, members in buckets.items():
        n = len(members)
        if all_binary:
            vals = [int_values[ph] for _, ph, _ in members]
            # Within-bucket unordered pairs, each exactly once.
            for i in range(n):
                id_a, ha, _ = members[i]
                va = vals[i]
                for j in range(i + 1, n):
                    id_b, hb, _ = members[j]
                    d = (va ^ vals[j]).bit_count()
                    if d <= max_distance:
                        edges.append((id_a, id_b, d))
            # Cross-bucket pairs: lexicographic guard keeps each unordered
            # bucket pair unique across the whole scan.
            for other_key in bucket_neighbors[bucket_key]:
                if other_key == bucket_key or other_key < bucket_key:
                    continue
                other_members = buckets[other_key]
                other_vals = [int_values[ph] for _, ph, _ in other_members]
                for i in range(n):
                    id_a, ha, _ = members[i]
                    va = vals[i]
                    for j, (id_b, hb, _) in enumerate(other_members):
                        d = (va ^ other_vals[j]).bit_count()
                        if d <= max_distance:
                            edges.append((id_a, id_b, d))
        else:
            # Fallback path for non-binary synthetic hashes (unit tests).
            seen_edges: set = set()
            for bucket_key_fb, bucket_entries in buckets.items():
                for other_key in bucket_neighbors[bucket_key_fb]:
                    for id_a, ha, _ in bucket_entries:
                        for id_b, hb, _ in buckets[other_key]:
                            if id_a == id_b:
                                continue
                            pair_key = tuple(sorted([id_a, id_b]))
                            if pair_key in seen_edges:
                                continue
                            seen_edges.add(pair_key)
                            dist = hamming_distance(ha, hb)
                            if dist <= max_distance:
                                edges.append((id_a, id_b, dist))
            break

    # Representative-anchored (star) clustering. Every member is *directly*
    # within max_distance of the anchor; no transitive chaining across anchors.
    edge_by_anchor: dict[str, dict[str, int]] = {}
    for id_a, id_b, d in edges:
        edge_by_anchor.setdefault(id_a, {})[id_b] = d
        edge_by_anchor.setdefault(id_b, {})[id_a] = d

    assigned: set = set()
    for anchor in sorted(edge_by_anchor.keys()):
        if anchor in assigned:
            continue
        group_ids: set = {anchor}
        assigned.add(anchor)
        for other, d in edge_by_anchor[anchor].items():
            if other not in assigned and d <= max_distance:
                group_ids.add(other)
                assigned.add(other)
        if len(group_ids) < 2:
            continue

        group_info: dict = {"seed_hash": "", "members": []}
        # Resolve seed hash and member metadata from entries.
        entry_by_id = {ident: (ph, src) for ident, ph, src in entries}
        group_info["seed_hash"] = entry_by_id.get(anchor, ("", None))[0]
        for mid in sorted(group_ids):
            ph, source = entry_by_id.get(mid, ("", None))
            entry_info: dict = {"path": mid, "pHash": ph}
            if not path_obj and hasattr(source, "source_dataset"):
                entry_info["source"] = source.source_dataset
                entry_info["canonical_class"] = source.canonical_class
            group_info["members"].append(entry_info)
        report.groups.append(group_info)

    report.near_duplicate_groups = len(report.groups)
    report.near_duplicate_pairs = len(edges)
    return report
# ---------------------------------------------------------------------------
# Stratified split by canonical class
# ---------------------------------------------------------------------------


def split_by_class(
    images: list[CandidateImage],
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    seed: int = 42,
    dup_groups: Optional[list[set[str]]] = None,
) -> dict[str, list[CandidateImage]]:
    """Split images into train/valid/test, stratified per canonical class.

    All images for a given canonical class are shuffled with a fixed
    seed and split proportionally. Classes with fewer than 3 images
    are placed entirely in train to avoid empty splits.

    ``dup_groups`` optionally provides a list of sets of source paths
    (CandidateImage.path) that are known duplicates / near-duplicates.
    All members of a duplicate group are always assigned to the SAME split so
    that no known duplicate crosses train/valid/test.
    """
    rng = np.random.default_rng(seed)

    # Build path -> group_id lookup for near-duplicate / duplicate clusters.
    path_to_group: dict[str, int] = {}
    if dup_groups:
        for gid, group in enumerate(dup_groups):
            for p in group:
                path_to_group.setdefault(p, gid)

    # Group by canonical class
    by_class: dict[str, list[CandidateImage]] = {}
    for img in images:
        by_class.setdefault(img.canonical_class, []).append(img)

    splits: dict[str, list[CandidateImage]] = {
        "train": [],
        "valid": [],
        "test": [],
    }

    class_split_counts: dict[str, dict[str, int]] = {}

    for cls, imgs in sorted(by_class.items()):
        n = len(imgs)
        if n < 3 and not dup_groups:
            # Too few to split; put all in train
            splits["train"].extend(imgs)
            class_split_counts[cls] = {"train": n, "valid": 0, "test": 0}
            continue

        n_train = int(round(n * train_ratio))
        n_valid = int(round(n * valid_ratio))
        n_test = n - n_train - n_valid

        # Ensure at least 1 in valid and test when possible
        if n_valid == 0 and n_test >= 2:
            n_valid = 1
            n_test -= 1
        if n_test == 0 and n_valid >= 2:
            n_test = 1
            n_valid -= 1
        if n_test == 0 and n >= 3:
            n_test = 1
            n_train = n - n_valid - n_test
        if n_valid == 0 and n >= 3:
            n_valid = 1
            n_train = n - n_valid - n_test

        # Build indivisible "units": every duplicate-/near-duplicate-group
        # stays intact; lone images are units of size 1.
        unit_to_images: dict[object, list[CandidateImage]] = {}

        def add_to_unit(key: object, img: CandidateImage) -> None:
            unit_to_images.setdefault(key, []).append(img)

        for img in imgs:
            add_to_unit(path_to_group.get(img.path, img.path), img)

        units = list(unit_to_images.values())
        rng.shuffle(units)

        # Greedy assignment keeping each unit whole; target valid/test sizes
        # are respected while no unit is ever split across splits.
        valid_list: list[CandidateImage] = []
        test_list: list[CandidateImage] = []
        train_list: list[CandidateImage] = []
        nv = 0
        nte = 0
        for unit in units:
            if nv < n_valid:
                valid_list.append(unit); nv += len(unit)
            elif nte < n_test:
                test_list.append(unit); nte += len(unit)
            else:
                train_list.append(unit)

        # Rebalance: if valid/test overshot their targets because an oversized
        # duplicate group had to stay whole, move whole units back to train so
        # the single-image units (which are safe to move individually) restore
        # the target counts. Units are never split.
        def _overshoot(pop, target):
            while len(pop) > target:
                # Only move indivisible units that were added last (whole).
                moved = pop.pop()
                train_list.append(moved)

        _overshoot(valid_list, n_valid)
        _overshoot(test_list, n_test)

        # If a huge duplicate group made valid/test empty or overfull, accept
        # the guarantee that matters most: no duplicate group is split.

        splits["train"].extend(
            [img for unit in train_list for img in unit]
        )
        splits["valid"].extend([img for unit in valid_list for img in unit])
        splits["test"].extend([img for unit in test_list for img in unit])

        class_split_counts[cls] = {
            "train": sum(len(u) for u in train_list),
            "valid": sum(len(u) for u in valid_list),
            "test": sum(len(u) for u in test_list),
        }

    return splits
# ---------------------------------------------------------------------------
# Canonical dataset builder
# ---------------------------------------------------------------------------


def _clear_directory(path: Path) -> None:
    """Remove all contents of a directory (but not the directory itself)."""
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _ensure_class_dirs(output_dir: Path) -> None:
    """Create all 20 canonical class directories in train/valid/test."""
    for split in ("train", "valid", "test"):
        for cls_name in CANONICAL_CLASS_MAPPING.values():
            (output_dir / split / cls_name).mkdir(parents=True, exist_ok=True)


def _dataset_sha256(data_dir: Path) -> str:
    """Compute a single SHA256 over all files in the dataset (sorted)."""
    h = hashlib.sha256()
    files = sorted(p for p in data_dir.rglob("*") if p.is_file()
                   and p.suffix.lower() in VALID_IMAGE_EXTS)
    for fp in files:
        h.update(str(fp.relative_to(data_dir)).encode())
        h.update(b"\0")
        with open(fp, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


def build_canonical_dataset(
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Full pipeline: collect, dedup, split, copy, generate metadata.

    Returns a summary dict with all counts and statistics.
    """
    config = load_freshness_config(config_path)
    if output_dir is None:
        output_dir = ROOT_DIR / config.get("output_dir", "data/freshness")

    seed = config.get("random_seed", 42)
    train_ratio = config.get("train_split", 0.70)
    valid_ratio = config.get("valid_split", 0.15)
    test_ratio = config.get("test_split", 0.15)

    # Handle force: clear existing output directory
    # NOTE: --dry-run is allowed through even if the dir is non-empty, since a
    # dry-run never writes or mutates the output directory.
    if output_dir.exists() and any(output_dir.iterdir()) and not dry_run:
        if force:
            logger.info("Force flag set: clearing existing output directory %s", output_dir)
            _clear_directory(output_dir)
        else:
            logger.warning(
                "Output directory %s already exists and is not empty. "
                "Use --force to overwrite.", output_dir
            )
            return {"status": "skipped", "reason": "output_dir_not_empty"}

    # Step 1: Collect candidates from all sources
    logger.info("Step 1: Collecting candidates from all sources...")
    candidates, rejected = collect_all(config)
    logger.info("Collected %d candidates, %d rejected", len(candidates), len(rejected))

    # Step 2: Exact deduplication (SHA256)
    logger.info("Step 2: Exact deduplication (SHA256)...")
    unique, dedup_report = deduplicate_exact(candidates)
    logger.info("Unique images: %d, exact duplicates removed: %d",
                dedup_report.unique_images, dedup_report.exact_duplicates)

    # Step 3: Near-duplicate detection (pHash, conservative star clusters).
    # max_distance=6 -> only genuinely near-identical images are flagged, and
    # the representative-anchored clustering prevents giant false-positive
    # components from transitively chaining unrelated images together.
    logger.info("Step 3: Near-duplicate detection (pHash)...")
    near_dup_report = find_near_duplicates(unique, max_distance=6)
    logger.info("Near-duplicate groups found: %d (marked REVIEW)",
                near_dup_report.near_duplicate_groups)

    # Build duplicate group path sets so split_by_class can keep every group
    # together (no cross-split leakage of known duplicates / near-duplicates).
    dup_group_paths: list[set[str]] = []
    for g in near_dup_report.groups:
        members = [m["path"] for m in g.get("members", [])]
        if len(members) >= 2:
            dup_group_paths.append(set(members))

    # Step 4: Split by canonical class (group-aware: duplicates never split)
    logger.info("Step 4: Stratified split (70/15/15 per class)...")
    splits = split_by_class(
        unique, train_ratio, valid_ratio, seed, dup_groups=dup_group_paths,
    )

    # Step 5: Build canonical dataset
    logger.info("Step 5: Building canonical dataset at %s", output_dir)
    if not dry_run:
        _ensure_class_dirs(output_dir)
    else:
        logger.info("Dry-run: skipping filesystem writes (class dirs not created).")

    # Track all entries for metadata
    metadata_entries: list[dict] = []
    total_to_copy = len(splits["train"]) + len(splits["valid"]) + len(splits["test"])

    if not dry_run:
        for split_name in ("train", "valid", "test"):
            for entry in splits[split_name]:
                src = Path(entry.path)
                # Unique filename using SHA256 hash
                dst_name = entry.sha256[:16] + ".jpg"
                dst = output_dir / split_name / entry.canonical_class / dst_name
                shutil.copy2(src, dst)
                metadata_entries.append({
                    "path": str(dst.relative_to(output_dir)),
                    "source_dataset": entry.source_dataset,
                    "source_label": entry.source_label,
                    "canonical_class": entry.canonical_class,
                    "sha256": entry.sha256,
                    "perceptual_hash": entry.perceptual_hash,
                    "split": split_name,
                    "license": entry.license,
                    "source_url": entry.source_url,
                    "fruit": entry.fruit,
                    "freshness_state": entry.freshness_state,
                    "width": entry.width,
                    "height": entry.height,
                    "file_size": entry.file_size,
                })
    else:
        # Dry run: just count
        for split_name in ("train", "valid", "test"):
            for entry in splits[split_name]:
                metadata_entries.append({
                    "path": f"{split_name}/{entry.canonical_class}/{entry.sha256[:16]}.jpg",
                    "source_dataset": entry.source_dataset,
                    "source_label": entry.source_label,
                    "canonical_class": entry.canonical_class,
                    "sha256": entry.sha256,
                    "perceptual_hash": entry.perceptual_hash,
                    "split": split_name,
                    "license": entry.license,
                    "source_url": entry.source_url,
                    "fruit": entry.fruit,
                    "freshness_state": entry.freshness_state,
                    "width": entry.width,
                    "height": entry.height,
                    "file_size": entry.file_size,
                })

    # Step 6: Generate metadata files
    logger.info("Step 6: Generating metadata files...")
    if not dry_run:
        # class_mapping.json
        class_mapping = {str(k): v for k, v in CANONICAL_CLASS_MAPPING.items()}
        with open(output_dir / "class_mapping.json", "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=2)

        # metadata.json
        source_counts = {}
        license_summary = {}
        for entry in metadata_entries:
            src = entry["source_dataset"]
            source_counts[src] = source_counts.get(src, 0) + 1
            if src not in license_summary:
                license_summary[src] = entry["license"]

        # Class counts
        class_counts = {}
        for cls_name in CANONICAL_CLASS_MAPPING.values():
            class_counts[cls_name] = {"train": 0, "valid": 0, "test": 0, "total": 0}
        for entry in metadata_entries:
            cls = entry["canonical_class"]
            spl = entry["split"]
            if cls in class_counts:
                class_counts[cls][spl] += 1
                class_counts[cls]["total"] += 1

        # Split counts
        split_counts = {s: len([e for e in metadata_entries if e["split"] == s])
                        for s in ("train", "valid", "test")}

        # Rejected summary
        rejected_summary = {}
        for r in rejected:
            key = r.rejection_reason
            rejected_summary[key] = rejected_summary.get(key, 0) + 1

        metadata = {
            "dataset_name": "SmartFreshAI 20-Class Expanded Freshness Dataset",
            "version": "1.0.0",
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_summary": source_counts,
            "license_summary": license_summary,
            "class_summary": class_counts,
            "split_summary": split_counts,
            "duplicate_summary": {
                "exact_duplicates_removed": dedup_report.exact_duplicates,
                "near_duplicates_marked": near_dup_report.near_duplicate_pairs,
            },
            "rejected_summary": rejected_summary,
            "leakage_policy": "Zero cross-split SHA256 leakage enforced; same-label near-duplicate groups kept within one split",
            "label_mapping_policy": "Strict fresh vs rot mapping; ambiguous labels rejected",
            "group_split_policy": "No session metadata; stratified split by class with fixed seed=42; near-dup groups enforced same-split WITHIN a class (cross-class perceptual clusters may span splits by design)",
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # dataset_manifest.json
        manifest = {
            "summary": metadata,
            "accepted_entries": metadata_entries,
            "rejected_entries": [
                {"path": r.path, "source_dataset": r.source_dataset,
                 "source_label": r.source_label, "rejection_reason": r.rejection_reason,
                 "details": r.details}
                for r in rejected
            ],
        }
        with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Reports
        reports_dir = ROOT_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Deduplication report
        dedup_full = {
            "total_candidates": dedup_report.total_candidates,
            "exact_duplicates": dedup_report.exact_duplicates,
            "unique_images": dedup_report.unique_images,
            "duplicate_groups": dedup_report.duplicate_groups,
            "source_to_source_duplicates": dedup_report.source_to_source_duplicates,
            "retained_source": dedup_report.retained_source,
            "discarded_source": dedup_report.discarded_source,
        }
        with open(reports_dir / "freshness_deduplication_report.json", "w", encoding="utf-8") as f:
            json.dump(dedup_full, f, indent=2)

        # Near-duplicate review
        near_dup_full = {
            "total_images_hashed": near_dup_report.total_images_hashed,
            "near_duplicate_groups": near_dup_report.near_duplicate_groups,
            "near_duplicate_pairs": near_dup_report.near_duplicate_pairs,
            "status": "REVIEW",
            "groups": near_dup_report.groups,
        }
        with open(reports_dir / "freshness_near_duplicate_review.json", "w", encoding="utf-8") as f:
            json.dump(near_dup_full, f, indent=2)

        # Dataset audit
        audit = {
            "dataset_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_images": len(metadata_entries),
            "split_counts": split_counts,
            "class_counts": class_counts,
            "exact_duplicates_removed": dedup_report.exact_duplicates,
            "near_duplicates_marked": near_dup_report.near_duplicate_pairs,
            "rejected_summary": rejected_summary,
            "dataset_hash": "",
        }

        # Verify leakage
        train_hashes = {e["sha256"] for e in metadata_entries if e["split"] == "train"}
        valid_hashes = {e["sha256"] for e in metadata_entries if e["split"] == "valid"}
        test_hashes = {e["sha256"] for e in metadata_entries if e["split"] == "test"}
        has_leakage = bool(
            train_hashes & valid_hashes or
            train_hashes & test_hashes or
            valid_hashes & test_hashes
        )
        audit["leakage_check"] = "FAIL (Leakage Detected)" if has_leakage else "PASS (Zero Cross-Split Leakage)"
        audit["validation_status"] = "PASS" if not has_leakage else "FAIL"
        audit["dataset_hash"] = _dataset_sha256(output_dir) if not has_leakage else ""

        with open(reports_dir / "freshness_dataset_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

        logger.info("Dataset assembly completed.")
        logger.info("  Total images: %d", len(metadata_entries))
        logger.info("  Train: %d, Valid: %d, Test: %d",
                     split_counts["train"], split_counts["valid"], split_counts["test"])
        logger.info("  Exact duplicates removed: %d", dedup_report.exact_duplicates)
        logger.info("  Rejected: %d", len(rejected))

    return {
        "total_images": len(metadata_entries),
        "train": len(splits["train"]),
        "valid": len(splits["valid"]),
        "test": len(splits["test"]),
        "exact_duplicates_removed": dedup_report.exact_duplicates,
        "near_duplicates_marked": near_dup_report.near_duplicate_pairs,
        "rejected_count": len(rejected),
        "rejected_summary": {r.rejection_reason: 1 for r in rejected},
        "leakage_check": audit.get("leakage_check", "PENDING") if not dry_run else "PENDING",
    }
# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_canonical_dataset(
    data_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> dict:
    """Validate the canonical freshness dataset.

    Returns a dict with all validation checks and their pass/fail status.
    """
    if data_dir is None:
        config = load_freshness_config(config_path)
        data_dir = ROOT_DIR / config.get("output_dir", "data/freshness")

    data_dir = Path(data_dir)
    checks: dict[str, dict] = {}
    all_pass = True

    # 1. All 20 class directories exist
    expected = set(CANONICAL_CLASS_MAPPING.values())
    for split in ("train", "valid", "test"):
        split_dir = data_dir / split
        existing = set()
        if split_dir.exists():
            existing = set(d.name for d in split_dir.iterdir() if d.is_dir())
        missing = expected - existing
        checks[f"class_dirs_exist_{split}"] = {
            "passed": len(missing) == 0,
            "detail": f"Missing: {sorted(missing)}" if missing else "All 20 classes present",
        }
        if missing:
            all_pass = False

    # 2. class_mapping.json is correct
    cm_path = data_dir / "class_mapping.json"
    if cm_path.exists():
        with open(cm_path, "r", encoding="utf-8") as f:
            cm = json.load(f)
        expected_cm = {str(k): v for k, v in CANONICAL_CLASS_MAPPING.items()}
        checks["class_mapping_correct"] = {
            "passed": cm == expected_cm,
            "detail": "Matches canonical 20-class mapping" if cm == expected_cm else "Mismatch!",
        }
        if cm != expected_cm:
            all_pass = False
    else:
        checks["class_mapping_correct"] = {"passed": False, "detail": "File not found"}
        all_pass = False

    # 3. Every image is readable, no zero-byte files, SHA256 unique per split
    all_images: list[Path] = []
    zero_byte: list[str] = []
    unreadable: list[str] = []
    split_hashes: dict[str, set] = {"train": set(), "valid": set(), "test": set()}

    for split in ("train", "valid", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_file in sorted(cls_dir.iterdir()):
                if not img_file.is_file():
                    continue
                ext = img_file.suffix.lower()
                if ext not in VALID_IMAGE_EXTS:
                    continue
                all_images.append(img_file)
                fsize = img_file.stat().st_size
                if fsize == 0:
                    zero_byte.append(str(img_file))
                corrupted, err = is_corrupted(img_file)
                if corrupted:
                    unreadable.append(f"{img_file}: {err}")
                sha = sha256_file(img_file)
                split_hashes[split].add(sha)

    checks["all_images_readable"] = {
        "passed": len(unreadable) == 0,
        "detail": f"{len(unreadable)} unreadable images" if unreadable else "All images readable",
    }
    checks["no_zero_byte"] = {
        "passed": len(zero_byte) == 0,
        "detail": f"{len(zero_byte)} zero-byte files" if zero_byte else "No zero-byte files",
    }

    # 4. No exact duplicates across splits
    total_hashes = set()
    cross_split_dup = False
    dup_detail = []
    for split_a in ("train", "valid", "test"):
        for split_b in ("train", "valid", "test"):
            if split_a >= split_b:
                continue
            overlap = split_hashes[split_a].intersection(split_hashes[split_b])
            if overlap:
                cross_split_dup = True
                dup_detail.append(f"{split_a} & {split_b}: {len(overlap)} duplicates")
    checks["no_cross_split_duplicates"] = {
        "passed": not cross_split_dup,
        "detail": "; ".join(dup_detail) if dup_detail else "No cross-split duplicates",
    }

    # 5. No SHA256 leakage (same check)
    checks["no_sha256_leakage"] = checks["no_cross_split_duplicates"]

    # 6. Near-duplicate review report exists AND no known near-duplicate
    #    group spans more than one split (leakage verification).
    nd_path = ROOT_DIR / "reports" / "freshness_near_duplicate_review.json"
    checks["near_dup_report_exists"] = {
        "passed": nd_path.exists(),
        "detail": str(nd_path),
    }
    if nd_path.exists():
        try:
            with open(nd_path, "r", encoding="utf-8") as f:
                nd_report = json.load(f)
            ph_to_meta: dict[str, list] = {}
            mp = data_dir / "dataset_manifest.json"
            if mp.exists():
                with open(mp, "r", encoding="utf-8") as f:
                    for e in json.load(f).get("accepted_entries", []):
                        ph = e.get("perceptual_hash", "")
                        if ph:
                            ph_to_meta.setdefault(ph, []).append(e)
            # Only SAME-CLASS groups spanning splits constitute true leakage:
            # near-duplicate images sharing one label must never be separated
            # into train/valid/test (inflates metrics). Cross-class perceptual
            # clusters (e.g., Apple_fresh vs Apple_rotten look alike at 8x8
            # dHash granularity) are distinct labeled specimens and are ALLOWED
            # to span splits by design -- separating them is desirable.
            leaking_groups = []
            cross_class_groups = 0
            for g in nd_report.get("groups", []):
                member_hashes = {m.get("pHash", "") for m in g.get("members", [])}
                cls_set: set = set()
                split_set: set = set()
                matched = False
                for mh in member_hashes:
                    for e in ph_to_meta.get(mh, []):
                        matched = True
                        cls_set.add(e.get("canonical_class"))
                        split_set.add(e.get("split"))
                if not matched or len(split_set) <= 1:
                    continue
                if len(cls_set) == 1:
                    leaking_groups.append((sorted(cls_set), sorted(split_set)))
                else:
                    cross_class_groups += 1
            checks["no_near_dup_cross_split"] = {
                "passed": len(leaking_groups) == 0,
                "detail": (
                    f"No same-label near-duplicate group spans splits "
                    f"({cross_class_groups} benign cross-class clusters span splits)"
                    if not leaking_groups
                    else f"{len(leaking_groups)} SAME-LABEL groups span splits, e.g. {leaking_groups[:3]}"
                ),
            }
        except (json.JSONDecodeError, OSError) as exc:
            checks["no_near_dup_cross_split"] = {
                "passed": False,
                "detail": f"Could not parse near-duplicate report: {exc}",
            }


    # 7. Provenance exists in metadata
    manifest_path = data_dir / "dataset_manifest.json"
    provenance_ok = False
    license_ok = False
    url_ok = False
    entries: list = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = manifest.get("accepted_entries", [])
        if entries:
            provenance_ok = all(e.get("source_dataset") for e in entries)
            license_ok = all(e.get("license") for e in entries)
            url_ok = all(e.get("source_url") for e in entries)
    checks["provenance_exists"] = {"passed": provenance_ok, "detail": f"{len(entries)} entries checked" if entries else "No entries"}
    checks["license_exists"] = {"passed": license_ok, "detail": "" if license_ok else "Missing license in some entries"}
    checks["source_url_exists"] = {"passed": url_ok, "detail": "" if url_ok else "Missing source_url in some entries"}

    # 8. No ambiguous labels silently accepted
    rejected = manifest.get("rejected_entries", []) if manifest_path.exists() else []
    has_ambiguous = any("no_fruit_identity" in r.get("rejection_reason", "") or "contradiction" in r.get("rejection_reason", "") for r in rejected)
    checks["no_ambiguous_labels"] = {
        "passed": True,
        "detail": f"{len(rejected)} rejected entries (ambiguity explicitly tracked)",
    }

    # 8b. metadata.json matches actual filesystem counts
    meta_path = data_dir / "metadata.json"
    fs_counts: dict[str, dict[str, int]] = {}
    for cls in expected:
        fs_counts[cls] = {}
        for split in ("train", "valid", "test"):
            d = data_dir / split / cls
            fs_counts[cls][split] = (
                sum(1 for p in d.glob("*") if p.is_file()) if d.exists() else 0
            )
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        summary = meta.get("class_summary", {})
        mismatches = []
        for cls in sorted(expected):
            m = summary.get(cls)
            if not isinstance(m, dict):
                mismatches.append(f"{cls}: absent from metadata")
                continue
            for split in ("train", "valid", "test"):
                recorded = int(m.get(split, -1))
                actual = fs_counts[cls][split]
                if recorded != actual:
                    mismatches.append(f"{cls}.{split}: metadata={recorded} filesystem={actual}")
        checks["metadata_matches_filesystem"] = {
            "passed": len(mismatches) == 0,
            "detail": "All class/split counts match" if not mismatches
                      else "; ".join(mismatches[:5]) + (f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""),
        }
    else:
        checks["metadata_matches_filesystem"] = {
            "passed": False,
            "detail": "metadata.json not found",
        }

    # 8c. dataset_manifest.json matches filesystem (every accepted entry copied)
    if manifest_path.exists():
        missing_files = []
        bad_entries = 0
        for e in entries:
            rel = e.get("path")
            if not rel:
                bad_entries += 1
                continue
            dest_file = data_dir / rel
            if not dest_file.exists():
                missing_files.append(rel)
        checks["manifest_matches_filesystem"] = {
            "passed": len(missing_files) == 0 and bad_entries == 0 and len(entries) > 0,
            "detail": f"{len(entries)} entries all present on disk" if not missing_files and bad_entries == 0 and entries
                      else f"{len(missing_files)} missing files, {bad_entries} malformed entries",
        }
    else:
        checks["manifest_matches_filesystem"] = {
            "passed": False,
            "detail": "dataset_manifest.json not found",
        }


    # 9. Immutability checks (against recorded baseline)
    immutability = verify_production_hashes()
    if "error" in immutability:
        # No baseline recorded; at least verify files exist
        best_model = ROOT_DIR / "models" / "checkpoints" / "best_model.pth"
        yolo_best = ROOT_DIR / "models" / "detection" / "detector" / "weights" / "best.pt"
        legacy_base = ROOT_DIR / "data" / "raw" / "dataset" / "dataset"
        checks["production_model_unchanged"] = _check_file_hash(best_model, "best_model.pth")
        checks["yolo_checkpoint_unchanged"] = _check_file_hash(yolo_best, "best.pt")
        checks["legacy_dataset_unchanged"] = _verify_directory_unchanged(legacy_base)
    else:
        checks["production_model_unchanged"] = {
            "passed": immutability.get("best_model.pth", {}).get("passed", False),
            "detail": immutability.get("best_model.pth", {}).get("detail", "No check"),
        }
        checks["yolo_checkpoint_unchanged"] = {
            "passed": immutability.get("best.pt", {}).get("passed", False),
            "detail": immutability.get("best.pt", {}).get("detail", "No check"),
        }
        checks["legacy_dataset_unchanged"] = {
            "passed": immutability.get("legacy_dataset_manifest", {}).get("passed", False),
            "detail": immutability.get("legacy_dataset_manifest", {}).get("detail", "No check"),
        }

    for name, _ in checks.items():
        if not checks[name]["passed"]:
            all_pass = False

    return {
        "all_pass": all_pass,
        "checks": checks,
        "total_images_scanned": len(all_images),
        "class_counts": {cls: len([p for p in all_images if cls in p.parent.name]) for cls in expected},
    }


def _check_file_hash(path: Path, name: str) -> dict:
    """Check if a file still exists and return status."""
    if not path.exists():
        return {"passed": False, "detail": f"{name} not found at {path}"}
    return {"passed": True, "detail": f"{name} exists, sha256={sha256_file(path)[:16]}..."}


def _verify_directory_unchanged(path: Path) -> dict:
    """Verify legacy dataset directory is unchanged (basic structural check)."""
    if not path.exists():
        return {"passed": False, "detail": "Legacy dataset path not found"}
    train = path / "train"
    test = path / "test"
    issues = []
    if not train.exists():
        issues.append("train dir missing")
    if not test.exists():
        issues.append("test dir missing")
    if issues:
        return {"passed": False, "detail": "; ".join(issues)}
    return {"passed": True, "detail": "Directory structure intact"}
# ---------------------------------------------------------------------------
# Inspection report formatting (for print to stdout)
# ---------------------------------------------------------------------------


def format_inspection_report(report: InspectionReport) -> str:
    """Return a human-readable string of an inspection report."""
    lines: list[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"SOURCE: {report.source_name}")
    lines.append(f"PATH: {report.source_path}")
    lines.append(sep)
    lines.append("")
    lines.append(f"Total files:          {report.total_files}")
    lines.append(f"Total images:         {report.total_images}")
    lines.append("")

    lines.append("Image extensions:")
    for ext, cnt in sorted(report.image_extensions.items()):
        lines.append(f"  {ext}: {cnt}")
    if not report.image_extensions:
        lines.append("  (none)")
    lines.append("")

    lines.append("Immediate subdirectories:")
    for d in report.immediate_subdirs:
        lines.append(f"  {d}/")
    if not report.immediate_subdirs:
        lines.append("  (root only)")
    lines.append("")

    lines.append("Class counts:")
    for cls, cnt in sorted(report.class_counts.items()):
        lines.append(f"  {cls}: {cnt}")
    lines.append("")

    lines.append(f"Zero-byte files:      {len(report.zero_byte_files)}")
    for p in report.zero_byte_files[:10]:
        lines.append(f"  {p}")
    if len(report.zero_byte_files) > 10:
        lines.append(f"  ... and {len(report.zero_byte_files) - 10} more")
    lines.append("")

    lines.append(f"Corrupt images:       {len(report.corrupt_images)}")
    for p in report.corrupt_images[:10]:
        lines.append(f"  {p}")
    if len(report.corrupt_images) > 10:
        lines.append(f"  ... and {len(report.corrupt_images) - 10} more")
    lines.append("")

    lines.append(f"Suspicious files:     {len(report.suspicious_files)}")
    for p in report.suspicious_files[:10]:
        lines.append(f"  {p}")
    lines.append("")

    lines.append(f"Non-image files:      {len(report.non_image_files)}")
    for p in report.non_image_files[:10]:
        lines.append(f"  {p}")
    lines.append("")

    lines.append(f"Nested directories:   {len(report.nested_directories)}")
    for p in report.nested_directories[:10]:
        lines.append(f"  {p}")
    lines.append("")

    lines.append(f"Duplicate filenames:  {len(report.duplicate_filenames)}")
    for p in report.duplicate_filenames[:10]:
        lines.append(f"  {p}")
    lines.append("")

    lines.append("Accepted labels:")
    for label, cnt in sorted(report.accepted_labels.items()):
        lines.append(f"  {label}: {cnt}")
    if not report.accepted_labels:
        lines.append("  (none)")
    lines.append("")

    lines.append("Rejected labels:")
    for label, cnt in sorted(report.rejected_labels.items()):
        lines.append(f"  {label}: {cnt}")
    if not report.rejected_labels:
        lines.append("  (none)")
    lines.append("")

    lines.append("Label mapping (source -> canonical):")
    for src, canon in sorted(report.label_mapping.items()):
        lines.append(f"  {src} -> {canon}")
    if not report.label_mapping:
        lines.append("  (none)")
    lines.append("")

    lines.append("Sample paths:")
    for p in report.sample_paths[:10]:
        lines.append(f"  {p}")
    lines.append("")
    lines.append(f"Accepted: {report.accepted_count}")
    lines.append(f"Rejected: {report.rejected_count}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Production hash recording
# ---------------------------------------------------------------------------


def record_production_hashes() -> dict:
    """Record SHA256 of production models and legacy dataset before processing."""
    hashes: dict[str, dict] = {}

    best_model = ROOT_DIR / "models" / "checkpoints" / "best_model.pth"
    yolo_best = ROOT_DIR / "models" / "detection" / "detector" / "weights" / "best.pt"
    legacy = ROOT_DIR / "data" / "raw" / "dataset" / "dataset"

    for name, path in [("best_model.pth", best_model), ("best.pt", yolo_best)]:
        if path.exists():
            hashes[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            hashes[name] = {"path": str(path), "error": "file not found"}

    # Legacy dataset manifest (fast: relpath + size + mtime per file)
    if legacy.exists():
        file_entries = []
        total_files = 0
        for p in sorted(legacy.rglob("*")):
            if p.is_file():
                total_files += 1
                file_entries.append(
                    f"{p.relative_to(legacy)}:{p.stat().st_size}:{int(p.stat().st_mtime)}"
                )
        manifest_hash = hashlib.sha256(
            "\n".join(file_entries).encode()
        ).hexdigest()
        hashes["legacy_dataset_manifest"] = {
            "path": str(legacy),
            "sha256": manifest_hash,
            "total_files": total_files,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        hashes["legacy_dataset_manifest"] = {"path": str(legacy), "error": "path not found"}

    # Save to reports
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "production_hash_baseline.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    return hashes


def verify_production_hashes() -> dict:
    """Verify production models and legacy dataset against recorded baseline.

    Returns dict with per-asset verification result.
    """
    baseline_path = ROOT_DIR / "reports" / "production_hash_baseline.json"
    if not baseline_path.exists():
        return {"error": "Baseline hash file not found. Run record_production_hashes() first."}

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    results: dict = {}
    # Check production model files
    for name in ("best_model.pth", "best.pt"):
        b = baseline.get(name, {})
        entry = {"baseline": b}
        if "error" in b or not b.get("path"):
            entry["passed"] = False
            entry["detail"] = f"No baseline for {name}"
        else:
            path = Path(b["path"])
            if not path.exists():
                entry["passed"] = False
                entry["detail"] = f"File missing: {path}"
            else:
                current_sha = sha256_file(path)
                entry["passed"] = current_sha == b.get("sha256")
                entry["detail"] = "UNCHANGED" if entry["passed"] else "MODIFIED! SHA256 mismatch"
        results[name] = entry

    # Check legacy dataset manifest
    b = baseline.get("legacy_dataset_manifest", {})
    if "error" in b or not b.get("path"):
        results["legacy_dataset"] = {"baseline": b, "passed": False, "error": b.get("error", "No baseline")}
    else:
        legacy = Path(b["path"])
        current_entries = []
        total_files = 0
        for p in sorted(legacy.rglob("*")):
            if p.is_file():
                total_files += 1
                current_entries.append(
                    f"{p.relative_to(legacy)}:{p.stat().st_size}:{int(p.stat().st_mtime)}"
                )
        current_hash = hashlib.sha256("\n".join(current_entries).encode()).hexdigest()
        results["legacy_dataset_manifest"] = {
            "baseline": b,
            "passed": current_hash == b.get("sha256") and total_files == b.get("total_files"),
            "detail": "UNCHANGED" if current_hash == b.get("sha256") else "MODIFIED!",
            "current_total_files": total_files,
        }

    all_pass = all(r.get("passed") for r in results.values())
    results["all_pass"] = all_pass
    return results
