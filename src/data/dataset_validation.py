"""Real-World Dataset Validation Toolkit — Phase 4A.

Deterministic, testable utilities for validating a webcam-collected fruit
dataset *before* any training occurs. Follows:

    MEASURE -> VALIDATE -> CLEAN -> SPLIT -> REPORT

The toolkit never modifies the original raw dataset. Rejected samples must be
moved explicitly by the operator.

Core capabilities:
- ``ImageInfo``: per-image geometry + quality metrics.
- ``DatasetScan``: aggregated scan of a folder tree.
- ``MetadataRecord``: parsed collection metadata from ``src/data/collection.py``.
- Duplicate detection: exact (MD5) and near (perceptual hash).
- Session-aware splitting (70/15/15 grouped by ``session_id`` when available).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "VALID_IMAGE_EXTS",
    "ImageQuality",
    "ImageInfo",
    "DatasetScan",
    "MetadataRecord",
    "DuplicatePair",
    "SplitResult",
    "scan_directory",
    "compute_image_info",
    "is_corrupted",
    "file_md5",
    "perceptual_hash",
    "hamming_distance",
    "find_exact_duplicates",
    "find_near_duplicates",
    "find_suspicious_groups",
    "load_metadata_dir",
    "parse_metadata_file",
    "split_by_session",
    "split_files",
    "group_files_by_session",
]

VALID_IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Quality thresholds (mirroring CollectionConfig defaults).
BLUR_THRESHOLD = 100.0
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220
CONTRAST_MIN = 20.0


@dataclass(frozen=True)
class ImageQuality:
    """Brightness / contrast / blur metrics for a single image."""

    brightness: float
    contrast: float
    blur_score: float  # Laplacian variance; higher = sharper.

    def is_dark(self, threshold: float = BRIGHTNESS_MIN) -> bool:
        return self.brightness < threshold

    def is_bright(self, threshold: float = BRIGHTNESS_MAX) -> bool:
        return self.brightness > threshold

    def is_blurry(self, threshold: float = BLUR_THRESHOLD) -> bool:
        return self.blur_score < threshold

    def is_low_contrast(self, threshold: float = CONTRAST_MIN) -> bool:
        return self.contrast < threshold


@dataclass(frozen=True)
class ImageInfo:
    """A validated image with geometry and quality metrics."""

    path: Path
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    quality: Optional[ImageQuality] = None
    readable: bool = False
    error: Optional[str] = None


@dataclass
class DatasetScan:
    """Result of scanning an image directory tree."""

    root: Path
    images: List[ImageInfo] = field(default_factory=list)
    unreadable: List[Tuple[Path, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.images)

    @property
    def readable(self) -> List[ImageInfo]:
        return [i for i in self.images if i.readable]

    def class_counts(self) -> Dict[str, int]:
        """Count images per class based on their parent folder name."""
        counts: Dict[str, int] = {}
        for img in self.images:
            cls = img.path.parent.name if img.path.parent != self.root else "unlabeled"
            counts[cls] = counts.get(cls, 0) + 1
        return counts


@dataclass(frozen=True)
class MetadataRecord:
    """Parsed collection metadata produced by ``src/data/collection.py``."""

    sample_id: str
    session_id: str
    timestamp: float
    image_path: str
    label: str
    predicted_class: Optional[str] = None
    predicted_confidence: Optional[float] = None
    detector_confidence: Optional[float] = None
    tracking_id: Optional[int] = None
    accepted: bool = True
    rejection_reason: Optional[str] = None
    raw: Dict = field(default_factory=dict, compare=False)

    REQUIRED_FIELDS = (
        "sample_id",
        "session_id",
        "timestamp",
        "image_path",
        "label",
    )

    @classmethod
    def from_dict(cls, data: Dict) -> "MetadataRecord":
        return cls(
            sample_id=str(data.get("sample_id", "")),
            session_id=str(data.get("session_id", "")),
            timestamp=float(data.get("timestamp", 0.0)),
            image_path=str(data.get("image_path", "")),
            label=str(data.get("label", "")),
            predicted_class=data.get("predicted_class"),
            predicted_confidence=(
                float(data["predicted_confidence"])
                if data.get("predicted_confidence") is not None
                else None
            ),
            detector_confidence=(
                float(data["detector_confidence"])
                if data.get("detector_confidence") is not None
                else None
            ),
            tracking_id=data.get("tracking_id"),
            accepted=bool(data.get("accepted", True)),
            rejection_reason=data.get("rejection_reason"),
            raw=data,
        )

    def missing_fields(self) -> List[str]:
        """Return required fields absent from the record."""
        missing = []
        for fname in self.REQUIRED_FIELDS:
            value = getattr(self, fname)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(fname)
        return missing


@dataclass(frozen=True)
class DuplicatePair:
    """A near/exact duplicate pair with a reason string."""

    path_a: Path
    path_b: Path
    kind: str  # "exact" or "near"
    similarity: float = 0.0


@dataclass(frozen=True)
class SplitResult:
    """Result of a (possibly session-aware) dataset split."""

    train_files: List[Path]
    val_files: List[Path]
    test_files: List[Path]
    by_session: bool
    session_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "train": len(self.train_files),
            "val": len(self.val_files),
            "test": len(self.test_files),
        }


# ---------------------------------------------------------------------------
# Image scanning / quality
# ---------------------------------------------------------------------------


def is_corrupted(path: Path) -> Tuple[bool, Optional[str]]:
    """Return ``(corrupted, error)`` for an image file."""
    if not path.exists():
        return True, "missing"
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return True, "cv2_decode_failed"
        if img.size == 0:
            return True, "zero_size"
    except Exception as exc:  # noqa: BLE001
        return True, f"cv2_exception:{exc}"
    return False, None


def compute_image_info(path: Path) -> ImageInfo:
    """Compute geometry + quality metrics for a single image."""
    if not path.exists():
        return ImageInfo(path=path, readable=False, error="missing")

    corrupted, err = is_corrupted(path)
    if corrupted:
        return ImageInfo(path=path, readable=False, error=err)

    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality = ImageQuality(brightness=brightness, contrast=contrast, blur_score=blur)
    return ImageInfo(
        path=path,
        width=w,
        height=h,
        aspect_ratio=(w / h if h > 0 else 0.0),
        quality=quality,
        readable=True,
    )


def scan_directory(directory: Path) -> DatasetScan:
    """Recursively scan a directory, computing info for every image file."""
    scan = DatasetScan(root=directory)
    if not directory.exists():
        return scan

    files = (
        f
        for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS
    )
    for path in sorted(files):
        info = compute_image_info(path)
        if not info.readable:
            scan.unreadable.append((path, info.error or "unknown"))
        scan.images.append(info)
    return scan


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file (streamed)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path, hash_size: int = 8) -> str:
    """Return a perceptual hash string (pHash) for an image.

    Returns "" if the image cannot be read.
    """
    try:
        img = Image.open(path).convert("L").resize((hash_size + 1, hash_size))
        pixels = np.asarray(img, dtype=np.int32)
        diff = pixels[1:, :] > pixels[:-1, :]
        return "".join("1" if d else "0" for d in diff.flatten())
    except Exception:  # noqa: BLE001
        return ""


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two equal-length hash strings."""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(ca != cb for ca, cb in zip(a, b))


def find_exact_duplicates(paths: Sequence[Path]) -> List[DuplicatePair]:
    """Find pairs of files sharing the same MD5 digest."""
    buckets: Dict[str, List[Path]] = {}
    for p in paths:
        try:
            buckets.setdefault(file_md5(p), []).append(p)
        except OSError:
            continue

    pairs: List[DuplicatePair] = []
    for digest in sorted(buckets):
        group = buckets[digest]
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pairs.append(
                        DuplicatePair(group[i], group[j], kind="exact", similarity=1.0)
                    )
    return pairs


def find_near_duplicates(
    paths: Sequence[Path],
    max_distance: int = 10,
) -> List[DuplicatePair]:
    """Find near-duplicate pairs via perceptual hash distance."""
    hashed: List[Tuple[Path, str]] = []
    for p in paths:
        ph = perceptual_hash(p)
        if ph:
            hashed.append((p, ph))

    pairs: List[DuplicatePair] = []
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            path_a, hash_a = hashed[i]
            path_b, hash_b = hashed[j]
            dist = hamming_distance(hash_a, hash_b)
            if dist <= max_distance:
                similarity = 1.0 - dist / max(len(hash_a), len(hash_b))
                pairs.append(
                    DuplicatePair(path_a, path_b, kind="near", similarity=similarity)
                )
    return pairs


def find_suspicious_groups(
    paths: Sequence[Path],
    max_distance: int = 10,
) -> List[Tuple[str, List[Path]]]:
    """Group near-duplicates into connected components (suspicious groups)."""
    pairs = find_near_duplicates(paths, max_distance=max_distance)
    adjacency: Dict[Path, Set[Path]] = {p: set() for p in paths}
    for pair in pairs:
        adjacency[pair.path_a].add(pair.path_b)
        adjacency[pair.path_b].add(pair.path_a)

    visited: Set[Path] = set()
    groups: List[List[Path]] = []
    for p in paths:
        if p in visited:
            continue
        stack = [p]
        group: List[Path] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            stack.extend(adjacency[cur] - visited)
        if len(group) >= 2:
            groups.append(sorted(group, key=lambda x: str(x)))

    result = []
    for g in groups:
        seed = ""
        for p in g:
            seed = perceptual_hash(p)
            if seed:
                break
        result.append((seed, g))
    return result


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------


def parse_metadata_file(path: Path) -> Optional[MetadataRecord]:
    """Parse a single JSON metadata file into a MetadataRecord.

    Returns None if the file is not valid JSON or lacks the required fields.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return MetadataRecord.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def load_metadata_dir(metadata_dir: Path) -> List[MetadataRecord]:
    """Load all per-sample metadata JSON files in a directory.

    Session-level ``session_*.json`` files are ignored (not per-sample).
    """
    records: List[MetadataRecord] = []
    if not metadata_dir.exists():
        return records
    for f in sorted(metadata_dir.glob("*.json")):
        if f.name.startswith("session_"):
            continue
        rec = parse_metadata_file(f)
        if rec is not None:
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------


def split_by_session(
    files_by_session: Dict[str, List[Path]],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Split *session IDs* into train/val/test session sets.

    Splitting whole sessions (not individual images) prevents frames from the
    same recording sequence leaking across splits.
    """
    session_ids = sorted(files_by_session.keys())
    if not session_ids:
        return set(), set(), set()

    rng = np.random.default_rng(seed)
    shuffled = list(session_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(round(n * train)))
    n_val = max(0, int(round(n * val)))
    n_test = max(0, n - n_train - n_val)

    train_s = set(shuffled[:n_train])
    val_s = set(shuffled[n_train:n_train + n_val])
    test_s = set(shuffled[n_train + n_val:])
    return train_s, val_s, test_s


def _group_files_by_label(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    """Group files by their parent folder name (label)."""
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        groups.setdefault(p.parent.name, []).append(p)
    return groups


def split_files(
    files: Sequence[Path],
    by_sessions: Optional[Dict[str, List[Path]]] = None,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Split image files into train/val/test.

    When ``by_sessions`` is provided the split is performed at the *session*
    level (recommended to prevent session leakage) and ``by_session=True``.
    Otherwise a label-stratified file-level split is used.
    """
    if by_sessions:
        train_s, val_s, test_s = split_by_session(
            by_sessions, train=train, val=val, test=test, seed=seed
        )
        train_files = [f for sid, fl in by_sessions.items() if sid in train_s for f in fl]
        val_files = [f for sid, fl in by_sessions.items() if sid in val_s for f in fl]
        test_files = [f for sid, fl in by_sessions.items() if sid in test_s for f in fl]
        return SplitResult(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            by_session=True,
            session_counts={
                "train": len(train_s),
                "val": len(val_s),
                "test": len(test_s),
            },
        )

    groups = _group_files_by_label(files)
    rng = np.random.default_rng(seed)
    train_files: List[Path] = []
    val_files: List[Path] = []
    test_files: List[Path] = []
    for cls, cls_files in groups.items():
        arr = np.array(sorted(cls_files))
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        train_files.extend(arr[:n_train])
        val_files.extend(arr[n_train:n_train + n_val])
        test_files.extend(arr[n_train + n_val:])
    return SplitResult(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        by_session=False,
    )


def group_files_by_session(
    images: Sequence[Path],
    session_lookup: Callable[[Path], Optional[str]],
) -> Dict[str, List[Path]]:
    """Group image files by their session id via a lookup callable.

    Files with no resolvable session id are excluded from grouping.
    """
    grouped: Dict[str, List[Path]] = {}
    for p in images:
        sid = session_lookup(p)
        if sid:
            grouped.setdefault(sid, []).append(p)
    return grouped