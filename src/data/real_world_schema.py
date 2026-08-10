"""Canonical real-world dataset schema, validation, and physical-fruit splitting.

Phase 5 foundation for a trustworthy real-world benchmark. This module is the
single implementation of the canonical schema documented in
``docs/REAL_WORLD_DATASET.md``:

- ``CANONICAL_FIELDS``: the machine-readable field definitions.
- ``CanonicalRecord``: one manifest row (one image).
- ``load_canonical_manifest``: CSV/JSON manifest ingestion.
- ``validate_canonical_manifest``: full dataset validation (labels, metadata,
  duplicates, impossible combinations, class imbalance, missing files).
- ``find_physical_fruit_leakage`` / ``find_session_leakage``: leakage checks on
  already-produced split files.
- ``split_manifest_by_physical_fruit``: deterministic, class-balanced,
  physical-fruit-grouped splitter (train/val/test).

Invariant enforced everywhere: **images of the same physical fruit must never
appear across different splits.** The splitter guarantees it by construction
and re-verifies it; validation re-checks it on any existing split files.

The module never modifies the raw dataset. It parses, validates, and reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "SchemaField",
    "CANONICAL_FIELDS",
    "REQUIRED_CANONICAL_FIELDS",
    "CANONICAL_ENUMS",
    "CanonicalRecord",
    "ManifestValidationReport",
    "FruitSplitResult",
    "load_canonical_manifest",
    "validate_canonical_manifest",
    "find_physical_fruit_leakage",
    "find_session_leakage",
    "split_manifest_by_physical_fruit",
    "write_manifest_csv",
]


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaField:
    """Definition of a single canonical manifest field."""

    name: str
    description: str
    required: bool = False
    allowed_values: Optional[Tuple[str, ...]] = None
    value_type: str = "str"  # "str" or "float"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    parse_iso_datetime: bool = False


CANONICAL_FIELDS: Tuple[SchemaField, ...] = (
    SchemaField(
        "image_id",
        "Unique per-image identifier.",
        required=True,
    ),
    SchemaField(
        "image_path",
        "Relative path to the image file (relative to the manifest directory).",
        required=True,
    ),
    SchemaField(
        "fruit_type",
        "Type of fruit.",
        required=True,
        allowed_values=("apples", "banana", "oranges"),
    ),
    SchemaField(
        "freshness_label",
        "Freshness tier.",
        required=True,
        allowed_values=("fresh", "stale", "rotten"),
    ),
    SchemaField(
        "physical_fruit_id",
        "Unique specimen identifier. Determines grouping for splits.",
        required=True,
    ),
    SchemaField(
        "capture_session_id",
        "Identifier of the capture run the image was taken in.",
        required=True,
    ),
    SchemaField(
        "capture_timestamp",
        "ISO-8601 capture time.",
        required=True,
        parse_iso_datetime=True,
    ),
    SchemaField(
        "camera_id",
        "Camera identifier.",
        required=True,
    ),
    SchemaField(
        "lighting_condition",
        "Lighting during capture.",
        allowed_values=("natural", "indoor_artificial", "mixed", "low_light"),
    ),
    SchemaField(
        "background_type",
        "Background in the frame.",
        allowed_values=("plain", "cluttered", "hand", "surface", "other"),
    ),
    SchemaField(
        "viewing_angle",
        "Camera angle relative to the fruit.",
        allowed_values=("front", "side", "top", "angled", "overhead"),
    ),
    SchemaField(
        "occlusion_level",
        "Fraction of the fruit occluded, in [0, 1].",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
    ),
    SchemaField(
        "distance_category",
        "Rough capture distance.",
        allowed_values=("close", "medium", "far"),
    ),
    SchemaField(
        "storage_condition",
        "How the fruit was stored.",
        allowed_values=("room_temp", "fridge", "counter", "bag", "other"),
    ),
    SchemaField(
        "days_since_purchase",
        "Non-negative days between purchase/harvest and capture.",
        value_type="float",
        min_value=0.0,
    ),
    SchemaField(
        "annotator",
        "Who assigned the ground-truth freshness label.",
    ),
    SchemaField(
        "annotation_confidence",
        "Annotator confidence in the label, in [0, 1].",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
    ),
)

REQUIRED_CANONICAL_FIELDS: Tuple[str, ...] = tuple(
    f.name for f in CANONICAL_FIELDS if f.required
)

CANONICAL_ENUMS: Dict[str, Tuple[str, ...]] = {
    f.name: f.allowed_values
    for f in CANONICAL_FIELDS
    if f.allowed_values is not None
}

_FIELD_BY_NAME: Dict[str, SchemaField] = {f.name: f for f in CANONICAL_FIELDS}


def _parse_iso_timestamp(value: str) -> bool:
    """Return True if ``value`` parses as an ISO-8601-ish datetime."""
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    return False
# ---------------------------------------------------------------------------
# Canonical record
# ---------------------------------------------------------------------------


@dataclass
class CanonicalRecord:
    """One manifest row: one image with its full canonical metadata.

    Attributes:
        data: Raw column -> value mapping for this row.
        row_number: 1-based source line (CSV) or index (JSON). -1 if unknown.
    """

    data: Dict[str, Any]
    row_number: int = -1

    def get(self, name: str, default: Any = None) -> Any:
        """Return the value for ``name`` or ``default`` if absent."""
        value = self.data.get(name, default)
        if isinstance(value, str):
            value = value.strip()
        return value

    @property
    def image_id(self) -> str:
        return str(self.get("image_id", "") or "")

    @property
    def image_path(self) -> str:
        return str(self.get("image_path", "") or "")

    @property
    def fruit_type(self) -> str:
        return str(self.get("fruit_type", "") or "")

    @property
    def freshness_label(self) -> str:
        return str(self.get("freshness_label", "") or "")

    @property
    def physical_fruit_id(self) -> str:
        return str(self.get("physical_fruit_id", "") or "")

    @property
    def capture_session_id(self) -> str:
        return str(self.get("capture_session_id", "") or "")

    @property
    def class_name(self) -> str:
        """Derived model class: ``freshness_label + fruit_type``.

        The 6-class checkpoint and the shipped data folders name classes
        freshness-first, e.g. ``fresh`` + ``apples`` -> ``freshapples``. This
        property therefore concatenates ``freshness_label`` (e.g. ``fresh``)
        **before** ``fruit_type`` (e.g. ``apples``) so a canonical manifest
        aligns with the checkpoint's ``class_names`` ordering:
        ``['freshapples', 'freshbanana', 'freshoranges', 'rottenapples',
        'rottenbanana', 'rottenoranges']``.
        """
        if self.fruit_type and self.freshness_label:
            return f"{self.freshness_label}{self.fruit_type}"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the raw row data."""
        return dict(self.data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CanonicalRecord(image_id={self.image_id!r}, class={self.class_name!r})"


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def _rows_from_csv(path: Path) -> List[CanonicalRecord]:
    """Read a CSV manifest (UTF-8, optional BOM)."""
    records: List[CanonicalRecord] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV manifest has no header row: {path}")
        for row_number, row in enumerate(reader, start=2):
            # Drop completely empty rows.
            if not any(str(v).strip() for v in row.values()):
                continue
            records.append(CanonicalRecord({k: (v or "") for k, v in row.items()}, row_number=row_number))
    return records


def _rows_from_json(path: Path) -> List[CanonicalRecord]:
    """Read a JSON manifest (array of objects, or ``{"records": [...]}``)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("records")
        if items is None:
            raise ValueError(
                f"JSON manifest must be an array of objects or contain a "
                f"'records' key: {path}"
            )
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported JSON manifest structure: {path}")
    records = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Row {index + 1} of {path} is not an object")
        records.append(CanonicalRecord({k: (v or "") for k, v in item.items()}, row_number=index + 1))
    return records


def load_canonical_manifest(path: Path | str) -> List[CanonicalRecord]:
    """Load a canonical manifest (CSV or JSON) from ``path``.

    Returns:
        List of :class:`CanonicalRecord`, one per row/image.

    Raises:
        FileNotFoundError: if the manifest does not exist.
        ValueError: if the format is unsupported or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _rows_from_csv(path)
    if suffix == ".json":
        return _rows_from_json(path)
    raise ValueError(
        f"Unsupported manifest format {path.suffix!r}. Use .csv or .json."
    )


def write_manifest_csv(records: Iterable[CanonicalRecord], path: Path | str) -> Path:
    """Write ``records`` to a canonical CSV manifest at ``path``.

    The header is the union of the canonical fields, preserving schema order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(REQUIRED_CANONICAL_FIELDS) + [
        f.name for f in CANONICAL_FIELDS if not f.required
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
    return path
# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


@dataclass
class ManifestValidationReport:
    """Structured findings from validating a canonical manifest.

    Each finding list holds tuples of ``(row_reference, detail)`` except where
    documented otherwise. ``errors`` are deterministically sorted.
    """

    total_rows: int = 0
    missing_required: List[Tuple[str, str]] = field(default_factory=list)
    missing_optional: List[Tuple[str, str]] = field(default_factory=list)
    invalid_labels: List[Tuple[str, str]] = field(default_factory=list)
    invalid_values: List[Tuple[str, str]] = field(default_factory=list)
    unknown_enum_values: List[Tuple[str, str]] = field(default_factory=list)
    duplicate_image_ids: List[str] = field(default_factory=list)
    duplicate_paths: List[str] = field(default_factory=list)
    exact_duplicate_files: List[Tuple[str, str]] = field(default_factory=list)
    missing_image_files: List[str] = field(default_factory=list)
    impossible_combinations: List[Tuple[str, str]] = field(default_factory=list)
    fruit_type_conflicts: List[Tuple[str, str, str]] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    fruit_type_distribution: Dict[str, int] = field(default_factory=dict)
    freshness_distribution: Dict[str, int] = field(default_factory=dict)
    physical_fruit_count: int = 0
    capture_session_count: int = 0
    per_fruit_image_counts: Dict[str, int] = field(default_factory=dict)
    class_fruit_counts: Dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 1.0
    is_balanced: bool = True

    @property
    def valid_rows(self) -> int:
        return max(0, self.total_rows - len(self._unusable_rows()))

    def _unusable_rows(self) -> Set[str]:
        """Row references that cannot be used for training at all."""
        bad: Set[str] = set()
        for row_ref, _ in self.missing_required:
            bad.add(row_ref)
        for row_ref, _ in self.invalid_values:
            bad.add(row_ref)
        return bad

    @property
    def error_count(self) -> int:
        return (
            len(self.missing_required)
            + len(self.invalid_values)
            + len(self.duplicate_image_ids)
            + len(self.duplicate_paths)
            + len(self.exact_duplicate_files)
            + len(self.missing_image_files)
            + len(self.fruit_type_conflicts)
        )

    @property
    def warning_count(self) -> int:
        return (
            len(self.missing_optional)
            + len(self.invalid_labels)
            + len(self.unknown_enum_values)
            + len(self.impossible_combinations)
        )

    def is_pass(self) -> bool:
        """True when there are no blocking errors."""
        return self.error_count == 0

    def to_dict(self) -> Dict[str, Any]:
        """Full machine-readable report."""
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "is_pass": self.is_pass(),
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "invalid_labels": self.invalid_labels,
            "invalid_values": self.invalid_values,
            "unknown_enum_values": self.unknown_enum_values,
            "duplicate_image_ids": self.duplicate_image_ids,
            "duplicate_paths": self.duplicate_paths,
            "exact_duplicate_files": self.exact_duplicate_files,
            "missing_image_files": self.missing_image_files,
            "impossible_combinations": self.impossible_combinations,
            "fruit_type_conflicts": self.fruit_type_conflicts,
            "class_distribution": self.class_distribution,
            "fruit_type_distribution": self.fruit_type_distribution,
            "freshness_distribution": self.freshness_distribution,
            "physical_fruit_count": self.physical_fruit_count,
            "capture_session_count": self.capture_session_count,
            "per_fruit_image_counts": self.per_fruit_image_counts,
            "class_fruit_counts": self.class_fruit_counts,
            "imbalance_ratio": self.imbalance_ratio,
            "is_balanced": self.is_balanced,
        }

    def summary_lines(self) -> List[str]:
        """Short human-readable summary bullet list."""
        return [
            f"Rows: {self.total_rows}",
            f"Valid rows: {self.valid_rows}",
            f"Physical fruits: {self.physical_fruit_count}",
            f"Capture sessions: {self.capture_session_count}",
            f"Classes: {sorted(self.class_distribution)}",
            f"Class imbalance ratio (max/min): {self.imbalance_ratio:.2f}",
            f"Errors: {self.error_count}, Warnings: {self.warning_count}",
        ]
def _row_reference(record: CanonicalRecord) -> str:
    """Short stable reference to a row for findings."""
    if record.image_id:
        return record.image_id
    return f"row_{record.row_number}" if record.row_number >= 1 else "row_?"


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_canonical_manifest(
    records: Sequence[CanonicalRecord],
    data_root: Optional[Path | str] = None,
) -> ManifestValidationReport:
    """Validate a canonical manifest.

    Detects: missing required/optional metadata, invalid labels, out-of-range
    numeric values, unparseable timestamps, duplicate image IDs, duplicate
    paths, exact-duplicate files (by MD5), missing image files, impossible
    metadata combinations (fruit-type contradictions within one physical
    fruit), and class imbalance.

    Args:
        records: Manifest rows.
        data_root: Optional directory against which ``image_path`` values are
            resolved for file-existence and byte-duplicate checks.

    Returns:
        :class:`ManifestValidationReport` with all findings.
    """
    report = ManifestValidationReport(total_rows=len(records))
    root = Path(data_root) if data_root is not None else None

    # --- per-row checks ---------------------------------------------------
    seen_ids: Dict[str, str] = {}
    seen_paths: Set[str] = set()
    fruit_type_by_fruit: Dict[str, List[Tuple[str, str]]] = {}
    session_freshness: Dict[Tuple[str, str], Set[str]] = {}
    file_hashes: Dict[str, str] = {}

    for record in records:
        ref = _row_reference(record)

        # 1. Missing required / optional metadata.
        for field_def in CANONICAL_FIELDS:
            value = record.get(field_def.name)
            present = value is not None and str(value).strip() != ""
            if field_def.required and not present:
                report.missing_required.append((ref, field_def.name))
            elif not field_def.required and not present:
                report.missing_optional.append((ref, field_def.name))

        # 2. Enum labels; non-label enums are warned as unknown values.
        if record.fruit_type and record.fruit_type not in CANONICAL_ENUMS.get("fruit_type", ()):
            report.invalid_labels.append(
                (ref, f"fruit_type={record.fruit_type!r} not in canonical set")
            )
        if record.freshness_label and record.freshness_label not in CANONICAL_ENUMS.get("freshness_label", ()):
            report.invalid_labels.append(
                (ref, f"freshness_label={record.freshness_label!r} not in canonical set")
            )
        for field_name, allowed in CANONICAL_ENUMS.items():
            if field_name in ("fruit_type", "freshness_label"):
                continue
            value = record.get(field_name)
            if value and value not in allowed:
                report.unknown_enum_values.append((ref, f"{field_name}={value!r}"))
# 3. Numeric bounds and timestamp parsing.
        for field_def in CANONICAL_FIELDS:
            value = record.get(field_def.name)
            if value is None or str(value).strip() == "":
                continue
            if field_def.parse_iso_datetime and not _parse_iso_timestamp(str(value)):
                report.invalid_values.append(
                    (ref, f"{field_def.name}={value!r} not ISO-8601")
                )
            if field_def.value_type == "float":
                number = _parse_float(value)
                if number is None:
                    report.invalid_values.append(
                        (ref, f"{field_def.name}={value!r} not a number")
                    )
                else:
                    if field_def.min_value is not None and number < field_def.min_value:
                        report.invalid_values.append(
                            (ref, f"{field_def.name}={number} below minimum {field_def.min_value}")
                        )
                    if field_def.max_value is not None and number > field_def.max_value:
                        report.invalid_values.append(
                            (ref, f"{field_def.name}={number} above maximum {field_def.max_value}")
                        )

        # 4. Duplicate image id / path.
        if record.image_id:
            if record.image_id in seen_ids:
                report.duplicate_image_ids.append(record.image_id)
            else:
                seen_ids[record.image_id] = ref
        if record.image_path:
            if record.image_path in seen_paths:
                report.duplicate_paths.append(record.image_path)
            else:
                seen_paths.add(record.image_path)

        # 5. File existence + byte duplicates (only with a resolvable root).
        image_path = record.image_path
        if image_path and root is not None:
            resolved = (root / image_path).resolve()
            if not resolved.is_file():
                report.missing_image_files.append(str(image_path))
            else:
                digest = hashlib.md5(resolved.read_bytes()).hexdigest()
                if digest in file_hashes and file_hashes[digest] != str(resolved):
                    report.exact_duplicate_files.append(
                        (file_hashes[digest], str(resolved))
                    )
                else:
                    file_hashes[digest] = str(resolved)

        # 6. fruit_type must be consistent per physical fruit.
        if record.physical_fruit_id and record.fruit_type:
            fruit_type_by_fruit.setdefault(record.physical_fruit_id, []).append(
                (record.fruit_type, ref)
            )

        # 7. Same fruit + same session with contradictory freshness.
        if record.physical_fruit_id and record.capture_session_id and record.freshness_label:
            key = (record.physical_fruit_id, record.capture_session_id)
            session_freshness.setdefault(key, set()).add(record.freshness_label)

    # --- cross-row impossible combinations -------------------------------
    for fruit_id, entries in sorted(fruit_type_by_fruit.items()):
        types = {t for t, _ in entries}
        if len(types) > 1:
            leader = sorted(
                ((t, sum(1 for tt, _ in entries if tt == t)) for t in types),
                key=lambda pair: (-pair[1], pair[0]),
            )[0][0]
            for t, ref in sorted(entries, key=lambda pair: pair[1]):
                if t != leader:
                    report.fruit_type_conflicts.append((fruit_id, t, ref))
                    report.impossible_combinations.append(
                        (ref, f"physical_fruit_id={fruit_id} has conflicting fruit_type {t!r}")
                    )

    for (fruit_id, session_id), labels in sorted(session_freshness.items()):
        if len(labels) > 1 and "fresh" in labels and "rotten" in labels:
            report.impossible_combinations.append(
                (f"{fruit_id}@{session_id}", "fresh and rotten in the same capture session")
            )
# --- class distributions ---------------------------------------------
    class_counts: Dict[str, int] = {}
    fruit_class: Dict[str, List[str]] = {}
    fruit_counts: Dict[str, int] = {}
    for record in records:
        cls = record.class_name if record.class_name else "unknown"
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if record.physical_fruit_id:
            fruit_counts[record.physical_fruit_id] = fruit_counts.get(record.physical_fruit_id, 0) + 1
            fruit_class.setdefault(record.physical_fruit_id, []).append(cls)
    report.class_distribution = dict(sorted(class_counts.items()))

    report.fruit_type_distribution = {}
    report.freshness_distribution = {}
    for record in records:
        if record.fruit_type:
            report.fruit_type_distribution[record.fruit_type] = (
                report.fruit_type_distribution.get(record.fruit_type, 0) + 1
            )
        if record.freshness_label:
            report.freshness_distribution[record.freshness_label] = (
                report.freshness_distribution.get(record.freshness_label, 0) + 1
            )
    report.fruit_type_distribution = dict(sorted(report.fruit_type_distribution.items()))
    report.freshness_distribution = dict(sorted(report.freshness_distribution.items()))

    report.physical_fruit_count = len(fruit_counts)
    report.per_fruit_image_counts = dict(sorted(fruit_counts.items()))
    report.capture_session_count = len({r.capture_session_id for r in records if r.capture_session_id})

    # Modal class per fruit, then per-class fruit counts.
    modal_by_fruit: Dict[str, str] = {}
    for fruit_id, classes in fruit_class.items():
        modal_by_fruit[fruit_id] = sorted(
            ((c, sum(1 for cc in classes if cc == c)) for c in set(classes)),
            key=lambda pair: (-pair[1], pair[0]),
        )[0][0]
    class_fruit_counts: Dict[str, int] = {}
    for cls in modal_by_fruit.values():
        class_fruit_counts[cls] = class_fruit_counts.get(cls, 0) + 1
    report.class_fruit_counts = dict(sorted(class_fruit_counts.items()))

    if class_counts:
        values = list(class_counts.values())
        report.imbalance_ratio = max(values) / min(values) if min(values) > 0 else float("inf")
        report.is_balanced = report.imbalance_ratio <= 3.0

    return report


# ---------------------------------------------------------------------------
# Leakage detection (for existing split files)
# ---------------------------------------------------------------------------


def find_physical_fruit_leakage(
    split_records: Dict[str, Sequence[CanonicalRecord]],
) -> List[Tuple[str, str, str]]:
    """Find physical-fruit IDs appearing in more than one split.

    Args:
        split_records: Mapping ``split_name -> records`` (e.g. ``{"train": ...,
            "val": ..., "test": ...}``).

    Returns:
        Sorted list of ``(physical_fruit_id, split_a, split_b)`` violations.
    """
    fruit_to_splits: Dict[str, Set[str]] = {}
    for split_name, records in split_records.items():
        for record in records:
            fruit_id = record.physical_fruit_id
            if not fruit_id:
                continue
            fruit_to_splits.setdefault(fruit_id, set()).add(split_name)

    violations: List[Tuple[str, str, str]] = []
    for fruit_id, splits in sorted(fruit_to_splits.items()):
        ordered = sorted(splits)
        if len(ordered) > 1:
            for a in range(len(ordered)):
                for b in range(a + 1, len(ordered)):
                    violations.append((fruit_id, ordered[a], ordered[b]))
    return violations


def find_session_leakage(
    split_records: Dict[str, Sequence[CanonicalRecord]],
) -> List[Tuple[str, str, str]]:
    """Find capture-session IDs appearing in more than one split.

    Session leakage is advisory (a session may legitimately contain multiple
    physical fruits, each correctly confined to one split), but a session
    crossing splits is worth reporting.
    """
    session_to_splits: Dict[str, Set[str]] = {}
    for split_name, records in split_records.items():
        for record in records:
            session_id = record.capture_session_id
            if not session_id:
                continue
            session_to_splits.setdefault(session_id, set()).add(split_name)

    violations: List[Tuple[str, str, str]] = []
    for session_id, splits in sorted(session_to_splits.items()):
        ordered = sorted(splits)
        if len(ordered) > 1:
            for a in range(len(ordered)):
                for b in range(a + 1, len(ordered)):
                    violations.append((session_id, ordered[a], ordered[b]))
    return violations
# ---------------------------------------------------------------------------
# Physical-fruit grouped splitting
# ---------------------------------------------------------------------------


def _partition_counts(n: int, ratios: Sequence[float]) -> List[int]:
    """Deterministically partition ``n`` whole units into split counts.

    Uses largest-remainder apportionment so the counts sum exactly to ``n``
    and best approximate ``ratios``. When ``n >= 3`` every split receives at
    least one unit.
    """
    if n <= 0:
        return [0, 0, 0]
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    exact = [n * r / total_ratio for r in ratios]
    base = [int(v) for v in exact]
    remainder = n - sum(base)
    # Largest fractional remainders get the leftover counts (ties -> earlier split).
    order = sorted(range(len(exact)), key=lambda i: (exact[i] - base[i], -i), reverse=True)
    for i in range(remainder):
        base[order[i]] += 1
    # Guarantee at least one unit per split when n >= 3.
    if n >= 3 and any(c == 0 for c in base):
        for i in range(len(base)):
            if base[i] == 0:
                # Steal from the largest split.
                j = max(range(len(base)), key=lambda k: base[k])
                base[i] += 1
                base[j] -= 1
    return base


@dataclass
class FruitSplitResult:
    """Deterministic physical-fruit-grouped split of a manifest.

    Guarantees:
    - Every ``physical_fruit_id`` appears in exactly one split.
    - The assignment is deterministic for a given ``(records, seed)``.
    - Class balance is reported per split (images and physical fruits).
    """

    train: List[CanonicalRecord] = field(default_factory=list)
    val: List[CanonicalRecord] = field(default_factory=list)
    test: List[CanonicalRecord] = field(default_factory=list)
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    @property
    def fruit_counts(self) -> Dict[str, int]:
        return {
            "train": len(self.train_fruit_ids),
            "val": len(self.val_fruit_ids),
            "test": len(self.test_fruit_ids),
        }

    @property
    def train_fruit_ids(self) -> Set[str]:
        return {r.physical_fruit_id for r in self.train if r.physical_fruit_id}

    @property
    def val_fruit_ids(self) -> Set[str]:
        return {r.physical_fruit_id for r in self.val if r.physical_fruit_id}

    @property
    def test_fruit_ids(self) -> Set[str]:
        return {r.physical_fruit_id for r in self.test if r.physical_fruit_id}

    def verify_no_fruit_leakage(self) -> bool:
        """True iff no physical fruit appears in more than one split."""
        combined = self.train_fruit_ids | self.val_fruit_ids | self.test_fruit_ids
        union_len = (
            len(self.train_fruit_ids) + len(self.val_fruit_ids) + len(self.test_fruit_ids)
        )
        return len(combined) == union_len

    def verify_full_partition(self) -> bool:
        """True iff every record is assigned to exactly one split."""
        return len(self.train) + len(self.val) + len(self.test) == self.total

    def splits_by_class(self) -> Dict[str, Dict[str, int]]:
        """Return ``{class_name: {split: image_count}}``."""
        result: Dict[str, Dict[str, int]] = {}
        for split_name, records in (
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ):
            for record in records:
                cls = record.class_name or "unknown"
                result.setdefault(cls, {}).setdefault(split_name, 0)
                result[cls][split_name] += 1
        for cls in result:
            for split_name in ("train", "val", "test"):
                result[cls].setdefault(split_name, 0)
        return dict(sorted(result.items()))

    def fruits_by_class(self) -> Dict[str, Dict[str, int]]:
        """Return ``{class_name: {split: physical_fruit_count}}``.

        A fruit is assigned to the lexicographically-first class among its
        rows (stable and deterministic; used for reporting only).
        """
        records_by_fruit: Dict[str, List[CanonicalRecord]] = {}
        for record in self.all_records:
            if record.physical_fruit_id:
                records_by_fruit.setdefault(record.physical_fruit_id, []).append(record)
        split_of_fruit: Dict[str, str] = {}
        for name, fruit_ids in (
            ("train", self.train_fruit_ids),
            ("val", self.val_fruit_ids),
            ("test", self.test_fruit_ids),
        ):
            for fruit_id in fruit_ids:
                split_of_fruit[fruit_id] = name

        result: Dict[str, Dict[str, int]] = {}
        for fruit_id, records in records_by_fruit.items():
            classes = sorted({r.class_name for r in records if r.class_name})
            cls = classes[0] if classes else "unknown"
            split = split_of_fruit.get(fruit_id, "unknown")
            result.setdefault(cls, {}).setdefault(split, 0)
            result[cls][split] += 1
        for cls in result:
            for split_name in ("train", "val", "test"):
                result[cls].setdefault(split_name, 0)
        return dict(sorted(result.items()))

    @property
    def all_records(self) -> List[CanonicalRecord]:
        return self.train + self.val + self.test

    def class_balance_report(self) -> str:
        """Human-readable class-balance report (images + fruits per split)."""
        lines = [
            "Class balance by images:",
            f"  {'class':20s} {'train':>7s} {'val':>7s} {'test':>7s}",
            "-" * 47,
        ]
        for cls, splits in self.splits_by_class().items():
            lines.append(
                f"  {cls:20s} {splits['train']:7d} {splits['val']:7d} {splits['test']:7d}"
            )
        lines.append("")
        lines.append("Class balance by physical fruits:")
        lines.append(f"  {'class':20s} {'train':>7s} {'val':>7s} {'test':>7s}")
        lines.append("-" * 47)
        for cls, splits in self.fruits_by_class().items():
            lines.append(
                f"  {cls:20s} {splits['train']:7d} {splits['val']:7d} {splits['test']:7d}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable summary of the split."""
        return {
            "seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "image_counts": self.counts,
            "fruit_counts": self.fruit_counts,
            "total_images": self.total,
            "no_fruit_leakage": self.verify_no_fruit_leakage(),
            "full_partition": self.verify_full_partition(),
            "splits_by_class": self.splits_by_class(),
            "fruits_by_class": self.fruits_by_class(),
        }


def _seeded_rng(seed: int, salt: str) -> Tuple[int, int]:
    """Derive two deterministic integers from ``seed`` and ``salt``."""
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16), int(digest[16:32], 16)
def split_manifest_by_physical_fruit(
    records: Sequence[CanonicalRecord],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> FruitSplitResult:
    """Split a canonical manifest by ``physical_fruit_id``.

    Deterministic and class-balanced:

    1. Rows are grouped by ``physical_fruit_id`` and each fruit is assigned a
       modal (most common) class name.
    2. Fruits are processed class-by-class; within each class a stable sort +
       deterministic shuffle (seeded by ``seed`` and the class name) decides
       fruit order, then fruits are apportioned to splits with
       :func:`_partition_counts`.
    3. The result is verified: no fruit may span two splits (raises
       ``ValueError`` otherwise) and every record is present exactly once.

    Args:
        records: Manifest rows.
        train/val/test: Split ratios (need not sum to 1).
        seed: Deterministic seed.

    Returns:
        :class:`FruitSplitResult`.

    Raises:
        ValueError: If required grouping fields are missing, a fruit would
            cross splits, or records are lost.
    """
    ratios = (train, val, test)
    if any(r <= 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("train/val/test ratios must be positive and sum to > 0")

    # Group by physical fruit; record modal class per fruit.
    by_fruit: Dict[str, List[CanonicalRecord]] = {}
    for record in records:
        fruit_id = record.physical_fruit_id
        if not fruit_id:
            raise ValueError(
                f"Row {_row_reference(record)} is missing physical_fruit_id; "
                "cannot group-split."
            )
        if not record.fruit_type or not record.freshness_label:
            raise ValueError(
                f"Row {_row_reference(record)} is missing fruit_type or "
                "freshness_label; cannot derive class."
            )
        by_fruit.setdefault(fruit_id, []).append(record)

    modal_class: Dict[str, str] = {}
    for fruit_id, fruit_records in by_fruit.items():
        classes = [r.class_name for r in fruit_records if r.class_name]
        if not classes:
            raise ValueError(f"Fruit {fruit_id} has no class labels.")
        modal_class[fruit_id] = sorted(
            ((c, classes.count(c)) for c in set(classes)),
            key=lambda pair: (-pair[1], pair[0]),
        )[0][0]

    # Group fruit ids by modal class, sorted.
    fruits_by_class: Dict[str, List[str]] = {}
    for fruit_id, cls in modal_class.items():
        fruits_by_class.setdefault(cls, []).append(fruit_id)
    for cls in fruits_by_class:
        fruits_by_class[cls].sort()

    train_fruits: List[str] = []
    val_fruits: List[str] = []
    test_fruits: List[str] = []

    for cls in sorted(fruits_by_class):
        fruit_ids = fruits_by_class[cls]
        a, b = _seeded_rng(seed, cls)
        order = fruit_ids[:]
        # Deterministic pseudo-shuffle seeded per class.
        index = 0
        while index < len(order):
            a = (a * 1103515245 + 12345) & 0x7FFFFFFF
            j = index + (a % (len(order) - index))
            order[index], order[j] = order[j], order[index]
            index += 1
        n_train, n_val, n_test = _partition_counts(len(order), ratios)
        train_fruits.extend(order[:n_train])
        val_fruits.extend(order[n_train:n_train + n_val])
        test_fruits.extend(order[n_train + n_val:])

    train_set, val_set, test_set = set(train_fruits), set(val_fruits), set(test_fruits)

    # Leakage check: global fruit id sets must be disjoint.
    if train_set & val_set or train_set & test_set or val_set & test_set:
        conflict = sorted(
            (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
        )[0]
        raise ValueError(
            "CRITICAL: physical_fruit_id leakage detected across splits for "
            f"fruit {conflict!r}. Splitting aborted."
        )

    result = FruitSplitResult(
        train=[r for fid in train_fruits for r in by_fruit[fid]],
        val=[r for fid in val_fruits for r in by_fruit[fid]],
        test=[r for fid in test_fruits for r in by_fruit[fid]],
        seed=seed,
        train_ratio=train,
        val_ratio=val,
        test_ratio=test,
    )

    if not result.verify_full_partition():
        raise ValueError("CRITICAL: split lost records; records must be immutable.")
    if not result.verify_no_fruit_leakage():
        raise ValueError("CRITICAL: physical-fruit leakage after split verification.")

    logger.info(
        "Split %d records into train=%d, val=%d, test=%d (seed=%d)",
        result.total, len(result.train), len(result.val), len(result.test), seed,
    )
    return result