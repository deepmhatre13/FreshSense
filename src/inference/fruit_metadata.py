"""Fruit metadata database loader (SmartFreshAI Phase 4).

Loads :file:`fruit_database.json` containing per-fruit storage, nutrition and
shelf-life metadata. This file is the SINGLE SOURCE OF TRUTH for shelf-life
ranges; no botanical day numbers are duplicated anywhere in Python code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["FruitMetadata", "FruitMetadataDatabase"]


def _validate_shelf_life_days(value) -> Optional[List[int]]:
    """Normalize ``typical_shelf_life_days`` into ``[min, max]`` ints.

    Returns ``None`` when the value is missing or invalid. This module never
    fabricates a fallback range -- callers must surface an explicit
    unsupported state instead (see ``ShelfLifeEstimator``).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, int, float)):
        return None  # a bare number/string is not a (min, max) range
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo, hi = value
    for bound in (lo, hi):
        # Strict integers only: fractional day counts are never silently
        # coerced (e.g. 3.5 -> 3), they make the entry explicitly invalid.
        if isinstance(bound, bool) or not isinstance(bound, int):
            return None
    lo_i, hi_i = int(lo), int(hi)
    if lo_i < 0 or hi_i <= 0 or lo_i > hi_i:
        return None
    return [lo_i, hi_i]


@dataclass
class FruitMetadata:
    """Metadata for a single fruit.

    Attributes:
        name: Lower-case fruit name.
        scientific_name: Botanical name.
        optimal_storage: Human-readable storage guidance.
        ideal_temperature_c: Ideal temperature range as a string.
        ideal_humidity_pct: Ideal humidity range as a string.
        typical_shelf_life_days: (min, max) typical shelf life in days, or
            None when missing/invalid in the source JSON. No default range is
            ever fabricated.
        spoilage_signs: List of spoilage indicators.
        nutrition: Dict of nutrition facts.
        storage_tip: Extra storage guidance.
    """

    name: str
    scientific_name: str = ""
    optimal_storage: str = ""
    ideal_temperature_c: str = ""
    ideal_humidity_pct: str = ""
    typical_shelf_life_days: Optional[List[int]] = None
    spoilage_signs: List[str] = field(default_factory=list)
    nutrition: Dict[str, float] = field(default_factory=dict)
    storage_tip: str = ""


class FruitMetadataDatabase:
    """Loads and serves fruit metadata."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path) if db_path else self._default_path()
        self._metadata: Dict[str, FruitMetadata] = {}
        self._validation_issues: List[str] = []
        self.load()

    @staticmethod
    def _default_path() -> Path:
        # fruit_database.json lives at the repository root.
        return Path(__file__).resolve().parents[2] / "fruit_database.json"

    def load(self) -> None:
        """Load and validate metadata from the JSON database.

        Missing or malformed entries are never silently repaired: a fruit
        whose ``typical_shelf_life_days`` range is absent or invalid is kept
        with ``typical_shelf_life_days=None`` and recorded in
        ``validation_issues`` so downstream consumers report an explicit
        ``unsupported`` shelf-life status instead of a fabricated default.
        """
        self._metadata.clear()
        self._validation_issues.clear()
        if not self.db_path.exists():
            logger.warning("Fruit database not found: %s", self.db_path)
            return
        try:
            with open(self.db_path, "r", encoding="utf-8-sig") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load fruit database %s: %s", self.db_path, exc)
            return

        if not isinstance(raw, dict):
            issue = f"{self.db_path}: top-level JSON must be an object mapping fruit names to metadata"
            self._validation_issues.append(issue)
            logger.error("%s", issue)
            return

        for name, data in raw.items():
            key = name.strip().lower()
            if not key or not isinstance(data, dict):
                issue = f"{name!r}: invalid entry (must map a non-empty name to an object)"
                self._validation_issues.append(issue)
                logger.warning("Fruit metadata issue: %s", issue)
                continue

            shelf_range = _validate_shelf_life_days(data.get("typical_shelf_life_days"))
            if shelf_range is None:
                issue = f"{key}: missing or invalid 'typical_shelf_life_days' (expected [min, max], 0 <= min <= max)"
                self._validation_issues.append(issue)
                logger.warning("Fruit metadata issue: %s", issue)

            meta = FruitMetadata(
                name=key,
                scientific_name=str(data.get("scientific_name", "")),
                optimal_storage=str(data.get("optimal_storage", "")),
                ideal_temperature_c=str(data.get("ideal_temperature_c", "")),
                ideal_humidity_pct=str(data.get("ideal_humidity_pct", "")),
                typical_shelf_life_days=shelf_range,
                spoilage_signs=list(data.get("spoilage_signs", [])),
                nutrition=dict(data.get("nutrition", {})),
                storage_tip=str(data.get("storage_tip", "")),
            )
            self._metadata[meta.name] = meta
        logger.info(
            "Fruit metadata loaded for %d fruits (%d validation issues)",
            len(self._metadata),
            len(self._validation_issues),
        )

    def get(self, name: str) -> Optional[FruitMetadata]:
        """Return metadata for a fruit by lower-case name."""
        return self._metadata.get(name.strip().lower())

    def get_all(self) -> Dict[str, FruitMetadata]:
        """Return all loaded metadata keyed by lower-case name."""
        return dict(self._metadata)

    def names(self) -> List[str]:
        """Return sorted list of fruit names."""
        return sorted(self._metadata)

    @property
    def metadata_available(self) -> bool:
        """Whether at least one valid fruit entry was loaded."""
        return bool(self._metadata)

    @property
    def validation_issues(self) -> List[str]:
        """Human-readable problems found during the last :meth:`load`."""
        return list(self._validation_issues)