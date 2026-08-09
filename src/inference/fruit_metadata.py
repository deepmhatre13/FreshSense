"""Fruit metadata database loader (FreshSense Phase 4).

Loads :file:`fruit_database.json` containing per-fruit storage, nutrition and
shelf-life metadata. This is the future source of truth for RAG queries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["FruitMetadata", "FruitMetadataDatabase"]


@dataclass
class FruitMetadata:
    """Metadata for a single fruit.

    Attributes:
        name: Lower-case fruit name.
        scientific_name: Botanical name.
        optimal_storage: Human-readable storage guidance.
        ideal_temperature_c: Ideal temperature range as a string.
        ideal_humidity_pct: Ideal humidity range as a string.
        typical_shelf_life_days: (min, max) typical shelf life in days.
        spoilage_signs: List of spoilage indicators.
        nutrition: Dict of nutrition facts.
        storage_tip: Extra storage guidance.
    """

    name: str
    scientific_name: str = ""
    optimal_storage: str = ""
    ideal_temperature_c: str = ""
    ideal_humidity_pct: str = ""
    typical_shelf_life_days: List[int] = field(default_factory=lambda: [3, 7])
    spoilage_signs: List[str] = field(default_factory=list)
    nutrition: Dict[str, float] = field(default_factory=dict)
    storage_tip: str = ""


class FruitMetadataDatabase:
    """Loads and serves fruit metadata."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path) if db_path else self._default_path()
        self._metadata: Dict[str, FruitMetadata] = {}
        self.load()

    @staticmethod
    def _default_path() -> Path:
        # fruit_database.json lives at the repository root.
        return Path(__file__).resolve().parents[2] / "fruit_database.json"

    def load(self) -> None:
        """Load metadata from the JSON database."""
        if not self.db_path.exists():
            logger.warning("Fruit database not found: %s", self.db_path)
            return
        try:
            with open(self.db_path, "r", encoding="utf-8-sig") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load fruit database %s: %s", self.db_path, exc)
            return

        for name, data in raw.items():
            meta = FruitMetadata(
                name=name.strip().lower(),
                scientific_name=data.get("scientific_name", ""),
                optimal_storage=data.get("optimal_storage", ""),
                ideal_temperature_c=data.get("ideal_temperature_c", ""),
                ideal_humidity_pct=data.get("ideal_humidity_pct", ""),
                typical_shelf_life_days=list(data.get("typical_shelf_life_days", [3, 7])),
                spoilage_signs=list(data.get("spoilage_signs", [])),
                nutrition=dict(data.get("nutrition", {})),
                storage_tip=data.get("storage_tip", ""),
            )
            self._metadata[meta.name] = meta
        logger.info("Fruit metadata loaded for %d fruits", len(self._metadata))

    def get(self, name: str) -> Optional[FruitMetadata]:
        """Return metadata for a fruit by lower-case name."""
        return self._metadata.get(name.strip().lower())

    def get_all(self) -> Dict[str, FruitMetadata]:
        """Return all loaded metadata keyed by lower-case name."""
        return dict(self._metadata)

    def names(self) -> List[str]:
        """Return sorted list of fruit names."""
        return sorted(self._metadata)

