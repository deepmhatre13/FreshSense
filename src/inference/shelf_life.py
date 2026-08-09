"""Shelf-life estimation for FreshSense Phase 4.

Estimates remaining shelf life using fruit metadata combined with a freshness
confidence heuristic. The closer to 100% fresh confidence, the closer to the
typical maximum shelf life.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.inference.fruit_metadata import FruitMetadata, FruitMetadataDatabase

logger = logging.getLogger(__name__)

__all__ = ["ShelfLifeEstimator", "ShelfLifeConfig", "ShelfLifeEstimate"]


@dataclass(frozen=True)
class ShelfLifeConfig:
    """Configuration for shelf-life estimation.

    Attributes:
        fresh_bonus: Weight applied to high fresh confidence.
        default_range: Fallback (min, max) days when metadata is missing.
    """

    fresh_bonus: float = 0.5
    default_range: Tuple[int, int] = (3, 10)


@dataclass(frozen=True)
class ShelfLifeEstimate:
    """Result of shelf-life estimation.

    Attributes:
        fruit: Fruit name.
        min_days: Estimated minimum remaining days.
        max_days: Estimated maximum remaining days.
        basis: Human-readable one-line summary.
        metadata: The fruit metadata used (if any).
    """

    fruit: str
    min_days: int
    max_days: int
    basis: str
    metadata: Optional[FruitMetadata] = None

    def to_range_string(self) -> str:
        return f"{self.min_days}-{self.max_days} days"


class ShelfLifeEstimator:
    """Estimates remaining shelf life from fruit metadata + freshness confidence.

    Estimate logic:
        range = metadata range (or fallback)
        remaining_max ~= range_max * fused_confidence
        remaining_min ~= range_min * fused_confidence
    """

    def __init__(
        self,
        config: ShelfLifeConfig,
        metadata_db: Optional[FruitMetadataDatabase] = None,
    ) -> None:
        self.config = config
        self.metadata_db = metadata_db or FruitMetadataDatabase()

    def estimate(
        self,
        fruit: str,
        fused_confidence: float,
        is_fresh: bool = True,
    ) -> ShelfLifeEstimate:
        """Estimate remaining shelf life.

        Args:
            fruit: Lower-case fruit name.
            fused_confidence: Fused detector+classifier confidence (0.0-1.0).
            is_fresh: Whether the fruit is classified as fresh.

        Returns:
            A ShelfLifeEstimate.
        """
        key = fruit.strip().lower()
        meta = self.metadata_db.get(key)
        meta_range = meta.typical_shelf_life_days if meta else self.config.default_range

        # Confidence heuristic: fresh and high-confidence -> longer shelf life.
        confidence = max(0.0, min(1.0, fused_confidence))
        if not is_fresh:
            # Rotten/stale fruit has essentially no remaining shelf life.
            return ShelfLifeEstimate(
                fruit=key,
                min_days=0,
                max_days=0,
                basis="Not suitable for storage - consume immediately",
                metadata=meta,
            )

        lo, hi = meta_range
        remaining_max = max(0, int(round(hi * confidence)))
        remaining_min = max(0, int(round(lo * confidence)))
        if remaining_min > remaining_max:
            remaining_min = remaining_max

        return ShelfLifeEstimate(
            fruit=key,
            min_days=remaining_min,
            max_days=remaining_max,
            basis=f"Estimated ~{remaining_min}-{remaining_max} days remaining",
            metadata=meta,
        )
