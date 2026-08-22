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

from typing import Literal


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
        basis_type: Distinctly expresses the basis of the estimate.
        metadata: The fruit metadata used (if any).
    """

    fruit: str
    min_days: int
    max_days: int
    basis: str
    basis_type: Literal["model", "metadata", "metadata_heuristic", "heuristic", "unavailable"]
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
        is_fresh: Optional[bool] = None,
        freshness_class: str = "fresh",
    ) -> ShelfLifeEstimate:
        """Estimate remaining shelf life.

        Args:
            fruit: Lower-case fruit name.
            fused_confidence: Fused detector+classifier confidence (0.0-1.0).
            is_fresh: Legacy flag for whether fruit is fresh (overridden if freshness_class provided).
            freshness_class: Freshness class string ("fresh", "stale", "rotten", "unsupported", "unknown").

        Returns:
            A ShelfLifeEstimate.
        """
        key = fruit.strip().lower()
        meta = self.metadata_db.get(key)

        if not meta:
            return ShelfLifeEstimate(
                fruit=key,
                min_days=0,
                max_days=0,
                basis="Metadata unavailable for shelf-life estimation",
                basis_type="unavailable",
                metadata=None,
            )

        # Map legacy is_fresh argument if explicitly set and freshness_class not modified
        if is_fresh is False and freshness_class == "fresh":
            freshness_class = "rotten"

        meta_range = meta.typical_shelf_life_days
        confidence = max(0.0, min(1.0, fused_confidence))

        if freshness_class == "rotten":
            return ShelfLifeEstimate(
                fruit=key,
                min_days=0,
                max_days=0,
                basis="Fruit is spoiled - consume immediately or discard",
                basis_type="metadata_heuristic",
                metadata=meta,
            )
        elif freshness_class == "stale":
            return ShelfLifeEstimate(
                fruit=key,
                min_days=0,
                max_days=1,
                basis="Fruit is stale - consume within 24 hours",
                basis_type="metadata_heuristic",
                metadata=meta,
            )

        lo, hi = meta_range
        remaining_max = max(0, int(round(hi * confidence)))
        remaining_min = max(0, int(round(lo * confidence)))
        if remaining_min > remaining_max:
            remaining_min = remaining_max

        if freshness_class in ("unsupported", "unknown"):
            basis_str = f"Estimated typical ~{remaining_min}-{remaining_max} days (freshness model unsupported)"
        else:
            basis_str = f"Estimated ~{remaining_min}-{remaining_max} days remaining"

        return ShelfLifeEstimate(
            fruit=key,
            min_days=remaining_min,
            max_days=remaining_max,
            basis=basis_str,
            basis_type="metadata_heuristic",
            metadata=meta,
        )
