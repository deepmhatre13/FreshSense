"""Fruit classification result types for FreshSense Phase 4.

Holds per-fruit stabilized classification output after detection, cropping,
and fusion. Each fruit gets its own independent stabiliser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.detection.base_detector import Detection
from src.inference.shelf_life import ShelfLifeEstimate
from src.inference.stabilizer import StabilizedPrediction

logger = logging.getLogger(__name__)

__all__ = ["FruitResult", "MultiFruitResult"]


@dataclass
class FruitResult:
    """Stabilized classification result for a single detected fruit.

    Attributes:
        detection: Source detection (box, label, tracking id).
        stabilized: Stabilized prediction from the EMA/vote/lock stabilizer.
        fused_confidence: Detector + classifier fused confidence.
        freshness_class: "fresh", "stale", or "rotten".
        shelf_life: Estimated remaining shelf life.
        is_uncertain: Whether the prediction was flagged uncertain.
    """

    detection: Detection
    stabilized: StabilizedPrediction
    fused_confidence: float
    freshness_class: str
    shelf_life: Optional[ShelfLifeEstimate] = None
    is_uncertain: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tracking_id": self.detection.tracking_id,
            "fruit": self.detection.label,
            "freshness": self.freshness_class,
            "confidence": self.fused_confidence,
            "detection_confidence": self.detection.confidence,
            "stabilized_confidence": self.stabilized.confidence,
            "ema_confidence": self.stabilized.ema_confidence,
            "is_uncertain": self.is_uncertain,
            "is_locked": self.stabilized.is_locked,
            "lock_count": self.stabilized.lock_count,
            "majority_label": self.stabilized.majority_label,
            "vote_counts": self.stabilized.vote_counts,
            "shelf_life": self.shelf_life.to_range_string() if self.shelf_life else None,
            "bounding_box": {
                "x1": self.detection.bbox.x1,
                "y1": self.detection.bbox.y1,
                "x2": self.detection.bbox.x2,
                "y2": self.detection.bbox.y2,
            },
            "center": list(self.detection.bbox.center),
        }


@dataclass
class MultiFruitResult:
    """Aggregated results for all fruits in a single frame.

    Attributes:
        fruits: List of FruitResult, one per tracked fruit.
        frame_width: Source frame width.
        frame_height: Source frame height.
        unidentified_count: Number of detected fruits not classified.
    """

    fruits: List[FruitResult] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    unidentified_count: int = 0

    def to_dict(self) -> dict:
        return {
            "fruits": [f.to_dict() for f in self.fruits],
            "fruit_count": len(self.fruits),
            "unidentified_count": self.unidentified_count,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
