"""Confidence fusion between detector and classifier (FreshSense Phase 4)."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ConfidenceFusion", "FusionConfig", "FusionResult", "ImprovedConfidenceFusion", "ConfidenceFusionConfig"]


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for confidence fusion."""

    detection_weight: float = 0.4
    classification_weight: float = 0.6
    floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.detection_weight < 0 or self.classification_weight < 0:
            raise ValueError("Weights must be non-negative.")
        if self.detection_weight + self.classification_weight <= 0:
            raise ValueError("Weights must sum to more than zero.")
        if self.floor < 0:
            raise ValueError("floor must be non-negative.")


@dataclass(frozen=True)
class FusionResult:
    """Result of fusing detector and classifier confidences."""

    detection_confidence: float
    classification_confidence: float
    fused_confidence: float


class ConfidenceFusion:
    """Weighted fusion of detector and classifier confidences."""

    def __init__(self, config: FusionConfig) -> None:
        self.config = config

    def fuse(
        self,
        detection_confidence: float,
        classification_confidence: float,
    ) -> FusionResult:
        """Combine the two confidences."""
        total_weight = self.config.detection_weight + self.config.classification_weight
        fused = (
            self.config.detection_weight * detection_confidence
            + self.config.classification_weight * classification_confidence
        ) / total_weight
        fused = max(self.config.floor, min(1.0, fused))
        return FusionResult(
            detection_confidence=detection_confidence,
            classification_confidence=classification_confidence,
            fused_confidence=fused,
        )



@dataclass
class ConfidenceFusionConfig:
    """Extended configuration for stability-aware fusion."""

    detector_weight: float = 0.3
    classifier_weight: float = 0.7
    stability_window: int = 5
    confidence_threshold: float = 0.5
    transition_smoothing: float = 0.3


@dataclass
class FrameFusionResult:
    """Extended fusion result with stability metadata.

    Note: Does not inherit FusionResult because Python 3.12 forbids
    non-frozen dataclasses from inheriting frozen dataclasses.
    """

    detection_confidence: float = 0.0
    classification_confidence: float = 0.0
    fused_confidence: float = 0.0
    stability_bonus: float = 0.0
    raw_confidence: float = 0.0
    is_stable: bool = True


class ImprovedConfidenceFusion:
    """Fuse detector and classifier confidences with temporal stability."""

    def __init__(self, config: ConfidenceFusionConfig | None = None):
        self.config = config or ConfidenceFusionConfig()
        self.history: deque[dict[str, Any]] = deque(maxlen=self.config.stability_window)
        self._last_confidence: float | None = None
        self._last_class: str | None = None
        self._last_timestamp: float | None = None

    def fuse(
        self,
        detector_conf: float | None,
        classifier_conf: float | None,
        class_name: str | None = None,
        timestamp: float | None = None,
    ) -> tuple[str | None, float, dict[str, Any]]:
        """Fuse confidences with stability-aware smoothing."""
        detector_conf = detector_conf if detector_conf is not None else 0.0
        classifier_conf = classifier_conf if classifier_conf is not None else 0.0
        raw_confidence = (
            self.config.detector_weight * detector_conf
            + self.config.classifier_weight * classifier_conf
        )
        stability_bonus = self._compute_stability_bonus(class_name, timestamp)
        confidence = max(0.0, min(1.0, raw_confidence + stability_bonus))
        selected_class = self._apply_smoothing(class_name, confidence)
        info = {
            "detector_confidence": float(detector_conf),
            "classifier_confidence": float(classifier_conf),
            "raw_confidence": float(raw_confidence),
            "stability_bonus": float(stability_bonus),
            "final_confidence": float(confidence),
        }
        if timestamp is not None:
            self.history.append(
                {
                    "class": selected_class,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }
            )
            self._last_timestamp = timestamp
        if selected_class is not None:
            self._last_class = selected_class
            self._last_confidence = confidence
        return selected_class, confidence, info

    def _compute_stability_bonus(self, class_name: str | None, timestamp: float | None) -> float:
        if not self.history or class_name is None or timestamp is None or self._last_timestamp is None:
            return 0.0
        if self._last_class != class_name:
            return -self.config.transition_smoothing
        recency = max(0.0, 1.0 - (timestamp - self._last_timestamp))
        return 0.05 * recency

    def _apply_smoothing(self, class_name: str | None, confidence: float) -> str | None:
        if class_name is None:
            return None
        if confidence < self.config.confidence_threshold:
            return self._last_class if self._last_class is not None else class_name
        return class_name

    def reset(self) -> None:
        self.history.clear()
        self._last_confidence = None
        self._last_class = None
        self._last_timestamp = None
