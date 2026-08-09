"""Adaptive inference cadence for FreshSense.

Controls how often classification is run based on:
- Detector stability (tracking)
- Confidence stability
- Motion detection
- Resource constraints
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CadenceConfig:
    """Configuration for adaptive inference cadence.

    Attributes:
        min_interval_ms: Minimum time between classifications.
        max_interval_ms: Maximum time between classifications.
        stability_frames: Number of stable detections required.
        confidence_threshold: Minimum confidence to consider stable.
    """

    min_interval_ms: float = 100.0
    max_interval_ms: float = 1000.0
    stability_frames: int = 3
    confidence_threshold: float = 0.7


class AdaptiveCadence:
    """Adaptive classification cadence controller.

    Reduces unnecessary classifications when:
    - Object is stably tracked
    - Confidence is stable
    - Little time has passed
    """

    def __init__(self, config: CadenceConfig | None = None):
        self.config = config or CadenceConfig()
        self._last_classification_time: float = 0.0
        self._stable_count: int = 0
        self._last_class: Optional[str] = None
        self._last_confidence: float = 0.0

    def should_classify(
        self,
        current_time: float,
        detector_confidence: float,
        predicted_class: Optional[str] = None,
    ) -> bool:
        """Determine if classification should run.

        Args:
            current_time: Current timestamp in seconds.
            detector_confidence: Current detector confidence.
            predicted_class: Current predicted class name.

        Returns:
            True if classification should run, False otherwise.
        """
        elapsed_ms = (current_time - self._last_classification_time) * 1000.0
        if elapsed_ms < self.config.min_interval_ms:
            return False
        if elapsed_ms >= self.config.max_interval_ms:
            self._stable_count = 0
            return True
        if detector_confidence < self.config.confidence_threshold:
            self._stable_count = 0
            return True
        if predicted_class is not None and predicted_class != self._last_class:
            self._stable_count = 0
            self._last_class = predicted_class
            return True
        self._stable_count += 1
        if self._stable_count >= self.config.stability_frames:
            if elapsed_ms >= self.config.min_interval_ms * 2:
                self._stable_count = 0
                return True
            return False
        return True

    def reset(self) -> None:
        """Reset cadence state."""
        self._last_classification_time = 0.0
        self._stable_count = 0
        self._last_class = None
        self._last_confidence = 0.0
