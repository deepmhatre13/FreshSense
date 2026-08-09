"""Detector confidence smoothing for FreshSense.

Smooths detector confidence over time to reduce noise from:
- Detection jitter
- Partial occlusions
- Lighting variations
- Frame rate variations
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class SmootherConfig:
    """Configuration for confidence smoothing.

    Attributes:
        window_size: Number of samples for moving average.
        outlier_threshold: Z-score threshold for outlier rejection.
        min_samples: Minimum samples before smoothing activates.
    """

    window_size: int = 5
    outlier_threshold: float = 2.0
    min_samples: int = 3


class DetectorConfidenceSmoother:
    """Smooth detector confidence over time.

    Uses exponential moving average with outlier rejection.
    """

    def __init__(self, config: SmootherConfig | None = None):
        self.config = config or SmootherConfig()
        self._history: deque[float] = deque(maxlen=self.config.window_size)
        self._smoothed: Optional[float] = None
        self._alpha: float = 2.0 / (self.config.window_size + 1.0)

    def update(self, confidence: float) -> float:
        """Update smoothed confidence with new sample.

        Args:
            confidence: New detector confidence value.

        Returns:
            Smoothed confidence value.
        """
        if not self._is_outlier(confidence):
            self._history.append(confidence)
            if self._smoothed is None:
                self._smoothed = confidence
            else:
                self._smoothed = self._alpha * confidence + (1.0 - self._alpha) * self._smoothed
        elif self._smoothed is not None:
            pass
        if len(self._history) < self.config.min_samples:
            return confidence
        return self._smoothed if self._smoothed is not None else confidence

    def _is_outlier(self, value: float) -> bool:
        """Check if value is a statistical outlier."""
        if len(self._history) < 2:
            return False
        mean = sum(self._history) / len(self._history)
        variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
        std = variance ** 0.5
        if std == 0:
            return False
        z_score = abs(value - mean) / std
        return z_score > self.config.outlier_threshold

    def get_current(self) -> Optional[float]:
        """Get current smoothed confidence.

        Returns:
            Current smoothed confidence or None if not enough data.
        """
        if len(self._history) < self.config.min_samples:
            return None
        return self._smoothed

    def reset(self) -> None:
        """Reset smoother state."""
        self._history.clear()
        self._smoothed = None
