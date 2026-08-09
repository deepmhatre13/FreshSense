"""Adaptive confidence threshold for FreshSense.

Dynamically adjusts classification confidence threshold based on:
- Recent prediction accuracy
- Confidence distribution
- Application requirements (speed vs accuracy trade-off)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ThresholdConfig:
    """Configuration for adaptive threshold.

    Attributes:
        initial_threshold: Starting confidence threshold.
        min_threshold: Minimum allowed threshold.
        max_threshold: Maximum allowed threshold.
        window_size: Number of recent predictions to consider.
        adjustment_rate: How quickly to adapt (0.0-1.0).
    """

    initial_threshold: float = 0.5
    min_threshold: float = 0.3
    max_threshold: float = 0.9
    window_size: int = 20
    adjustment_rate: float = 0.1


class AdaptiveThreshold:
    """Adaptive confidence threshold controller.

    Adjusts threshold based on recent performance to balance
    precision and recall.
    """

    def __init__(self, config: ThresholdConfig | None = None):
        self.config = config or ThresholdConfig()
        self._threshold = self.config.initial_threshold
        self._history: deque[tuple[float, bool]] = deque(maxlen=self.config.window_size)
        self._last_adjustment_time: float = 0.0
        self._adjustment_interval: float = 5.0

    @property
    def threshold(self) -> float:
        """Current confidence threshold."""
        return self._threshold

    def record_prediction(self, confidence: float, is_correct: bool) -> None:
        """Record a prediction result for threshold adaptation.

        Args:
            confidence: Prediction confidence.
            is_correct: Whether prediction was correct.
        """
        self._history.append((confidence, is_correct))
        current_time = time.time()
        if current_time - self._last_adjustment_time >= self._adjustment_interval:
            self._adjust_threshold()
            self._last_adjustment_time = current_time

    def should_accept(self, confidence: float) -> bool:
        """Check if confidence meets current threshold.

        Args:
            confidence: Prediction confidence.

        Returns:
            True if prediction should be accepted.
        """
        return confidence >= self._threshold

    def _adjust_threshold(self) -> None:
        """Adjust threshold based on recent performance."""
        if len(self._history) < self.config.window_size // 2:
            return
        confidences = [c for c, _ in self._history]
        correct = [c for c, cor in self._history if cor]
        incorrect = [c for c, cor in self._history if not cor]
        if not correct or not incorrect:
            return
        mean_correct = sum(correct) / len(correct)
        mean_incorrect = sum(incorrect) / len(incorrect)
        if mean_correct < self._threshold:
            self._threshold = max(
                self.config.min_threshold,
                self._threshold - self.config.adjustment_rate * 0.1,
            )
        elif mean_incorrect > self._threshold:
            self._threshold = min(
                self.config.max_threshold,
                self._threshold + self.config.adjustment_rate * 0.1,
            )

    def get_stats(self) -> dict:
        """Get threshold statistics.

        Returns:
            Dictionary with threshold stats.
        """
        if not self._history:
            return {"threshold": self._threshold, "num_samples": 0}
        confidences = [c for c, _ in self._history]
        correct = [c for c, cor in self._history if cor]
        return {
            "threshold": self._threshold,
            "num_samples": len(self._history),
            "mean_confidence": sum(confidences) / len(confidences),
            "accuracy": len(correct) / len(self._history) if self._history else 0.0,
        }

    def reset(self) -> None:
        """Reset threshold to initial value."""
        self._threshold = self.config.initial_threshold
        self._history.clear()
        self._last_adjustment_time = 0.0
