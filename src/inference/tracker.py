"""Prediction tracker with temporal smoothing for FreshSense Phase 2.

This module provides prediction tracking and smoothing to prevent flickering
in real-time inference:

- Sliding window majority voting
- Confidence smoothing with exponential moving average
- Temporal consistency enforcement
- Configurable window size and smoothing factor
- Thread-safe operations

The PredictionTracker class maintains a history of predictions and applies
temporal smoothing to produce stable, consistent outputs.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PredictionTracker", "TrackerConfig", "TrackedPrediction"]


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for prediction tracking.

    Attributes:
        window_size: Number of frames to keep in sliding window.
        smoothing_factor: Exponential moving average factor (0.0-1.0).
            Higher values give more weight to recent predictions.
        confidence_threshold: Minimum average confidence to accept prediction.
        min_consistency: Minimum ratio of consistent predictions (0.0-1.0).
    """

    window_size: int = 5
    smoothing_factor: float = 0.3
    confidence_threshold: float = 0.5
    min_consistency: float = 0.6

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if not 0.0 <= self.smoothing_factor <= 1.0:
            raise ValueError("smoothing_factor must be in [0.0, 1.0].")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0].")
        if not 0.0 <= self.min_consistency <= 1.0:
            raise ValueError("min_consistency must be in [0.0, 1.0].")


@dataclass(frozen=True)
class TrackedPrediction:
    """Smoothed prediction result from the tracker.

    Attributes:
        label: Predicted class label.
        confidence: Smoothed confidence score (0.0-1.0).
        raw_label: Most recent raw prediction label.
        raw_confidence: Most recent raw confidence score.
        consistency: Ratio of consistent predictions in window (0.0-1.0).
        window_size: Actual number of frames in window.
        is_stable: True if prediction meets stability criteria.
    """

    label: str
    confidence: float
    raw_label: str
    raw_confidence: float
    consistency: float
    window_size: int
    is_stable: bool


class PredictionTracker:
    """Tracks predictions over time with temporal smoothing.

    This class maintains a sliding window of recent predictions and applies:
    - Majority voting for label stability
    - Exponential moving average for confidence smoothing
    - Consistency checking to filter out flickering predictions

    Thread-safe for concurrent access.

    Args:
        config: TrackerConfig instance with tracking settings.
        class_names: List of class names for label mapping.
    """

    def __init__(self, config: TrackerConfig, class_names: List[str]) -> None:
        self.config = config
        self.class_names = class_names
        self._lock = Lock()

        # Sliding window of predictions
        self._window: Deque[Tuple[str, float]] = deque(maxlen=config.window_size)

        # Smoothed confidence (exponential moving average)
        self._smoothed_confidence: float = 0.0
        self._initialized: bool = False

        # Last prediction
        self._last_label: str = class_names[0] if class_names else "unknown"
        self._last_confidence: float = 0.0

    def update(self, label: str, confidence: float) -> TrackedPrediction:
        """Update tracker with new prediction.

        Args:
            label: Predicted class label.
            confidence: Prediction confidence (0.0-1.0).

        Returns:
            TrackedPrediction with smoothed results and stability metrics.
        """
        with self._lock:
            # Add to sliding window
            self._window.append((label, confidence))

            # Update smoothed confidence (EMA)
            if not self._initialized:
                self._smoothed_confidence = confidence
                self._initialized = True
            else:
                self._smoothed_confidence = (
                    self.config.smoothing_factor * confidence
                    + (1.0 - self.config.smoothing_factor) * self._smoothed_confidence
                )

            # Calculate metrics
            consistency = self._calculate_consistency()
            majority_label = self._majority_vote()
            is_stable = self._check_stability(consistency)

            # Update last prediction
            self._last_label = majority_label
            self._last_confidence = self._smoothed_confidence

            return TrackedPrediction(
                label=majority_label,
                confidence=self._smoothed_confidence,
                raw_label=label,
                raw_confidence=confidence,
                consistency=consistency,
                window_size=len(self._window),
                is_stable=is_stable,
            )

    def _calculate_consistency(self) -> float:
        """Calculate consistency ratio in current window.

        Returns:
            Ratio of most common label to total frames (0.0-1.0).
        """
        if not self._window:
            return 0.0

        # Count label occurrences
        label_counts: Dict[str, int] = {}
        for label, _ in self._window:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Return ratio of most common label
        max_count = max(label_counts.values())
        return max_count / len(self._window)

    def _majority_vote(self) -> str:
        """Get majority vote label from window.

        Returns:
            Most common label in window.
        """
        if not self._window:
            return self._last_label

        # Count label occurrences
        label_counts: Dict[str, int] = {}
        for label, _ in self._window:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Return most common label
        return max(label_counts.items(), key=lambda x: x[1])[0]

    def _check_stability(self, consistency: float) -> bool:
        """Check if prediction meets stability criteria.

        Args:
            consistency: Consistency ratio (0.0-1.0).

        Returns:
            True if prediction is stable enough to display.
        """
        return (
            consistency >= self.config.min_consistency
            and self._smoothed_confidence >= self.config.confidence_threshold
        )

    def get_last_prediction(self) -> Optional[Tuple[str, float]]:
        """Get last smoothed prediction.

        Returns:
            Tuple of (label, confidence) or None if no predictions yet.
        """
        if not self._initialized:
            return None
        return self._last_label, self._last_confidence

    def get_window_stats(self) -> Dict[str, float]:
        """Get statistics about current prediction window.

        Returns:
            Dictionary with window statistics.
        """
        if not self._window:
            return {
                "window_size": 0,
                "consistency": 0.0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            }

        confidences = [conf for _, conf in self._window]
        consistency = self._calculate_consistency()

        return {
            "window_size": len(self._window),
            "consistency": consistency,
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        }

    def reset(self) -> None:
        """Reset tracker state."""
        with self._lock:
            self._window.clear()
            self._smoothed_confidence = 0.0
            self._initialized = False
            self._last_label = self.class_names[0] if self.class_names else "unknown"
            self._last_confidence = 0.0
        logger.info("Prediction tracker reset.")

    def __str__(self) -> str:
        """Return human-readable string of tracker state."""
        stats = self.get_window_stats()
        return (
            f"PredictionTracker("
            f"label={self._last_label}, "
            f"confidence={self._last_confidence:.3f}, "
            f"consistency={stats['consistency']:.2f}, "
            f"window={stats['window_size']}/{self.config.window_size})"
        )


if __name__ == "__main__":
    # Quick self-test.
    logging.basicConfig(level=logging.INFO)

    class_names = ["fresh", "stale", "rotten"]
    config = TrackerConfig(window_size=5, smoothing_factor=0.3)
    tracker = PredictionTracker(config, class_names)

    print("Simulating predictions with flickering...")
    predictions = [
        ("fresh", 0.9),
        ("fresh", 0.85),
        ("stale", 0.7),  # Flicker
        ("fresh", 0.88),
        ("fresh", 0.92),
        ("fresh", 0.95),
    ]

    for i, (label, conf) in enumerate(predictions):
        result = tracker.update(label, conf)
        print(
            f"Frame {i + 1}: raw={label}({conf:.2f}) -> "
            f"smoothed={result.label}({result.confidence:.2f}), "
            f"consistency={result.consistency:.2f}, stable={result.is_stable}"
        )

    print("\nFinal tracker state:")
    print(tracker)

    print("\nWindow stats:")
    stats = tracker.get_window_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")