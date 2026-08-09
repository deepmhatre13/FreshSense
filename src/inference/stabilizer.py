"""Temporal prediction stabilization for FreshSense Phase 3.

This module provides production-grade temporal smoothing to eliminate
prediction flickering in real-time inference:

- Rolling prediction history
- Exponential Moving Average (EMA) for confidence smoothing
- Majority voting for label stability
- Confidence averaging
- Prediction locking to prevent instant class switching

The Stabilizer class is completely independent from the Predictor and
can be used as a drop-in wrapper around any prediction source.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Stabilizer", "StabilizerConfig", "StabilizedPrediction"]


@dataclass(frozen=True)
class StabilizerConfig:
    """Configuration for temporal prediction stabilization.

    Attributes:
        ema_alpha: Exponential moving average smoothing factor (0.0-1.0).
        vote_window: Number of recent predictions for majority voting.
        lock_frames: Consecutive conflicting predictions required to switch class.
        confidence_threshold: Minimum smoothed confidence to accept prediction.
    """

    ema_alpha: float = 0.2
    vote_window: int = 15
    lock_frames: int = 5
    confidence_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0.0, 1.0].")
        if self.vote_window <= 0:
            raise ValueError("vote_window must be positive.")
        if self.lock_frames <= 0:
            raise ValueError("lock_frames must be positive.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0].")


@dataclass(frozen=True)
class StabilizedPrediction:
    """Smoothed prediction result from the stabilizer.

    Attributes:
        label: Stabilized prediction label after temporal smoothing.
        confidence: Smoothed confidence score after EMA (0.0-1.0).
        raw_label: Most recent raw prediction label from the model.
        raw_confidence: Most recent raw confidence score from the model.
        ema_confidence: Confidence after exponential moving average.
        majority_label: Label determined by majority voting.
        vote_counts: Vote count distribution across classes.
        is_locked: True if prediction is locked (class switch pending).
        lock_count: Number of consecutive conflicting predictions seen.
        is_uncertain: True if confidence is below threshold.
        window_size: Actual number of frames in the voting window.
    """

    label: str
    confidence: float
    raw_label: str
    raw_confidence: float
    ema_confidence: float
    majority_label: str
    vote_counts: dict
    is_locked: bool
    lock_count: int
    is_uncertain: bool
    window_size: int


class Stabilizer:
    """Temporal prediction stabilizer for real-time inference.

    This class applies multiple smoothing techniques to raw predictions.

    Args:
        config: StabilizerConfig instance with smoothing settings.
        class_names: List of valid class names for validation.
    """

    def __init__(self, config: StabilizerConfig, class_names: Optional[List[str]] = None) -> None:
        self.config = config
        self.class_names = class_names or []
        self._lock = Lock()

        # EMA state
        self._ema_confidence: float = 0.0
        self._ema_initialized: bool = False

        # Prediction history for majority voting
        self._history: Deque[Tuple[str, float]] = deque(maxlen=config.vote_window)

        # Lock state
        self._locked_label: str = ""
        self._lock_count: int = 0

        # Counters
        self.total_updates: int = 0
        self.total_locks: int = 0
        self.total_switches: int = 0
    def update(self, label: str, confidence: float) -> StabilizedPrediction:
        """Process a new raw prediction and return stabilized result.

        Args:
            label: Raw prediction label from the model.
            confidence: Raw confidence score (0.0-1.0).

        Returns:
            StabilizedPrediction with smoothed values and stability info.
        """
        with self._lock:
            self.total_updates += 1

            # Validate input
            if not label or confidence < 0.0 or confidence > 1.0:
                logger.warning("Invalid prediction: label=%s, confidence=%.3f", label, confidence)
                label = label or "unknown"
                confidence = max(0.0, min(1.0, confidence))

            # Store raw prediction in history
            self._history.append((label, confidence))

            # 1. EMA smoothing for confidence
            ema_confidence = self._compute_ema(confidence)

            # 2. Majority voting for label
            majority_label, vote_counts = self._compute_majority_vote()

            # 3. Prediction locking logic.
            # Lock against the raw incoming label (not the majority vote) so a
            # single conflicting raw prediction triggers the lock, requiring
            # lock_frames consecutive conflicting labels to actually switch the class.
            locked_label, lock_count, is_locked, switched = self._apply_lock(label)

            # Update lock stats
            if switched:
                self.total_switches += 1
            if is_locked:
                self.total_locks += 1

            # Determine final label (locked or majority)
            final_label = locked_label if is_locked else majority_label

            # Determine if uncertain
            is_uncertain = ema_confidence < self.config.confidence_threshold

            # Use EMA confidence as the primary confidence measure
            final_confidence = ema_confidence

            return StabilizedPrediction(
                label=final_label,
                confidence=final_confidence,
                raw_label=label,
                raw_confidence=confidence,
                ema_confidence=ema_confidence,
                majority_label=majority_label,
                vote_counts=vote_counts,
                is_locked=is_locked,
                lock_count=lock_count,
                is_uncertain=is_uncertain,
                window_size=len(self._history),
            )
    def _compute_ema(self, new_confidence: float) -> float:
        """Compute exponential moving average of confidence."""
        alpha = self.config.ema_alpha

        if not self._ema_initialized:
            self._ema_confidence = new_confidence
            self._ema_initialized = True
        else:
            self._ema_confidence = alpha * new_confidence + (1.0 - alpha) * self._ema_confidence

        return self._ema_confidence

    def _compute_majority_vote(self) -> Tuple[str, dict]:
        """Compute majority vote from prediction history."""
        if not self._history:
            return self.class_names[0] if self.class_names else "unknown", {}

        vote_counts: dict = {}
        for label, _ in self._history:
            vote_counts[label] = vote_counts.get(label, 0) + 1

        majority_label = max(vote_counts, key=vote_counts.get)
        return majority_label, vote_counts

    def _apply_lock(self, proposed_label: str) -> Tuple[str, int, bool, bool]:
        """Apply prediction locking logic.

        Returns:
            Tuple of (final_label, lock_count, is_locked, switched).
        """
        if not self._locked_label:
            self._locked_label = proposed_label
            self._lock_count = 0
            return proposed_label, 0, False, False

        if proposed_label == self._locked_label:
            self._lock_count = 0
            return self._locked_label, 0, False, False

        self._lock_count += 1

        if self._lock_count >= self.config.lock_frames:
            old_label = self._locked_label
            self._locked_label = proposed_label
            self._lock_count = 0
            logger.debug("Prediction lock released: %s -> %s", old_label, proposed_label)
            return proposed_label, 0, False, True

        return self._locked_label, self._lock_count, True, False
    def get_stats(self) -> dict:
        """Get stabilizer statistics."""
        with self._lock:
            vote_counts = {}
            if self._history:
                for label, _ in self._history:
                    vote_counts[label] = vote_counts.get(label, 0) + 1

            return {
                "total_updates": self.total_updates,
                "total_switches": self.total_switches,
                "total_locks": self.total_locks,
                "current_locked_label": self._locked_label,
                "current_lock_count": self._lock_count,
                "window_size": len(self._history),
                "vote_counts": vote_counts,
                "ema_confidence": self._ema_confidence,
            }

    def reset(self) -> None:
        """Reset all stabilizer state."""
        with self._lock:
            self._ema_confidence = 0.0
            self._ema_initialized = False
            self._history.clear()
            self._locked_label = ""
            self._lock_count = 0
            self.total_updates = 0
            self.total_locks = 0
            self.total_switches = 0
        logger.info("Stabilizer reset.")

    def __str__(self) -> str:
        """Return human-readable string of stabilizer state."""
        stats = self.get_stats()
        return (
            f"Stabilizer("
            f"locked={stats['current_locked_label']}, "
            f"lock_count={stats['current_lock_count']}, "
            f"window={stats['window_size']}/{self.config.vote_window}, "
            f"ema={stats['ema_confidence']:.3f}, "
            f"switches={stats['total_switches']})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class_names = ["fresh", "stale", "rotten"]
    config = StabilizerConfig(
        ema_alpha=0.2,
        vote_window=15,
        lock_frames=5,
        confidence_threshold=0.70,
    )
    stabilizer = Stabilizer(config, class_names)

    print("Simulating predictions with flickering...")
    print("=" * 60)

    predictions = [
        ("fresh", 0.92),
        ("fresh", 0.88),
        ("fresh", 0.90),
        ("rotten", 0.75),
        ("rotten", 0.72),
        ("fresh", 0.85),
        ("fresh", 0.89),
        ("fresh", 0.91),
        ("fresh", 0.93),
        ("fresh", 0.90),
    ]

    for i, (label, conf) in enumerate(predictions):
        result = stabilizer.update(label, conf)
        print(
            f"Frame {i + 1:2d}: raw={label}({conf:.2f}) -> "
            f"stabilized={result.label}({result.confidence:.2f}), "
            f"majority={result.majority_label}, "
            f"locked={result.is_locked}, "
            f"lock_count={result.lock_count}, "
            f"uncertain={result.is_uncertain}"
        )

    print("=" * 60)
    print("\nFinal stabilizer state:")
    print(stabilizer)
