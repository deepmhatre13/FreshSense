"""Session statistics and logging for FreshSense Phase 3.

This module provides comprehensive statistics tracking and session logging
for real-time inference:

- Total frames, predictions, average confidence, FPS, latency
- Stable predictions, uncertain frames, motion skips, lighting warnings
- CSV session logging with timestamps
- JSON summary on exit
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["SessionStatistics", "SessionLogger"]


@dataclass
class SessionStatistics:
    """Statistics for a single inference session."""

    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    total_frames: int = 0
    total_predictions: int = 0
    stable_predictions: int = 0
    uncertain_frames: int = 0
    motion_skips: int = 0
    lighting_warnings: int = 0

    # Performance metrics
    confidences: List[float] = field(default_factory=list)
    fps_values: List[float] = field(default_factory=list)
    latency_values: List[float] = field(default_factory=list)

    # Quality metrics
    brightness_values: List[float] = field(default_factory=list)
    blur_values: List[float] = field(default_factory=list)

    def record_prediction(self, confidence: float, latency_ms: float, fps: float) -> None:
        """Record a successful prediction."""
        self.total_predictions += 1
        self.confidences.append(confidence)
        self.latency_values.append(latency_ms)
        self.fps_values.append(fps)

    def record_uncertain(self) -> None:
        """Record an uncertain frame."""
        self.uncertain_frames += 1

    def record_motion_skip(self) -> None:
        """Record a motion-based inference skip."""
        self.motion_skips += 1

    def record_lighting_warning(self) -> None:
        """Record a lighting quality warning."""
        self.lighting_warnings += 1

    def record_quality(self, brightness: float, blur_variance: float) -> None:
        """Record quality metrics."""
        self.brightness_values.append(brightness)
        self.blur_values.append(blur_variance)

    def finalize(self) -> None:
        """Mark session as ended."""
        self.end_time = time.perf_counter()

    @property
    def elapsed_time(self) -> float:
        """Total session duration in seconds."""
        end = self.end_time if self.end_time > 0 else time.perf_counter()
        return end - self.start_time

    @property
    def average_confidence(self) -> float:
        """Average prediction confidence."""
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    @property
    def average_fps(self) -> float:
        """Average FPS."""
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0.0

    @property
    def average_latency(self) -> float:
        """Average inference latency in ms."""
        return sum(self.latency_values) / len(self.latency_values) if self.latency_values else 0.0

    def to_dict(self) -> dict:
        """Convert statistics to dictionary."""
        return {
            "session_duration_seconds": round(self.elapsed_time, 2),
            "total_frames": self.total_frames,
            "total_predictions": self.total_predictions,
            "stable_predictions": self.stable_predictions,
            "uncertain_frames": self.uncertain_frames,
            "motion_skips": self.motion_skips,
            "lighting_warnings": self.lighting_warnings,
            "average_confidence": round(self.average_confidence, 4),
            "average_fps": round(self.average_fps, 2),
            "average_latency_ms": round(self.average_latency, 2),
            "avg_brightness": round(sum(self.brightness_values) / len(self.brightness_values), 2) if self.brightness_values else 0.0,
            "avg_blur_variance": round(sum(self.blur_values) / len(self.blur_values), 2) if self.blur_values else 0.0,
        }


class SessionLogger:
    """Logs inference session data to CSV and saves summary JSON."""

    def __init__(self, log_dir: str = "logs/session", save_logs: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.save_logs = save_logs
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path: Optional[Path] = None
        self.csv_file = None
        self.csv_writer = None

        if self.save_logs:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.log_dir / f"session_{self.session_id}.csv"
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "timestamp", "prediction", "confidence", "fps",
                "latency_ms", "brightness", "blur_variance", "warnings"
            ])
            logger.info("Session logging to: %s", self.csv_path)

    def log_frame(
        self,
        prediction: str,
        confidence: float,
        fps: float,
        latency_ms: float,
        brightness: float,
        blur_variance: float,
        warnings: Optional[List[str]] = None,
    ) -> None:
        """Log a single frame's data."""
        if not self.save_logs or self.csv_writer is None:
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        warnings_str = "; ".join(warnings) if warnings else ""
        self.csv_writer.writerow([
            timestamp,
            prediction,
            f"{confidence:.4f}",
            f"{fps:.2f}",
            f"{latency_ms:.2f}",
            f"{brightness:.2f}",
            f"{blur_variance:.2f}",
            warnings_str,
        ])

    def save_summary(self, stats: SessionStatistics) -> Optional[Path]:
        """Save session summary as JSON."""
        if not self.save_logs:
            return None

        if self.csv_file:
            self.csv_file.close()

        summary_path = self.log_dir / f"session_{self.session_id}_summary.json"
        summary = stats.to_dict()
        summary["session_id"] = self.session_id

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Session summary saved to: %s", summary_path)
        return summary_path

    def close(self) -> None:
        """Close log file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing SessionStatistics and SessionLogger...")
    print("=" * 60)

    # Create statistics
    stats = SessionStatistics()

    # Simulate some predictions
    for i in range(10):
        stats.record_prediction(
            confidence=0.8 + i * 0.01,
            latency_ms=20.0 + i * 0.5,
            fps=28.0 + i * 0.1,
        )
        stats.record_quality(brightness=128.0, blur_variance=150.0)

    stats.record_uncertain()
    stats.record_motion_skip()
    stats.record_lighting_warning()
    stats.finalize()

    print("\nStatistics:")
    for key, value in stats.to_dict().items():
        print(f"  {key}: {value}")

    # Test logger
    logger_test = SessionLogger(save_logs=True)
    for i in range(5):
        logger_test.log_frame(
            prediction="fresh",
            confidence=0.85 + i * 0.02,
            fps=29.0,
            latency_ms=18.0,
            brightness=120.0,
            blur_variance=200.0,
            warnings=[],
        )

    summary_path = logger_test.save_summary(stats)
    print(f"\nSummary saved to: {summary_path}")
    logger_test.close()
