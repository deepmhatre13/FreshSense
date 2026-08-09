"""FPS monitoring and performance tracking for FreshSense Phase 2.

This module provides real-time performance monitoring for the inference pipeline:

- Current FPS calculation
- Average FPS over rolling window
- Inference latency tracking
- Frame processing time measurement
- Rolling statistics with configurable window size
- Thread-safe operations for multi-threaded environments

The FPSMonitor class is designed to be lightweight and non-blocking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["FPSMonitor", "FPSStats"]


@dataclass(frozen=True)
class FPSConfig:
    """Configuration for FPS monitoring.

    Attributes:
        window_size: Number of frames to use for rolling average.
        update_interval: Minimum seconds between FPS updates.
        log_interval: Minimum seconds between FPS log messages.
    """

    window_size: int = 30
    update_interval: float = 0.5
    log_interval: float = 5.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")


@dataclass(frozen=True)
class FPSStats:
    """Snapshot of FPS statistics.

    Attributes:
        current_fps: Instantaneous FPS based on last frame.
        average_fps: Rolling average FPS over window.
        min_fps: Minimum FPS in current window.
        max_fps: Maximum FPS in current window.
        avg_latency_ms: Average inference latency in milliseconds.
        avg_frame_time_ms: Average total frame processing time in milliseconds.
        total_frames: Total number of frames processed.
        elapsed_time: Total elapsed time in seconds.
    """

    current_fps: float
    average_fps: float
    min_fps: float
    max_fps: float
    avg_latency_ms: float
    avg_frame_time_ms: float
    total_frames: int
    elapsed_time: float


class FPSMonitor:
    """Real-time FPS and performance monitoring.

    This class tracks:
    - Frame timestamps for FPS calculation
    - Inference latency
    - Frame processing time
    - Rolling statistics

    Thread-safe for concurrent access.

    Args:
        config: FPSConfig instance with monitoring settings.
    """

    def __init__(self, config: FPSConfig) -> None:
        self.config = config
        self._lock = Lock()

        # Timing data
        self._frame_times: List[float] = []
        self._latencies: List[float] = []
        self._frame_processing_times: List[float] = []

        # Counters
        self.total_frames: int = 0
        self.start_time: float = time.perf_counter()
        self.last_frame_time: float = 0.0
        self.last_inference_start: float = 0.0
        self.last_inference_end: float = 0.0

        # Cached stats
        self._current_fps: float = 0.0
        self._last_log_time: float = 0.0

    def start_frame(self) -> float:
        """Mark the start of a new frame.

        Call this at the beginning of each frame processing cycle.

        Returns:
            Timestamp of frame start.
        """
        now = time.perf_counter()
        self.last_inference_start = now
        return now

    def end_frame(self, inference_complete: bool = True) -> float:
        """Mark the end of frame processing.

        Call this after all processing (including inference and overlay) is complete.

        Args:
            inference_complete: If True, record inference latency.

        Returns:
            Timestamp of frame end.
        """
        now = time.perf_counter()
        self.last_inference_end = now

        with self._lock:
            self.total_frames += 1

            # Record frame time
            if self.last_frame_time > 0:
                frame_time = now - self.last_frame_time
                self._frame_times.append(frame_time)
                self._frame_processing_times.append(now - self.last_inference_start)

                # Keep window bounded
                if len(self._frame_times) > self.config.window_size:
                    self._frame_times.pop(0)
                    self._frame_processing_times.pop(0)

            # Record inference latency
            if inference_complete and self.last_inference_start > 0:
                latency = now - self.last_inference_start
                self._latencies.append(latency)
                if len(self._latencies) > self.config.window_size:
                    self._latencies.pop(0)

            self.last_frame_time = now

            # Calculate current FPS
            if self._frame_times:
                recent_frame_time = self._frame_times[-1]
                self._current_fps = 1.0 / recent_frame_time if recent_frame_time > 0 else 0.0

        return now

    def get_stats(self) -> FPSStats:
        """Get current FPS and performance statistics.

        Returns:
            FPSStats dataclass with current metrics.
        """
        with self._lock:
            elapsed = time.perf_counter() - self.start_time

            # Calculate rolling average FPS
            if self._frame_times:
                avg_frame_time = sum(self._frame_times) / len(self._frame_times)
                avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                min_fps = 1.0 / max(self._frame_times) if self._frame_times else 0.0
                max_fps = 1.0 / min(self._frame_times) if self._frame_times else 0.0
            else:
                avg_fps = 0.0
                min_fps = 0.0
                max_fps = 0.0

            # Calculate average latency
            avg_latency_ms = (
                (sum(self._latencies) / len(self._latencies)) * 1000.0
                if self._latencies
                else 0.0
            )

            # Calculate average frame processing time
            avg_frame_time_ms = (
                (sum(self._frame_processing_times) / len(self._frame_processing_times)) * 1000.0
                if self._frame_processing_times
                else 0.0
            )

            return FPSStats(
                current_fps=self._current_fps,
                average_fps=avg_fps,
                min_fps=min_fps,
                max_fps=max_fps,
                avg_latency_ms=avg_latency_ms,
                avg_frame_time_ms=avg_frame_time_ms,
                total_frames=self.total_frames,
                elapsed_time=elapsed,
            )

    def should_update(self) -> bool:
        """Check if enough time has passed for a stats update.

        Returns:
            True if update interval has elapsed.
        """
        return (time.perf_counter() - self._last_log_time) >= self.config.update_interval

    def should_log(self) -> bool:
        """Check if enough time has passed for a log message.

        Returns:
            True if log interval has elapsed.
        """
        return (time.perf_counter() - self._last_log_time) >= self.config.log_interval

    def mark_updated(self) -> None:
        """Mark that stats were updated (reset timers)."""
        self._last_log_time = time.perf_counter()

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._frame_times.clear()
            self._latencies.clear()
            self._frame_processing_times.clear()
            self.total_frames = 0
            self.start_time = time.perf_counter()
            self.last_frame_time = 0.0
            self.last_inference_start = 0.0
            self.last_inference_end = 0.0
            self._current_fps = 0.0
            self._last_log_time = 0.0
        logger.info("FPS monitor reset.")

    def get_formatted_stats(self) -> dict:
        """Get formatted statistics for display.

        Returns:
            Dictionary with formatted metric names and values.
        """
        stats = self.get_stats()
        return {
            "FPS": f"{stats.current_fps:.1f}",
            "Avg FPS": f"{stats.average_fps:.1f}",
            "Latency": f"{stats.avg_latency_ms:.1f}ms",
            "Frame Time": f"{stats.avg_frame_time_ms:.1f}ms",
            "Frames": str(stats.total_frames),
            "Time": f"{stats.elapsed_time:.0f}s",
        }

    def __str__(self) -> str:
        """Return human-readable string of current stats."""
        stats = self.get_stats()
        return (
            f"FPSMonitor("
            f"current={stats.current_fps:.1f}, "
            f"avg={stats.average_fps:.1f}, "
            f"latency={stats.avg_latency_ms:.1f}ms, "
            f"frames={stats.total_frames})"
        )


if __name__ == "__main__":
    # Quick self-test.
    logging.basicConfig(level=logging.INFO)

    config = FPSConfig(window_size=10, update_interval=0.5, log_interval=2.0)
    monitor = FPSMonitor(config)

    print("Simulating 100 frames at ~30 FPS...")
    for i in range(100):
        monitor.start_frame()

        # Simulate inference time
        time.sleep(0.025)  # ~40ms inference

        monitor.end_frame(inference_complete=True)

        # Print stats every 20 frames
        if (i + 1) % 20 == 0:
            stats = monitor.get_stats()
            print(
                f"\nFrame {i + 1}: "
                f"FPS={stats.current_fps:.1f}, "
                f"Avg FPS={stats.average_fps:.1f}, "
                f"Latency={stats.avg_latency_ms:.1f}ms, "
                f"Frame Time={stats.avg_frame_time_ms:.1f}ms"
            )

    print("\nFinal stats:")
    final_stats = monitor.get_stats()
    for key, value in final_stats.__dict__.items():
        print(f"  {key}: {value}")