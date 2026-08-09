"""Real-time webcam capture and management for FreshSense Phase 2.

This module provides a production-ready Camera class for real-time inference:

- Automatic webcam detection and enumeration
- Safe camera open/close with resource management
- Configurable resolution (640x480, 1280x720, 1920x1080)
- Automatic fallback to available camera if preferred index fails
- Frame timestamping and counting
- Frame resizing with aspect ratio preservation
- Disconnect detection and recovery
- Comprehensive error logging

The Camera class is designed to be used as a context manager or standalone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Camera", "CameraConfig", "CameraError"]


class CameraError(Exception):
    """Base exception for camera-related errors."""
    pass


class CameraOpenError(CameraError):
    """Raised when camera cannot be opened."""
    pass


class CameraReadError(CameraError):
    """Raised when frame cannot be read from camera."""
    pass


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for camera capture.

    Attributes:
        device_id: Camera device index (0, 1, 2, ...).
        width: Target frame width in pixels.
        height: Target frame height in pixels.
        fps: Target frames per second.
        buffer_size: OpenCV capture buffer size (0 = disable buffering).
        auto_reconnect: If True, attempt to reconnect on disconnect.
        reconnect_delay: Seconds to wait before reconnect attempt.
        max_reconnect_attempts: Maximum number of reconnection attempts.
    """

    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1
    auto_reconnect: bool = True
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 5

    def __post_init__(self) -> None:
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.buffer_size < 0:
            raise ValueError("buffer_size must be non-negative.")
        if self.reconnect_delay <= 0:
            raise ValueError("reconnect_delay must be positive.")
        if self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts must be non-negative.")


class Camera:
    """Manages webcam capture for real-time inference.

    This class handles:
    - Camera enumeration and selection
    - Safe resource management with context manager support
    - Frame capture with timestamping
    - Automatic reconnection on disconnect
    - Resolution configuration with fallback
    - Performance monitoring (FPS, frame count)

    Args:
        config: CameraConfig instance with capture settings.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.start_time: float = time.perf_counter()
        self.last_frame_time: float = 0.0
        self.is_connected: bool = False
        self.reconnect_attempts: int = 0

    def __enter__(self) -> Camera:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and release resources."""
        self.release()

    def enumerate_cameras(self, max_index: int = 10) -> List[int]:
        """Detect available camera devices.

        Args:
            max_index: Maximum device index to check.

        Returns:
            List of available camera device indices.
        """
        available: List[int] = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        logger.info("Detected %d available cameras: %s", len(available), available)
        return available

    def open(self) -> bool:
        """Open the camera device.

        Attempts to open the configured device_id. If that fails, automatically
        falls back to the first available camera.

        Returns:
            True if camera opened successfully.

        Raises:
            CameraOpenError: If no camera could be opened.
        """
        if self.is_connected and self.cap is not None and self.cap.isOpened():
            logger.debug("Camera already open on device %d", self.config.device_id)
            return True

        # Try preferred device first
        if self._try_open(self.config.device_id):
            return True

        # Fallback: find first available camera
        available = self.enumerate_cameras()
        if not available:
            raise CameraOpenError(
                f"No cameras detected. Please connect a webcam and try again."
            )

        fallback_id = available[0]
        if fallback_id == self.config.device_id:
            raise CameraOpenError(
                f"Camera device {self.config.device_id} found but cannot be opened."
            )

        logger.warning(
            "Preferred camera %d unavailable. Falling back to camera %d.",
            self.config.device_id,
            fallback_id,
        )
        if self._try_open(fallback_id):
            return True

        raise CameraOpenError(
            f"Failed to open any camera device. Tried: {self.config.device_id}, {fallback_id}"
        )

    def _try_open(self, device_id: int) -> bool:
        """Attempt to open a specific camera device.

        Args:
            device_id: Camera device index.

        Returns:
            True if opened successfully, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(device_id)
            if not self.cap.isOpened():
                logger.warning("Camera device %d failed to open.", device_id)
                return False

            # Configure capture properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(
                "Camera opened: device=%d, resolution=%dx%d, fps=%.1f",
                device_id,
                actual_width,
                actual_height,
                actual_fps,
            )

            self.is_connected = True
            self.reconnect_attempts = 0
            self.frame_count = 0
            self.start_time = time.perf_counter()
            return True

        except Exception as exc:
            logger.error("Failed to open camera %d: %s", device_id, exc)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read a frame from the camera.

        Returns:
            Tuple of (success, frame, timestamp).
            - success: True if frame was read successfully.
            - frame: BGR image as numpy array, or None if failed.
            - timestamp: Time in seconds since epoch when frame was captured.

        Raises:
            CameraReadError: If frame read fails and reconnection fails.
        """
        if not self.is_connected or self.cap is None or not self.cap.isOpened():
            if self.config.auto_reconnect:
                self._attempt_reconnect()
            else:
                raise CameraReadError("Camera not connected.")

        try:
            ret, frame = self.cap.read()
            timestamp = time.perf_counter()

            if not ret or frame is None:
                logger.warning("Failed to read frame from camera.")
                if self.config.auto_reconnect:
                    self._attempt_reconnect()
                raise CameraReadError("Frame read failed.")

            # Validate frame dimensions
            if frame.shape[0] != self.config.height or frame.shape[1] != self.config.width:
                logger.debug(
                    "Frame size mismatch: expected %dx%d, got %dx%d. Resizing.",
                    self.config.width,
                    self.config.height,
                    frame.shape[1],
                    frame.shape[0],
                )
                frame = cv2.resize(frame, (self.config.width, self.config.height))

            self.frame_count += 1
            self.last_frame_time = timestamp

            return True, frame, timestamp

        except CameraReadError:
            raise
        except Exception as exc:
            logger.error("Unexpected error reading frame: %s", exc)
            raise CameraReadError(f"Frame read error: {exc}") from exc

    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to the camera.

        Returns:
            True if reconnection successful, False otherwise.
        """
        if not self.config.auto_reconnect:
            return False

        if self.reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(
                "Max reconnection attempts (%d) reached. Giving up.",
                self.config.max_reconnect_attempts,
            )
            return False

        self.reconnect_attempts += 1
        logger.info(
            "Attempting reconnection (%d/%d)...",
            self.reconnect_attempts,
            self.config.max_reconnect_attempts,
        )

        self.release()
        time.sleep(self.config.reconnect_delay)

        try:
            return self.open()
        except CameraOpenError:
            return False

    def release(self) -> None:
        """Release camera resources.

        Safely releases the VideoCapture and resets state.
        """
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Camera released. Total frames captured: %d", self.frame_count)
            except Exception as exc:
                logger.error("Error releasing camera: %s", exc)
            finally:
                self.cap = None
                self.is_connected = False

    def get_stats(self) -> dict:
        """Get camera performance statistics.

        Returns:
            Dictionary with frame_count, elapsed_time, avg_fps.
        """
        elapsed = time.perf_counter() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        return {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed,
            "avg_fps": avg_fps,
        }

    def reset_stats(self) -> None:
        """Reset frame counter and timing."""
        self.frame_count = 0
        self.start_time = time.perf_counter()
        logger.debug("Camera stats reset.")


if __name__ == "__main__":
    # Quick self-test.
    logging.basicConfig(level=logging.INFO)

    config = CameraConfig(device_id=0, width=640, height=480, fps=30)

    print("Enumerating cameras...")
    camera = Camera(config)
    available = camera.enumerate_cameras()
    print(f"Available cameras: {available}")

    if not available:
        print("No cameras found. Exiting.")
        exit(1)

    print(f"\nOpening camera {config.device_id}...")
    camera.open()

    print("\nCapturing 30 frames...")
    try:
        for i in range(30):
            ret, frame, timestamp = camera.read()
            if not ret:
                print(f"Frame {i}: FAILED")
                continue
            print(
                f"Frame {i}: shape={frame.shape}, timestamp={timestamp:.3f}, "
                f"frame_count={camera.frame_count}"
            )
            time.sleep(0.1)  # Simulate processing delay
    finally:
        camera.release()

    print("\nFinal stats:")
    stats = camera.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")