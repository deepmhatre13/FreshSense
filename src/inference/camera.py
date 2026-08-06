"""Webcam capture and management for real-time FreshSense inference."""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Camera"]


class Camera:
    """Manages webcam capture for real-time inference."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """Start camera capture."""
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            logger.error("Failed to open camera device %d", self.device_id)
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info("Camera started: %dx%d @ %.1f FPS", actual_width, actual_height, actual_fps)
        return True

    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        return frame

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")

    @staticmethod
    def list_cameras() -> List[int]:
        """List available camera devices."""
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available