"""Image quality assessment for FreshSense Phase 3.

This module provides image quality metrics for real-time inference:

- Brightness detection (average frame intensity)
- Contrast estimation (standard deviation of intensities)
- Blur detection (Laplacian variance)
- Motion detection (frame-difference estimation)

Poor image quality can lead to unreliable predictions. This module
provides early warning signals to skip inference or alert the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["QualityAssessor", "QualityConfig", "QualityReport"]


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for image quality assessment.

    Attributes:
        brightness_min: Minimum acceptable average brightness (0-255).
        brightness_max: Maximum acceptable average brightness (0-255).
        blur_threshold: Minimum Laplacian variance for non-blurry image.
        contrast_min: Minimum acceptable contrast (std dev, 0-255).
        motion_threshold: Maximum frame difference for static scene.
        use_motion_detection: If True, compute and check motion.
        use_quality_warning: If True, display quality warnings.
    """

    brightness_min: int = 40
    brightness_max: int = 220
    blur_threshold: float = 100.0
    contrast_min: float = 20.0
    motion_threshold: float = 35.0
    use_motion_detection: bool = True
    use_quality_warning: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.brightness_min <= 255:
            raise ValueError("brightness_min must be in [0, 255].")
        if not 0 <= self.brightness_max <= 255:
            raise ValueError("brightness_max must be in [0, 255].")
        if self.brightness_min >= self.brightness_max:
            raise ValueError("brightness_min must be less than brightness_max.")
        if self.blur_threshold <= 0:
            raise ValueError("blur_threshold must be positive.")
        if self.contrast_min < 0:
            raise ValueError("contrast_min must be non-negative.")
        if self.motion_threshold < 0:
            raise ValueError("motion_threshold must be non-negative.")


@dataclass(frozen=True)
class QualityReport:
    """Image quality assessment report.

    Attributes:
        brightness: Average frame brightness (0-255).
        contrast: Frame contrast (standard deviation, 0-255).
        blur_variance: Laplacian variance for blur detection.
        motion_detected: True if significant motion detected.
        motion_score: Normalized motion score (0.0-1.0+).
        is_brightness_ok: True if brightness is within acceptable range.
        is_contrast_ok: True if contrast is above minimum.
        is_blur_ok: True if blur variance is above threshold.
        is_motion_ok: True if motion is below threshold.
        is_quality_ok: Overall quality assessment.
        warnings: List of quality warning messages.
        quality_score: Overall quality score (0.0-1.0).
    """

    brightness: float
    contrast: float
    blur_variance: float
    motion_detected: bool
    motion_score: float
    is_brightness_ok: bool
    is_contrast_ok: bool
    is_blur_ok: bool
    is_motion_ok: bool
    is_quality_ok: bool
    warnings: list
    quality_score: float


class QualityAssessor:
    """Assesses image quality for real-time inference.

    This class computes various quality metrics and determines if the
    frame is suitable for reliable inference.

    Args:
        config: QualityConfig instance with assessment settings.
    """

    def __init__(self, config: QualityConfig) -> None:
        self.config = config
        self._previous_frame: Optional[np.ndarray] = None
        self._lock = False  # Simple lock flag for motion detection

    def assess(self, frame: np.ndarray) -> QualityReport:
        """Assess the quality of a frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            QualityReport with all metrics and warnings.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Brightness
        brightness = float(np.mean(gray))
        is_brightness_ok = self.config.brightness_min <= brightness <= self.config.brightness_max

        # 2. Contrast
        contrast = float(np.std(gray))
        is_contrast_ok = contrast >= self.config.contrast_min

        # 3. Blur (Laplacian variance)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blur_ok = blur_variance >= self.config.blur_threshold

        # 4. Motion detection
        motion_detected = False
        motion_score = 0.0
        is_motion_ok = True

        if self.config.use_motion_detection and self._previous_frame is not None:
            prev_gray = cv2.cvtColor(self._previous_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = float(np.mean(diff)) / 255.0
            motion_detected = motion_score > self.config.motion_threshold / 100.0
            is_motion_ok = not motion_detected

        # Store current frame for next motion check
        self._previous_frame = frame.copy()

        # Build warnings
        warnings = []
        if not is_brightness_ok:
            if brightness < self.config.brightness_min:
                warnings.append("Lighting Too Low")
            else:
                warnings.append("Lighting Too Bright")
        if not is_contrast_ok:
            warnings.append("Low Contrast")
        if not is_blur_ok:
            warnings.append("Image Blurry")
        if motion_detected:
            warnings.append("Hold fruit still")

        # Overall quality score (0.0-1.0)
        checks = [is_brightness_ok, is_contrast_ok, is_blur_ok, is_motion_ok]
        quality_score = sum(checks) / len(checks)
        is_quality_ok = all(checks)

        return QualityReport(
            brightness=brightness,
            contrast=contrast,
            blur_variance=blur_variance,
            motion_detected=motion_detected,
            motion_score=motion_score,
            is_brightness_ok=is_brightness_ok,
            is_contrast_ok=is_contrast_ok,
            is_blur_ok=is_blur_ok,
            is_motion_ok=is_motion_ok,
            is_quality_ok=is_quality_ok,
            warnings=warnings,
            quality_score=quality_score,
        )

    def reset(self) -> None:
        """Reset previous frame for motion detection."""
        self._previous_frame = None
        logger.debug("Quality assessor reset.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = QualityConfig()
    assessor = QualityAssessor(config)

    # Create test frames
    print("Testing QualityAssessor...")
    print("=" * 60)

    # Good quality frame
    good_frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    report = assessor.assess(good_frame)
    print(f"\nGood frame: quality_score={report.quality_score:.2f}")
    print(f"  brightness={report.brightness:.1f}, contrast={report.contrast:.1f}")
    print(f"  blur_variance={report.blur_variance:.1f}, quality_ok={report.is_quality_ok}")
    print(f"  warnings: {report.warnings}")

    # Dark frame
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    report = assessor.assess(dark_frame)
    print(f"\nDark frame: quality_score={report.quality_score:.2f}")
    print(f"  brightness={report.brightness:.1f}, warnings: {report.warnings}")

    # Blurry frame (low variance)
    blurry_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    report = assessor.assess(blurry_frame)
    print(f"\nBlurry frame: quality_score={report.quality_score:.2f}")
    print(f"  blur_variance={report.blur_variance:.1f}, warnings: {report.warnings}")
