"""Image quality validation for the FreshSense dataset pipeline.

The :class:`ImageQualityChecker` inspects an image for:

- Minimum resolution (too-small images are useless for a CNN).
- Blurriness (Laplacian variance).
- Underexposure / overexposure (mean grayscale brightness).
- Corrupted / undecodable files.

It is designed to be *non-fatal*: :meth:`validate` never raises. Corrupted or
unreadable images are simply reported as invalid so the dataset pipeline can
skip them without crashing a DataLoader worker.

Performance notes:

- The grayscale image is computed **once** and shared by the blur and
  brightness metrics (avoids a duplicate ``cvtColor`` per image).
- The checker is immutable after construction, which makes it safe to share
  across DataLoader worker processes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ImageQualityChecker", "QualityReport"]


@dataclass(frozen=True)
class QualityReport:
    """Structured result of an image quality check."""

    path: Path
    readable: bool
    resolution_ok: bool
    blur_score: float
    is_blurry: bool
    brightness: float
    is_dark: bool
    is_overexposed: bool

    @property
    def is_valid(self) -> bool:
        """True if the image is usable for training."""
        return (
            self.readable
            and self.resolution_ok
            and not self.is_blurry
            and not self.is_dark
            and not self.is_overexposed
        )

    @property
    def rejection_reasons(self) -> list[str]:
        """Human-readable list of why this image was rejected (empty if valid)."""
        reasons: list[str] = []
        if not self.readable:
            reasons.append("unreadable")
        if not self.resolution_ok:
            reasons.append("low-resolution")
        if self.is_blurry:
            reasons.append(f"blurry (variance={self.blur_score:.1f})")
        if self.is_dark:
            reasons.append(f"underexposed (brightness={self.brightness:.1f})")
        if self.is_overexposed:
            reasons.append(f"overexposed (brightness={self.brightness:.1f})")
        return reasons


class ImageQualityChecker:
    """Validates image quality for the FreshSense dataset.

    Args:
        min_width: Minimum acceptable image width in pixels.
        min_height: Minimum acceptable image height in pixels.
        blur_threshold: Laplacian variance below this is considered blurry.
        dark_threshold: Mean grayscale below this is considered underexposed.
        bright_threshold: Mean grayscale above this is considered overexposed.

    Raises:
        ValueError: If any threshold combination is invalid.
    """

    def __init__(
        self,
        min_width: int = 224,
        min_height: int = 224,
        blur_threshold: float = 100.0,
        dark_threshold: int = 40,
        bright_threshold: int = 220,
    ) -> None:
        if min_width <= 0 or min_height <= 0:
            raise ValueError("min_width and min_height must be positive.")
        if blur_threshold < 0:
            raise ValueError("blur_threshold cannot be negative.")
        if not 0 <= dark_threshold < bright_threshold <= 255:
            raise ValueError(
                "Thresholds must satisfy 0 <= dark < bright <= 255."
            )

        self.min_width = min_width
        self.min_height = min_height
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load an image as BGR using OpenCV.

        Args:
            image_path: Path to the image file.

        Returns:
            The image as a BGR numpy array.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file exists but cannot be decoded as an image.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # IMREAD_COLOR guarantees a 3-channel BGR array (or None on failure).
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        return image

    def check_resolution(self, image: np.ndarray) -> bool:
        """Return True if the image meets the minimum resolution."""
        height, width = image.shape[:2]
        return width >= self.min_width and height >= self.min_height

    def blur_score(self, image: np.ndarray) -> float:
        """Return the Laplacian variance (higher = sharper).

        Args:
            image: A BGR image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def brightness_score(self, image: np.ndarray) -> float:
        """Return the mean grayscale brightness in [0, 255].

        Args:
            image: A BGR image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _compute_metrics(self, image: np.ndarray) -> tuple[bool, float, float]:
        """Compute resolution / blur / brightness sharing one grayscale pass.

        Args:
            image: A BGR image.

        Returns:
            ``(resolution_ok, blur_score, brightness)``
        """
        resolution_ok = self.check_resolution(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        return resolution_ok, blur, brightness

    def validate(self, image_path: Union[str, Path]) -> QualityReport:
        """Validate an image without raising.

        This method is safe to call from a DataLoader worker. Corrupted or
        unreadable images are reported as invalid rather than raising.

        Args:
            image_path: Path to the image file.

        Returns:
            A QualityReport describing the image's quality.
        """
        image_path = Path(image_path)

        # Try to load; on any failure, report as unreadable/invalid.
        try:
            image = self.load_image(image_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping invalid image %s: %s", image_path, exc)
            return QualityReport(
                path=image_path,
                readable=False,
                resolution_ok=False,
                blur_score=0.0,
                is_blurry=True,
                brightness=0.0,
                is_dark=True,
                is_overexposed=False,
            )

        # Compute each metric sharing one grayscale conversion.
        resolution_ok, blur, brightness = self._compute_metrics(image)

        report = QualityReport(
            path=image_path,
            readable=True,
            resolution_ok=resolution_ok,
            blur_score=blur,
            is_blurry=blur < self.blur_threshold,
            brightness=brightness,
            is_dark=brightness < self.dark_threshold,
            is_overexposed=brightness > self.bright_threshold,
        )

        if not report.is_valid:
            logger.debug(
                "Skipping %s: %s",
                image_path,
                ", ".join(report.rejection_reasons),
            )
        return report

    def is_valid(self, image_path: Union[str, Path]) -> bool:
        """Return True if the image passes all quality checks.

        Args:
            image_path: Path to the image file.
        """
        return self.validate(image_path).is_valid


if __name__ == "__main__":
    # Quick self-test.
    logging.basicConfig(level=logging.INFO)
    checker = ImageQualityChecker()
    report = checker.validate("sample.jpg")
    print("=" * 40)
    print(f"Path          : {report.path}")
    print(f"Readable      : {report.readable}")
    print(f"Resolution OK : {report.resolution_ok}")
    print(f"Blur Score    : {report.blur_score:.2f}")
    print(f"Is Blurry     : {report.is_blurry}")
    print(f"Brightness    : {report.brightness:.2f}")
    print(f"Is Dark       : {report.is_dark}")
    print(f"Overexposed   : {report.is_overexposed}")
    print(f"Is Valid      : {report.is_valid}")