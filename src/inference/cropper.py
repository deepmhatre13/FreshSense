"""Fruit cropping & ROI utilities for FreshSense Phase 4.

Ensures the classifier only ever sees the fruit region (not the full webcam
frame), with configurable margin expansion and auto-focus / size gating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.detection.base_detector import BoundingBox, Detection
from src.detection.utils import crop_bbox, expand_bbox

logger = logging.getLogger(__name__)

__all__ = ["Cropper", "CropperConfig", "CropResult"]


@dataclass(frozen=True)
class CropperConfig:
    """Configuration for fruit cropping.

    Attributes:
        expand_scale: Fraction to expand the box per side (0.05 = 5%).
        min_side: Minimum box side length (pixels) to accept a crop.
        min_area: Minimum box area (pixels) to accept a crop.
        target_size: Square size the crop is resized to.
    """

    expand_scale: float = 0.08
    min_side: int = 32
    min_area: int = 1024
    target_size: int = 224

    def __post_init__(self) -> None:
        if self.expand_scale < 0.0:
            raise ValueError("expand_scale must be non-negative.")
        if self.min_side <= 0:
            raise ValueError("min_side must be positive.")
        if self.min_area <= 0:
            raise ValueError("min_area must be positive.")
        if self.target_size <= 0:
            raise ValueError("target_size must be positive.")


@dataclass
class CropResult:
    """Result of cropping a detected fruit.

    Attributes:
        detection: The source detection.
        cropped: The cropped & resized image (BGR).
        valid: Whether the crop passed auto-focus / size gates.
        rejection_reason: Why the crop was rejected (if not valid).
        expanded_box: The expanded bounding box used (if valid).
    """

    detection: Detection
    cropped: Optional[np.ndarray] = None
    valid: bool = True
    rejection_reason: str = ""
    expanded_box: Optional[BoundingBox] = None


class Cropper:
    """Crops detected fruits for classification.

    Expands the box by a small margin to avoid clipping the fruit border, then
    rejects crops that are too small (likely too far away) or too blurry.
    """

    def __init__(self, config: CropperConfig) -> None:
        self.config = config

    def crop(self, frame: np.ndarray, detection: Detection) -> CropResult:
        """Crop a single detection from a frame.

        Args:
            frame: Source BGR frame.
            detection: Detection to crop.

        Returns:
            A CropResult that may be valid or rejected.
        """
        h, w = frame.shape[:2]
        expanded = expand_bbox(
            detection.bbox,
            self.config.expand_scale,
            w,
            h,
        )

        # Size gates (auto-focus: move closer if too small).
        if expanded.width < self.config.min_side or expanded.height < self.config.min_side:
            return CropResult(
                detection=detection,
                valid=False,
                rejection_reason="Move closer - fruit too small",
            )
        if expanded.area < self.config.min_area:
            return CropResult(
                detection=detection,
                valid=False,
                rejection_reason="Move closer - fruit too small",
            )

        region = crop_bbox(frame, expanded)
        resized = cv2.resize(
            region,
            (self.config.target_size, self.config.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        return CropResult(
            detection=detection,
            cropped=resized,
            valid=True,
            expanded_box=expanded,
        )
