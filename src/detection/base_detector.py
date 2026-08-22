"""Abstract detector interface for the FreshSense Phase 4 detection module.

This module defines the contract that every object detector must implement.
By adhering to this interface, any detector (YOLO, RT-DETR, Grounding DINO,
TensorRT, ONNX) can be plugged into the inference pipeline without changing
``pipeline.py``.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "DetectorConfig",
]


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge (inclusive).
        y1: Top edge (inclusive).
        x2: Right edge (exclusive).
        y2: Bottom edge (exclusive).
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def clamp(self, width: int, height: int) -> "BoundingBox":
        """Return a copy clipped to the given image dimensions."""
        return BoundingBox(
            x1=max(0, self.x1),
            y1=max(0, self.y1),
            x2=min(width, self.x2),
            y2=min(height, self.y2),
        )

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)


@dataclass
class Detection:
    """A single detected object.

    Attributes:
        label: Detected class name (e.g. "Apple").
        confidence: Detection confidence score (0.0-1.0).
        bbox: Bounding box in pixel coordinates.
        class_id: Numeric class identifier.
        tracking_id: Optional persistent tracker id.
        cropped_image: Optional cropped image of the object.
        timestamp: Detection timestamp (seconds since epoch).
    """

    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int = -1
    tracking_id: int = -1
    cropped_image: Optional[np.ndarray] = None
    timestamp: float = 0.0

    @property
    def class_name(self) -> str:
        """Alias for label to satisfy contract."""
        return self.label

    @property
    def x1(self) -> int:
        """Left edge."""
        return self.bbox.x1

    @property
    def y1(self) -> int:
        """Top edge."""
        return self.bbox.y1

    @property
    def x2(self) -> int:
        """Right edge."""
        return self.bbox.x2

    @property
    def y2(self) -> int:
        """Bottom edge."""
        return self.bbox.y2

    def to_dict(self) -> Dict:
        """Convert to dictionary (LangGraph/API-friendly)."""
        return {
            "class_id": self.class_id,
            "class_name": self.label,
            "class": self.label,
            "confidence": self.confidence,
            "x1": self.bbox.x1,
            "y1": self.bbox.y1,
            "x2": self.bbox.x2,
            "y2": self.bbox.y2,
            "bounding_box": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
            },
            "area": self.bbox.area,
            "center": list(self.bbox.center),
            "tracking_id": self.tracking_id,
            "timestamp": self.timestamp,
        }


@dataclass
class DetectionResult:
    """Collection of detections for a single frame.

    Attributes:
        detections: List of Detection objects.
        frame_width: Width of the source frame.
        frame_height: Height of the source frame.
        latency_ms: Detection latency in milliseconds.
        image_id: Identifier for the processed frame.
    """

    detections: List[Detection] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    latency_ms: float = 0.0
    image_id: str = ""

    @property
    def count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> Dict:
        """Convert to dictionary (LangGraph-friendly)."""
        return {
            "detections": [d.to_dict() for d in self.detections],
            "count": self.count,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "latency_ms": self.latency_ms,
            "image_id": self.image_id,
        }


@dataclass(frozen=True)
class DetectorConfig:
    """Base configuration shared by all detectors.

    Attributes:
        model_path: Path to the detector weights (default: baseline best.pt).
        confidence_threshold: Minimum detection confidence.
        iou_threshold: IoU threshold for non-max suppression.
        image_size: Inference image resolution (default: 640).
        device: Inference device ("cuda", "cpu", or "auto").
        max_detections: Maximum number of detections per frame.
        class_names: List of supported class names (optional override).
    """

    model_path: str = "models/detection/detector/weights/best.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "auto"
    max_detections: int = 20
    class_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0].")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0.0, 1.0].")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive.")


class BaseDetector(abc.ABC):
    """Abstract base class for all object detectors.

    Every detector implementation must provide ``load``, ``detect``,
    ``warmup``, and ``shutdown``. This enables swapping detection backends
    (YOLO, RT-DETR, Grounding DINO, TensorRT, ONNX) without modifying the
    inference pipeline.

    Args:
        config: DetectorConfig (or a subclass) with settings.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.is_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load detector weights and prepare the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a frame.

        Args:
            frame: BGR image from OpenCV (height, width, 3).

        Returns:
            DetectionResult with all detected objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def warmup(self) -> None:
        """Warm up the detector on a dummy frame."""
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release detector resources."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Common utilities
    # ------------------------------------------------------------------

    @property
    def device_str(self) -> str:
        """Resolve the configured device to a concrete device string."""
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def get_supported_classes(self) -> List[str]:
        """Return the supported class names.

        Returns:
            List of class names if configured, else an empty list.
        """
        return list(self.config.class_names) if self.config.class_names else []
