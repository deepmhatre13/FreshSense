"""Concrete detector implementations for FreshSense Phase 4.

The default detector is YOLOv11 (``YOLODetector``) via the Ultralytics library.
If Ultralytics is unavailable, a lightweight ``SimpleDetector`` (color/motion
based) and a deterministic ``MockDetector`` are provided so the pipeline remains
testable without a GPU or custom weights.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import cv2
import numpy as np

from src.detection.base_detector import (
    BaseDetector,
    BoundingBox,
    Detection,
    DetectionResult,
    DetectorConfig,
)
from src.detection.utils import non_max_suppression

logger = logging.getLogger(__name__)

__all__ = ["YOLODetector", "SimpleDetector", "MockDetector"]

from pathlib import Path

# Frozen 10-class taxonomy from dataset v2 baseline
FROZEN_10_CLASSES = {
    0: "Apple",
    1: "Grape",
    2: "Kiwi",
    3: "Mango",
    4: "Orange",
    5: "Strawberry",
    6: "banana",
    7: "cherry",
    8: "chickoo",
    9: "guava",
}

DEFAULT_YOLO_CLASSES = list(FROZEN_10_CLASSES.values())


class YOLODetector(BaseDetector):
    """YOLOv11 object detector wrapping Ultralytics.

    Loads the trained production baseline checkpoint
    (``models/detection/detector/weights/best.pt`` by default), reuses a
    single model instance, prefers GPU, and falls back to CPU.

    Note:
        Requires the ``ultralytics`` package. If it is not installed, calling
        :meth:`load` raises :class:`ImportError`.
    """

    def __init__(
        self,
        config: DetectorConfig,
        weight_name: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self.weight_name = weight_name or config.model_path or "models/detection/detector/weights/best.pt"
        self.model = None

    def load(self) -> None:
        """Load the YOLO model and move it to the selected device."""
        weight_path = Path(self.weight_name)
        # Verify checkpoint file exists if it refers to a local file path
        if ("/" in self.weight_name or "\\" in self.weight_name or self.weight_name.endswith(".pt")) and not weight_path.exists():
            # Standard Ultralytics pretrained model names auto-download if simple filename in cwd
            if not (weight_path.name == self.weight_name and (weight_path.exists() or self.weight_name in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])):
                raise FileNotFoundError(
                    f"YOLO model weights file not found: '{self.weight_name}'"
                )

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "Ultralytics is required for YOLODetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        device = self.device_str  # "cuda" or "cpu"
        self.model = YOLO(self.weight_name)
        self.model.to(device)
        self.is_loaded = True
        logger.info(
            "YOLODetector loaded: %s on %s", self.weight_name, device
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run YOLO inference on a frame."""
        if self.model is None:
            raise RuntimeError("YOLODetector not loaded. Call load() first.")

        # Input validation for frame robustness
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning("Invalid or unreadable input image frame received by YOLODetector.")
            return DetectionResult(
                detections=[],
                frame_width=frame.shape[1] if (isinstance(frame, np.ndarray) and frame.ndim >= 2) else 0,
                frame_height=frame.shape[0] if (isinstance(frame, np.ndarray) and frame.ndim >= 1) else 0,
                latency_ms=0.0,
            )

        start = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        h, w = frame.shape[:2]
        detections: List[Detection] = []

        if results:
            boxes = results[0].boxes
            names = getattr(results[0], "names", FROZEN_10_CLASSES) or FROZEN_10_CLASSES
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (names[cls_id] if cls_id < len(names) else str(cls_id))
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            label=label,
                            confidence=conf,
                            bbox=BoundingBox(
                                int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                            ),
                            class_id=cls_id,
                            timestamp=time.time(),
                        )
                    )

        detections = detections[: self.config.max_detections]
        return DetectionResult(
            detections=detections,
            frame_width=w,
            frame_height=h,
            latency_ms=latency_ms,
            image_id=f"det_{int(start * 1000)}",
        )

    def warmup(self) -> None:
        """Warm up the model on a dummy frame."""
        if self.model is None:
            return
        dummy = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.config.image_size, verbose=False)
        logger.info("YOLODetector warmup complete.")

    def shutdown(self) -> None:
        """Release the model."""
        self.model = None
        self.is_loaded = False
        logger.info("YOLODetector shutdown.")


class SimpleDetector(BaseDetector):
    """A lightweight, dependency-free detector for development/testing.

    It segments large saturated colour blobs (a crude stand-in for real object
    detection). It exists so the pipeline and tracker can be exercised without
    Ultralytics. It is NOT suitable for production.
    """

    def __init__(
        self,
        config: DetectorConfig,
        hsv_ranges: Optional[dict] = None,
    ) -> None:
        super().__init__(config)
        self.hsv_ranges = hsv_ranges or {
            "fruit": ((0, 100, 60), (179, 255, 255)),
        }

    def load(self) -> None:
        self.is_loaded = True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections: List[Detection] = []

        for label, (lower, upper) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 1500:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                detections.append(
                    Detection(
                        label=label,
                        confidence=0.7,
                        bbox=BoundingBox(x, y, x + bw, y + bh),
                        timestamp=time.time(),
                    )
                )

        detections = non_max_suppression(detections, self.config.iou_threshold)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(
            detections=detections,
            frame_width=w,
            frame_height=h,
            latency_ms=latency_ms,
        )

    def warmup(self) -> None:
        pass

    def shutdown(self) -> None:
        self.is_loaded = False


class MockDetector(BaseDetector):
    """Deterministic detector returning configurable detections.

    Used for unit tests and integration tests so that downstream logic (crop,
    tracking, classification) can be verified without real models.
    """

    def __init__(
        self,
        config: DetectorConfig,
        detections: Optional[List[Detection]] = None,
    ) -> None:
        super().__init__(config)
        self._detections = detections or []

    def load(self) -> None:
        self.is_loaded = True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        return DetectionResult(
            detections=list(self._detections),
            frame_width=w,
            frame_height=h,
            latency_ms=1.0,
        )

    def warmup(self) -> None:
        pass

    def shutdown(self) -> None:
        self.is_loaded = False
