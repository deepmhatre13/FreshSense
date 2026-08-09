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

# COCO subset of fruit/vegetable classes relevant to FreshSense.
COCO_FRUIT_IDS = {
    "apple": 47,
    "banana": 52,
    "orange": 49,
    "broccoli": 46,
    "carrot": 51,
    "hot dog": 53,  # placeholder, unused
    "pizza": 55,  # placeholder, unused
}

DEFAULT_YOLO_CLASSES = [
    "apple",
    "banana",
    "orange",
    "broccoli",
    "carrot",
]


def _auto_download_yolo(weight_name: str) -> str:
    """Best-effort helper that returns the path Ultralytics will use.

    Ultralytics downloads known weights (e.g. ``yolo11n.pt``) on first use when
    given the bare name. This helper only logs the behaviour for clarity.

    Args:
        weight_name: Model weight name (e.g. "yolo11n.pt").

    Returns:
        The weight name (Ultralytics resolves autodownload internally).
    """
    logger.info("Ultralytics will auto-download weights if missing: %s", weight_name)
    return weight_name


class YOLODetector(BaseDetector):
    """YOLOv11 object detector wrapping Ultralytics.

    Loads ``yolo11n.pt`` (auto-downloaded by Ultralytics if missing), reuses a
    single model instance, prefers GPU, and falls back to CPU.

    Note:
        Requires the ``ultralytics`` package. If it is not installed, calling
        :meth:`load` raises :class:`ImportError`.
    """

    def __init__(
        self,
        config: DetectorConfig,
        weight_name: str = "yolo11n.pt",
    ) -> None:
        super().__init__(config)
        self.weight_name = _auto_download_yolo(weight_name)
        self.model = None

    def load(self) -> None:
        """Load the YOLO model and move it to the selected device."""
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

        start = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        h, w = frame.shape[:2]
        detections: List[Detection] = []

        if results:
            boxes = results[0].boxes
            names = results[0].names
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    label = names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            label=label,
                            confidence=conf,
                            bbox=BoundingBox(
                                int(x1), int(y1), int(x2), int(y2)
                            ),
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
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
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
