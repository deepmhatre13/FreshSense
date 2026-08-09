"""FreshSense Phase 4 - Object detection module.

Structure:
    - base_detector.py : Abstract detector interface + data types
    - detector.py      : Concrete detectors (YOLO / simple / mock)
    - factory.py       : Detector factory (strategy/plugin registry)
    - utils.py         : IoU, NMS, cropping geometry
    - visualizer.py    : Box/label drawing helpers
"""

from src.detection.base_detector import (
    BaseDetector,
    BoundingBox,
    Detection,
    DetectionResult,
    DetectorConfig,
)
from src.detection.detector import MockDetector, SimpleDetector, YOLODetector
from src.detection.factory import DetectorFactory
from src.detection.utils import (
    center_distance,
    crop_bbox,
    expand_bbox,
    filter_detections,
    iou,
    letterbox,
    non_max_suppression,
)
from src.detection.visualizer import draw_detections, draw_box

# Fruit/vegetable classes SmartFreshAI is built to recognise.
SUPPORTED_CLASSES = [
    "apple",
    "banana",
    "orange",
    "mango",
    "tomato",
    "potato",
]

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "DetectorConfig",
    "DetectorFactory",
    "MockDetector",
    "SimpleDetector",
    "YOLODetector",
    "SUPPORTED_CLASSES",
    "center_distance",
    "crop_bbox",
    "draw_box",
    "draw_detections",
    "expand_bbox",
    "filter_detections",
    "iou",
    "letterbox",
    "non_max_suppression",
]
