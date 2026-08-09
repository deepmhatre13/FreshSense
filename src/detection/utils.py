"""Utility helpers for the FreshSense Phase 4 detection module.

Provides geometry helpers (IoU, non-max suppression), crop utilities, and
frame normalization used by the detectors and the inference pipeline.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from src.detection.base_detector import BoundingBox, Detection

logger = logging.getLogger(__name__)

__all__ = [
    "iou",
    "non_max_suppression",
    "center_distance",
    "crop_bbox",
    "expand_bbox",
    "letterbox",
    "filter_detections",
]


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute Intersection-over-Union of two bounding boxes.

    Args:
        a: First bounding box.
        b: Second bounding box.

    Returns:
        IoU in the range [0.0, 1.0]. 0.0 means no overlap.
    """
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = a.area + b.area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def center_distance(a: BoundingBox, b: BoundingBox) -> float:
    """Euclidean distance between two bounding-box centers.

    Args:
        a: First bounding box.
        b: Second bounding box.

    Returns:
        Euclidean distance in pixels.
    """
    ax, ay = a.center
    bx, by = b.center
    return float(np.hypot(ax - bx, ay - by))


def non_max_suppression(
    detections: Sequence[Detection],
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """Apply greedy non-max suppression.

    Keeps high-confidence detections and removes lower-confidence ones that
    overlap significantly.

    Args:
        detections: Detections to filter.
        iou_threshold: IoU threshold above which the lower-confidence box is
            suppressed.

    Returns:
        Filtered list of detections sorted by confidence descending.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: List[Detection] = []

    while ordered:
        best = ordered.pop(0)
        keep.append(best)
        # Keep boxes that don't overlap too much with the best.
        ordered = [d for d in ordered if iou(best.bbox, d.bbox) <= iou_threshold]

    return keep


def crop_bbox(frame: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Extract the region of a frame inside a bounding box.

    Args:
        frame: BGR image.
        box: Bounding box (clamped to frame).

    Returns:
        Cropped image region.
    """
    h, w = frame.shape[:2]
    box = box.clamp(w, h)
    if box.area <= 0:
        raise ValueError("Cannot crop an empty bounding box.")
    return frame[box.y1 : box.y2, box.x1 : box.x2].copy()


def expand_bbox(
    box: BoundingBox,
    scale: float,
    width: int,
    height: int,
) -> BoundingBox:
    """Expand a bounding box by a scale factor and clamp to frame.

    ``scale`` (e.g. 0.08 for 8%) grows the box by that fraction on each side to
    avoid clipping the object border.

    Args:
        box: Original bounding box.
        scale: Expansion fraction per side (0.0-1.0).
        width: Frame width for clamping.
        height: Frame height for clamping.

    Returns:
        Expanded bounding box clamped to frame dimensions.
    """
    pad_x = int(box.width * scale)
    pad_y = int(box.height * scale)
    expanded = BoundingBox(
        x1=box.x1 - pad_x,
        y1=box.y1 - pad_y,
        x2=box.x2 + pad_x,
        y2=box.y2 + pad_y,
    )
    return expanded.clamp(width, height)


def letterbox(
    frame: np.ndarray,
    size: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float]:
    """Resize a frame to fit within ``size`` preserving aspect ratio.

    Args:
        frame: BGR image.
        size: Target (width, height).
        color: Fill color for letterbox bands.

    Returns:
        Tuple of (letterboxed frame, scale factor).
    """
    target_w, target_h = size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = target_w - new_w, target_h - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale


def filter_detections(
    detections: Sequence[Detection],
    min_confidence: float = 0.0,
    min_area: int = 0,
    min_side: int = 0,
) -> List[Detection]:
    """Filter detections by confidence and size constraints.

    Args:
        detections: Detections to filter.
        min_confidence: Minimum confidence to keep.
        min_area: Minimum box area in pixels to keep.
        min_side: Minimum box side length in pixels to keep.

    Returns:
        Filtered list of detections.
    """
    result: List[Detection] = []
    for d in detections:
        if d.confidence < min_confidence:
            continue
        if d.bbox.area < min_area:
            continue
        if min(d.bbox.width, d.bbox.height) < min_side:
            continue
        result.append(d)
    return result
