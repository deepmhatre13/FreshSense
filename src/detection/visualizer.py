"""Visualization helpers for the FreshSense Phase 4 detection module.

Draws bounding boxes, labels, and tracking IDs onto frames for debugging and
preview output.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

from src.detection.base_detector import BoundingBox, Detection, DetectionResult

logger = logging.getLogger(__name__)

__all__ = ["draw_detections", "draw_box", "color_for_label"]

# A small, stable colour palette (BGR).
_PALETTE: Sequence[Tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (0, 0, 255),  # red
    (255, 128, 0),  # orange
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (128, 0, 255),  # purple
    (0, 255, 255),  # yellow
    (128, 128, 255),  # light purple
]


def color_for_label(label: str) -> Tuple[int, int, int]:
    """Return a stable colour for a label based on its hash."""
    idx = abs(hash(label)) % len(_PALETTE)
    return _PALETTE[idx]


def draw_box(
    frame: np.ndarray,
    box: BoundingBox,
    color: Tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
) -> None:
    """Draw a single bounding box (and optional label) in place."""
    cv2.rectangle(
        frame,
        (box.x1, box.y1),
        (box.x2, box.y2),
        color,
        thickness,
    )
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness_t = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness_t)
        cv2.rectangle(
            frame,
            (box.x1, box.y1 - th - 6),
            (box.x1 + tw + 6, box.y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (box.x1 + 3, box.y1 - 4),
            font,
            scale,
            (255, 255, 255),
            thickness_t,
            cv2.LINE_AA,
        )


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    show_tracking_id: bool = True,
) -> np.ndarray:
    """Draw all detections onto a copy of the frame.

    Args:
        frame: Original BGR frame.
        detections: Detections to draw.
        show_tracking_id: If True, prepend the tracking id to the label.

    Returns:
        A copy of the frame with boxes drawn.
    """
    out = frame.copy()
    for det in detections:
        color = color_for_label(det.label)
        label = det.label
        if show_tracking_id and det.tracking_id >= 0:
            label = f"#{det.tracking_id} {det.label}"
        draw_box(out, det.bbox, color, label)
    return out
