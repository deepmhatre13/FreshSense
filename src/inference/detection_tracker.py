"""Multi-object fruit tracker for FreshSense Phase 4.

A lightweight IoU / centre-distance based tracker inspired by ByteTrack. It
assigns persistent tracking IDs to detected fruits across frames so that each
fruit keeps a stable identity (and its own classifier stabilizer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.detection.base_detector import BoundingBox, Detection
from src.detection.utils import center_distance, iou

logger = logging.getLogger(__name__)

__all__ = ["DetectionTracker", "TrackerConfig", "TrackedObject"]


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for the multi-object tracker.

    Attributes:
        iou_threshold: IoU threshold to match a new detection to a track.
        max_center_distance: Max centre distance to match (pixels).
        max_lost_frames: Frames a track may be lost before it is removed.
        min_hits: Frames a track must survive before it emits a tracking id.
    """

    iou_threshold: float = 0.3
    max_center_distance: float = 120.0
    max_lost_frames: int = 15
    min_hits: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0.0, 1.0].")
        if self.max_center_distance < 0:
            raise ValueError("max_center_distance must be non-negative.")
        if self.max_lost_frames < 0:
            raise ValueError("max_lost_frames must be non-negative.")
        if self.min_hits < 0:
            raise ValueError("min_hits must be non-negative.")


@dataclass
class TrackedObject:
    """Stateful representation of a tracked fruit.

    Attributes:
        tracking_id: Persistent identifier.
        label: Current class label.
        bbox: Latest bounding box.
        confidence: Latest confidence.
        lost_frames: Consecutive frames with no detection.
        hits: Number of times the track has been matched.
        active: Whether the track is currently considered present.
    """

    tracking_id: int
    label: str
    bbox: BoundingBox
    confidence: float = 0.0
    lost_frames: int = 0
    hits: int = 1
    active: bool = True


class DetectionTracker:
    """Assigns persistent IDs to detections across frames.

    Uses a greedy Hungarian-like matching between existing tracks and new
    detections based on IoU (falling back to centre distance) so boxes don't
    jump between fruits.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id: int = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Associate new detections with existing tracks.

        Args:
            detections: Detections from the current frame (without IDs).

        Returns:
            The same detections annotated with tracking IDs. New tracks are
            created for unmatched detections; unmatched tracks are marked lost.
        """
        # 1. Age existing tracks.
        for track in self._tracks.values():
            track.lost_frames += 1
            if track.lost_frames > self.config.max_lost_frames:
                track.active = False

        matched = set()
        out: List[Detection] = []

        # 2. Greedy match each new detection to the best existing track.
        unmatched_detections = list(detections)
        active_tracks = [t for t in self._tracks.values() if t.active]

        # Sort detections by confidence to match confident items first.
        unmatched_detections.sort(key=lambda d: d.confidence, reverse=True)

        for det in unmatched_detections:
            best_track = None
            best_score = -1.0
            for track in active_tracks:
                if track.tracking_id in matched:
                    continue
                iou_score = iou(track.bbox, det.bbox)
                distance_score = 1.0 - (
                    center_distance(track.bbox, det.bbox)
                    / max(1.0, self.config.max_center_distance)
                )
                # Use IoU primarily; fall back to distance for small boxes.
                score = max(iou_score, distance_score)
                if (
                    iou_score >= self.config.iou_threshold
                    or distance_score >= 0.5
                ):
                    if score > best_score:
                        best_score = score
                        best_track = track

            if best_track is not None:
                matched.add(best_track.tracking_id)
                best_track.bbox = det.bbox
                best_track.label = det.label
                best_track.confidence = det.confidence
                best_track.lost_frames = 0
                best_track.hits += 1
                best_track.active = True
                det.tracking_id = best_track.tracking_id
                out.append(det)
            else:
                # Create a new track.
                track = TrackedObject(
                    tracking_id=self._next_id,
                    label=det.label,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
                self._tracks[self._next_id] = track
                det.tracking_id = self._next_id
                self._next_id += 1
                matched.add(track.tracking_id)
                out.append(det)

        return out

    def get_active_tracks(self) -> List[TrackedObject]:
        """Return currently-active tracks."""
        return [t for t in self._tracks.values() if t.active]

    def get_track(self, tracking_id: int) -> Optional[TrackedObject]:
        """Return a track by id."""
        return self._tracks.get(tracking_id)

    def reset(self) -> None:
        """Clear all tracks and reset id counter."""
        self._tracks.clear()
        self._next_id = 0
