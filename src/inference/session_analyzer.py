"""Session-level analysis for FreshSense Phase 4.

Semantic clarification of key metrics
--------------------------------------
total_frames         : total frames processed (including empty)
frames_with_detections: frames that had >= 1 fruit detection
total_detections     : sum of fruit detections across all frames
                       (same fruit in 90 frames → 90 detections)
unique_tracks        : number of distinct tracking IDs seen
                       (same fruit tracked across 90 frames → 1 track)
fruits_detected      : alias for unique_tracks (unique physical fruits)

This distinction matters for reporting: one banana visible for 90 frames
must NOT appear as "90 fruits detected."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SessionAnalyzer", "SessionSummary"]


@dataclass
class SessionSummary:
    """Summary statistics for a camera session.

    Attributes
    ----------
    session_id          : opaque session identifier
    duration_seconds    : wall-clock duration (max_ts - min_ts)
    total_frames        : frames submitted to add_frame()
    fruits_detected     : unique physical fruits (= unique_tracks)
    unique_tracks       : distinct tracking IDs encountered
    frames_with_detections : frames that contained at least one fruit
    total_detections    : raw count of detections across all frames
    fresh_count         : total fresh detections across all frames
    stale_count         : total stale detections across all frames
    rotten_count        : total rotten detections across all frames
    uncertain_count     : total uncertain detections across all frames
    avg_confidence      : mean classifier confidence across all detections
    min_confidence      : minimum confidence observed
    max_confidence      : maximum confidence observed
    avg_image_quality   : mean image-quality score (0-1) if available
    unstable_detections : frames with >1 fruit and mixed freshness classes
    class_switches      : total label transitions per track (aggregated)
    avg_track_duration  : average seconds a track was visible
    lowest_confidence_fruit : info dict for the detection with lowest conf
    highest_risk_fruit  : info dict for the detection with highest risk
    fruit_details       : raw list of per-detection dicts
    """

    session_id: str
    duration_seconds: float
    total_frames: int
    fruits_detected: int          # = unique_tracks (unique physical fruits)
    unique_tracks: int
    frames_with_detections: int = 0
    total_detections: int = 0
    fresh_count: int = 0
    stale_count: int = 0
    rotten_count: int = 0
    uncertain_count: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    avg_image_quality: float = 0.0
    unstable_detections: int = 0
    class_switches: int = 0
    avg_track_duration: float = 0.0
    lowest_confidence_fruit: Optional[Dict[str, Any]] = None
    highest_risk_fruit: Optional[Dict[str, Any]] = None
    fruit_details: List[Dict[str, Any]] = field(default_factory=list)


class SessionAnalyzer:
    """Analyzes a complete camera session.

    Usage
    -----
    analyzer = SessionAnalyzer()
    for frame_data in frames:
        analyzer.add_frame(frame_data)
    summary = analyzer.analyze(session_id="abc")
    report  = analyzer.to_text_report(summary)

    Frame data schema
    -----------------
    {
        "timestamp":    float,                        # seconds
        "predictions":  [{"fruit_name": str, "freshness_class": str}, ...],
        "confidences":  [float, ...],
        "tracking_ids": [int | None, ...],
        "quality_metrics": {"blur_score": float},    # optional
        "frame_number": int,                          # optional
    }
    """

    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []
        # tracks maps tracking_id -> list of {"timestamp": float, "prediction": dict}
        self.tracks: Dict[int, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(self, frame_data: Dict[str, Any]) -> None:
        """Add one frame's inference results to the session."""
        self.frames.append(frame_data)

        tracking_ids: List[Optional[int]] = frame_data.get("tracking_ids", [])
        predictions: List[Dict[str, Any]] = frame_data.get("predictions", [])

        for track_id, pred in zip(tracking_ids, predictions):
            if track_id is not None:
                self.tracks.setdefault(track_id, []).append(
                    {
                        "timestamp": frame_data.get("timestamp", 0.0),
                        "prediction": pred,
                    }
                )

    def analyze(self, session_id: str = "unknown") -> SessionSummary:
        """Analyze collected frames and produce a summary.

        Key guarantee: one fruit tracked across N frames contributes
        exactly 1 to `unique_tracks` / `fruits_detected`, regardless of N.
        """
        if not self.frames:
            return SessionSummary(
                session_id=session_id,
                duration_seconds=0.0,
                total_frames=0,
                fruits_detected=0,
                unique_tracks=0,
            )

        total_frames = len(self.frames)
        timestamps = [f.get("timestamp", 0.0) for f in self.frames]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0

        # ── Per-detection tallies ──────────────────────────────────────
        all_confidences: List[float] = []
        all_qualities: List[float] = []
        class_counts: Dict[str, int] = {
            "fresh": 0, "stale": 0, "rotten": 0, "uncertain": 0
        }
        fruit_details: List[Dict[str, Any]] = []
        total_detections = 0
        frames_with_detections = 0
        unstable_count = 0

        for frame in self.frames:
            predictions: List[Dict[str, Any]] = frame.get("predictions", [])
            confidences: List[float] = frame.get("confidences", [])
            quality_metrics: Dict[str, Any] = frame.get("quality_metrics", {})

            if predictions:
                frames_with_detections += 1

            for i, pred in enumerate(predictions):
                conf = confidences[i] if i < len(confidences) else 0.0
                total_detections += 1
                all_confidences.append(conf)

                freshness = pred.get("freshness_class", "uncertain")
                class_counts[freshness] = class_counts.get(freshness, 0) + 1

                fruit_details.append(
                    {
                        "fruit_type": pred.get("fruit_name", "unknown"),
                        "freshness": freshness,
                        "confidence": conf,
                        "frame": frame.get("frame_number", 0),
                    }
                )

            if quality_metrics:
                blur = quality_metrics.get("blur_score", None)
                if blur is not None:
                    # Normalise: good blur score ~ 200, map to [0,1]
                    all_qualities.append(min(1.0, float(blur) / 200.0))

            # Unstable frame: multiple fruits present with mixed classes
            if len(predictions) > 1:
                classes = [p.get("freshness_class") for p in predictions]
                if len(set(classes)) > 1:
                    unstable_count += 1

        # ── Per-track tallies ─────────────────────────────────────────
        unique_tracks = len(self.tracks)   # ← unique physical fruits
        class_switch_count = 0
        track_durations: List[float] = []

        for track_frames in self.tracks.values():
            if len(track_frames) > 1:
                classes = [f["prediction"].get("freshness_class") for f in track_frames]
                switches = sum(
                    1 for i in range(1, len(classes)) if classes[i] != classes[i - 1]
                )
                class_switch_count += switches

                ts = [f["timestamp"] for f in track_frames]
                track_durations.append(max(ts) - min(ts))

        # ── Aggregate stats ───────────────────────────────────────────
        avg_conf = float(np.mean(all_confidences)) if all_confidences else 0.0
        min_conf = float(min(all_confidences)) if all_confidences else 0.0
        max_conf = float(max(all_confidences)) if all_confidences else 0.0
        avg_quality = float(np.mean(all_qualities)) if all_qualities else 0.0
        avg_track_dur = float(np.mean(track_durations)) if track_durations else 0.0

        # ── Special fruits ───────────────────────────────────────────
        lowest_conf_fruit: Optional[Dict[str, Any]] = None
        highest_risk_fruit: Optional[Dict[str, Any]] = None

        if fruit_details:
            lowest_conf_fruit = min(fruit_details, key=lambda x: x["confidence"])

            def _risk_score(f: Dict[str, Any]) -> float:
                base = 1.0 - f["confidence"]
                mult = (
                    2.0 if f["freshness"] == "rotten"
                    else 1.0 if f["freshness"] == "stale"
                    else 0.0
                )
                return base * mult

            highest_risk_fruit = max(fruit_details, key=_risk_score)

        return SessionSummary(
            session_id=session_id,
            duration_seconds=duration,
            total_frames=total_frames,
            # fruits_detected == unique physical fruits (NOT total_detections)
            fruits_detected=unique_tracks,
            unique_tracks=unique_tracks,
            frames_with_detections=frames_with_detections,
            total_detections=total_detections,
            fresh_count=class_counts.get("fresh", 0),
            stale_count=class_counts.get("stale", 0),
            rotten_count=class_counts.get("rotten", 0),
            uncertain_count=class_counts.get("uncertain", 0),
            avg_confidence=avg_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            avg_image_quality=avg_quality,
            unstable_detections=unstable_count,
            class_switches=class_switch_count,
            avg_track_duration=avg_track_dur,
            lowest_confidence_fruit=lowest_conf_fruit,
            highest_risk_fruit=highest_risk_fruit,
            fruit_details=fruit_details,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_text_report(self, summary: SessionSummary) -> str:
        """Generate a human-readable text report."""
        lines = [
            "=" * 60,
            "SESSION ANALYSIS REPORT",
            "=" * 60,
            f"Session ID:    {summary.session_id}",
            f"Duration:      {summary.duration_seconds:.1f} seconds",
            f"Total frames:  {summary.total_frames}",
            f"Frames w/ detections: {summary.frames_with_detections}",
            "",
            "FRUIT DETECTION",
            f"  Unique fruits (tracks): {summary.unique_tracks}",
            f"  Total detections:       {summary.total_detections}",
            f"  Fresh:     {summary.fresh_count}",
            f"  Stale:     {summary.stale_count}",
            f"  Rotten:    {summary.rotten_count}",
            f"  Uncertain: {summary.uncertain_count}",
            "",
            "CONFIDENCE",
            f"  Average: {summary.avg_confidence:.3f}",
            f"  Min:     {summary.min_confidence:.3f}",
            f"  Max:     {summary.max_confidence:.3f}",
            "",
            "STABILITY",
            f"  Unstable frames:    {summary.unstable_detections}",
            f"  Class switches:     {summary.class_switches}",
            f"  Avg track duration: {summary.avg_track_duration:.2f}s",
            "",
            "QUALITY",
            f"  Avg image quality: {summary.avg_image_quality:.3f}",
            "",
        ]

        if summary.lowest_confidence_fruit:
            lc = summary.lowest_confidence_fruit
            lines += [
                "LOWEST CONFIDENCE FRUIT",
                f"  Type:       {lc.get('fruit_type')}",
                f"  Freshness:  {lc.get('freshness')}",
                f"  Confidence: {lc.get('confidence', 0):.3f}",
                "",
            ]

        if summary.highest_risk_fruit:
            hr = summary.highest_risk_fruit
            lines += [
                "HIGHEST RISK FRUIT",
                f"  Type:       {hr.get('fruit_type')}",
                f"  Freshness:  {hr.get('freshness')}",
                f"  Confidence: {hr.get('confidence', 0):.3f}",
                "",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all session data."""
        self.frames.clear()
        self.tracks.clear()
