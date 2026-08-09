"""Real-world data collection for FreshSense Phase 4.

Captures frames from webcam with manual labeling and comprehensive metadata.

Architecture
------------
Camera / image array
    ↓
_compute_quality()      ← blur, brightness, contrast
    ↓
_check_quality()        ← gate: accept or reject
    ↓
save image              ← accepted/ or rejected/ directory
    ↓
write metadata JSON     ← metadata/ directory
    ↓
update CollectionStats
    ↓
return CollectedSample
"""

from __future__ import annotations

import csv
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CollectionConfig",
    "QualityMetrics",
    "CollectedSample",
    "CollectionStats",
    "RealWorldCollector",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CollectionConfig:
    """Configuration for real-world data collection.

    Attributes
    ----------
    output_dir      : root directory for collected data
    capture_dir     : subdirectory for raw captures (unused; images go to
                      accepted/ or rejected/ directly)
    accepted_dir    : subdirectory for quality-passing images
    rejected_dir    : subdirectory for quality-failing images
    metadata_dir    : subdirectory for per-sample JSON metadata
    image_size      : (width, height) to resize saved images
    quality_checks  : if False, all images are accepted regardless of metrics
    blur_threshold  : minimum Laplacian variance (higher = sharper required)
    brightness_min  : minimum mean pixel brightness [0–255]
    brightness_max  : maximum mean pixel brightness [0–255]
    contrast_min    : minimum pixel std-deviation
    auto_save       : reserved (not used in current interactive loop)
    auto_save_interval : reserved
    """

    output_dir: Path = field(default_factory=lambda: Path("data/real_world"))
    capture_dir: str = "raw"
    accepted_dir: str = "accepted"
    rejected_dir: str = "rejected"
    metadata_dir: str = "metadata"
    image_size: Tuple[int, int] = (224, 224)
    quality_checks: bool = True
    blur_threshold: float = 100.0
    brightness_min: int = 40
    brightness_max: int = 220
    contrast_min: float = 20.0
    auto_save: bool = False
    auto_save_interval: int = 30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QualityMetrics:
    """Image quality metrics for a captured frame."""

    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    aspect_ratio: float = 0.0


@dataclass
class CollectedSample:
    """A single collected real-world sample with full metadata."""

    sample_id: str
    session_id: str
    timestamp: float
    image_path: str
    label: str
    accepted: bool = True
    rejection_reason: Optional[str] = None
    predicted_class: Optional[str] = None
    predicted_confidence: Optional[float] = None
    detector_confidence: Optional[float] = None
    tracking_id: Optional[int] = None
    quality: Optional[QualityMetrics] = None
    resolution: Tuple[int, int] = (0, 0)
    source_information: Dict[str, Any] = field(default_factory=dict)
    metadata_path: str = ""


@dataclass
class CollectionStats:
    """Statistics for a collection session."""

    total_captured: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    per_class_counts: Dict[str, int] = field(default_factory=dict)
    session_duration_seconds: float = 0.0
    avg_blur_score: float = 0.0
    avg_brightness: float = 0.0
    avg_contrast: float = 0.0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class RealWorldCollector:
    """Collects real-world fruit images from webcam with metadata.

    Usage
    -----
    collector = RealWorldCollector(config)
    collector.start_session()
    sample = collector.capture(frame, label="fresh")
    stats  = collector.end_session()
    """

    def __init__(self, config: Optional[CollectionConfig] = None) -> None:
        self.config = config or CollectionConfig()
        self.session_id: str = str(uuid.uuid4())[:8]
        self.session_start_time: float = 0.0
        self.stats = CollectionStats()
        self.samples: List[CollectedSample] = []
        self.frame_count: int = 0
        self.last_sample: Optional[CollectedSample] = None
        self._setup_directories()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_directories(self) -> None:
        base = self.config.output_dir
        self.raw_dir = base / self.config.capture_dir
        self.accepted_dir_path = base / self.config.accepted_dir
        self.rejected_dir_path = base / self.config.rejected_dir
        self.metadata_dir_path = base / self.config.metadata_dir
        for directory in (
            self.raw_dir,
            self.accepted_dir_path,
            self.rejected_dir_path,
            self.metadata_dir_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a new collection session."""
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start_time = time.time()
        self.stats = CollectionStats()
        self.samples = []
        self.frame_count = 0
        logger.info("Started collection session: %s", self.session_id)

    def end_session(self) -> CollectionStats:
        """End the session and persist metadata."""
        self.stats.session_duration_seconds = time.time() - self.session_start_time
        self._save_session_metadata()
        return self.stats

    # ------------------------------------------------------------------
    # Core capture
    # ------------------------------------------------------------------

    def capture(
        self,
        frame: np.ndarray,
        label: str,
        predicted_class: Optional[str] = None,
        predicted_confidence: Optional[float] = None,
        detector_confidence: Optional[float] = None,
        tracking_id: Optional[int] = None,
        source_info: Optional[Dict[str, Any]] = None,
    ) -> CollectedSample:
        """Capture and process a single frame.

        Args
        ----
        frame               : BGR numpy array from OpenCV
        label               : ground-truth freshness label
        predicted_class     : classifier output (if available)
        predicted_confidence: classifier confidence (if available)
        detector_confidence : detector confidence (if available)
        tracking_id         : tracking ID from DetectionTracker (if available)
        source_info         : arbitrary key-value metadata

        Returns
        -------
        CollectedSample with image saved to disk and metadata written.
        """
        sample_id = f"{self.session_id}_{self.frame_count:06d}"
        timestamp = time.time()
        self.frame_count += 1

        # Quality assessment
        quality = self._compute_quality(frame)
        accepted, rejection_reason = self._check_quality(quality)

        # Choose destination directory
        stem = f"{sample_id}_{label}"
        save_dir = self.accepted_dir_path if accepted else self.rejected_dir_path
        save_path = save_dir / f"{stem}.jpg"

        # Resize and write image
        resized = cv2.resize(frame, self.config.image_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(save_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Build sample record
        sample = CollectedSample(
            sample_id=sample_id,
            session_id=self.session_id,
            timestamp=timestamp,
            image_path=str(save_path),
            label=label,
            accepted=accepted,
            rejection_reason=rejection_reason,
            predicted_class=predicted_class,
            predicted_confidence=predicted_confidence,
            detector_confidence=detector_confidence,
            tracking_id=tracking_id,
            quality=quality,
            resolution=quality.resolution,
            source_information=source_info or {},
        )

        # Write per-sample metadata
        metadata_path = self.metadata_dir_path / f"{sample_id}.json"
        metadata_path.write_text(
            json.dumps(asdict(sample), indent=2, default=str), encoding="utf-8"
        )
        sample.metadata_path = str(metadata_path)

        # Update stats
        self.stats.total_captured += 1
        if accepted:
            self.stats.total_accepted += 1
            self.stats.per_class_counts[label] = (
                self.stats.per_class_counts.get(label, 0) + 1
            )
        else:
            self.stats.total_rejected += 1

        self.samples.append(sample)
        self.last_sample = sample
        return sample

    # ------------------------------------------------------------------
    # Interactive mode
    # ------------------------------------------------------------------

    def run_interactive(
        self,
        camera_device: int = 0,
        predictor: Optional[Any] = None,
    ) -> List[CollectedSample]:
        """Run interactive webcam collection.

        Keyboard shortcuts
        ------------------
        Space → capture (unlabeled)
        1     → fresh
        2     → stale
        3     → rotten
        4     → uncertain
        q     → quit
        d     → toggle debug overlay
        """
        self.start_session()

        cap = cv2.VideoCapture(camera_device)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {camera_device}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        logger.info("Interactive collection started.")
        captured_samples: List[CollectedSample] = []
        debug = False

        label_keys = {
            ord(" "): "unlabeled",
            ord("1"): "fresh",
            ord("2"): "stale",
            ord("3"): "rotten",
            ord("4"): "uncertain",
        }

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                display_frame = frame.copy()
                self.frame_count += 1

                pred_class = None
                pred_conf = None
                if predictor is not None:
                    try:
                        result = predictor.predict(frame)
                        pred_class = result.freshness_class
                        pred_conf = result.confidence
                    except Exception:  # noqa: BLE001
                        pass

                cv2.putText(
                    display_frame,
                    f"Session: {self.session_id}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Captured: {len(captured_samples)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                if pred_class and pred_conf is not None:
                    cv2.putText(
                        display_frame,
                        f"Pred: {pred_class} ({pred_conf:.2f})",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                    )

                cv2.imshow("FreshSense Collection", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("d"):
                    debug = not debug
                elif key in label_keys:
                    sample = self.capture(
                        frame,
                        label=label_keys[key],
                        predicted_class=pred_class,
                        predicted_confidence=pred_conf,
                        source_info={
                            "frame": self.frame_count,
                            "camera": camera_device,
                        },
                    )
                    captured_samples.append(sample)

        finally:
            cap.release()
            cv2.destroyAllWindows()

        self.end_session()
        return captured_samples

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_accepted_samples(self) -> List[CollectedSample]:
        """Return samples that passed quality checks."""
        return [s for s in self.samples if s.accepted]

    def get_rejected_samples(self) -> List[CollectedSample]:
        """Return samples that failed quality checks."""
        return [s for s in self.samples if not s.accepted]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_quality(self, frame: np.ndarray) -> QualityMetrics:
        """Compute image quality metrics from a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        return QualityMetrics(
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            resolution=(w, h),
            aspect_ratio=float(w) / float(h) if h > 0 else 0.0,
        )

    def _check_quality(
        self, quality: QualityMetrics
    ) -> Tuple[bool, Optional[str]]:
        """Return (accepted, rejection_reason)."""
        if not self.config.quality_checks:
            return True, None
        if quality.blur_score < self.config.blur_threshold:
            return (
                False,
                f"blur_score={quality.blur_score:.1f} < {self.config.blur_threshold}",
            )
        if quality.brightness < self.config.brightness_min:
            return (
                False,
                f"brightness={quality.brightness:.1f} < {self.config.brightness_min}",
            )
        if quality.brightness > self.config.brightness_max:
            return (
                False,
                f"brightness={quality.brightness:.1f} > {self.config.brightness_max}",
            )
        if quality.contrast < self.config.contrast_min:
            return (
                False,
                f"contrast={quality.contrast:.1f} < {self.config.contrast_min}",
            )
        return True, None

    def _save_session_metadata(self) -> None:
        """Persist session-level metadata JSON."""
        session_meta = {
            "session_id": self.session_id,
            "start_time": self.session_start_time,
            "end_time": time.time(),
            "duration_seconds": self.stats.session_duration_seconds,
            "stats": asdict(self.stats),
            "sample_ids": [s.sample_id for s in self.samples],
        }
        path = self.metadata_dir_path / f"session_{self.session_id}.json"
        path.write_text(json.dumps(session_meta, indent=2, default=str), encoding="utf-8")
