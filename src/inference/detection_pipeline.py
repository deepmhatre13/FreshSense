"""Detection + classification pipeline for FreshSense Phase 4.

Transforms the webcam feed into a multi-stage vision pipeline:

    Camera -> Quality -> Detection -> Crop -> Track -> Classify -> Stabilize -> Overlay

Each tracked fruit gets its own independent Phase-3 stabilizer, so predictions
for individual fruits stay stable regardless of other fruits in the frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection import (
    DetectorConfig,
    Detection,
    SUPPORTED_CLASSES,
)
from src.detection.factory import DetectorFactory
from src.inference.confidence_fusion import ConfidenceFusion, FusionConfig
from src.inference.cropper import Cropper, CropperConfig
from src.inference.detection_tracker import DetectionTracker, TrackerConfig
from src.inference.fruit_result import FruitResult, MultiFruitResult
from src.inference.quality import QualityAssessor, QualityConfig, QualityReport
from src.inference.shelf_life import (
    ShelfLifeEstimator,
    ShelfLifeConfig,
    normalize_storage_condition,
)
from src.inference.stabilizer import Stabilizer, StabilizerConfig
from src.inference.predictor import Predictor

# Freshness label vocabulary passed to each per-fruit stabilizer.
from src.freshness import (
    FRESH,
    ROTTEN,
    UNCERTAIN,
    DATA_NOT_AVAILABLE,
    freshness_supported,
)

# Production freshness vocabulary (controlled).
# The stabiliser must know every value the pipeline can emit, including
# ``"uncertain"`` (below confidence threshold) and ``"data_not_available"``
# (fruit without a validated freshness model).
FRESHNESS_CLASSES = [FRESH, ROTTEN, UNCERTAIN, DATA_NOT_AVAILABLE]


logger = logging.getLogger(__name__)

__all__ = ["DetectionPipeline", "DetectionPipelineConfig"]


@dataclass(frozen=True)
class DetectionPipelineConfig:
    """Configuration aggregating all Phase 4 sub-configs.

    Attributes:
        detector_name: Backend name ("yolo", "simple", "mock").
        detector_weights: Model weight file name.
        confidence_threshold: Min detection confidence.
        iou_threshold: NMS IoU threshold.
        max_detections: Max detections per frame.
        classify_every_n_frames: Adaptive classification cadence.
        crop_expand_scale: Box expansion fraction.
        crop_min_side: Min crop side in pixels.
        crop_min_area: Min crop area in pixels.
        crop_target_size: Resize target for classifier input.
        tracker_iou_threshold: Tracker association IoU.
        tracker_max_distance: Tracker association distance.
        tracker_max_lost_frames: Track expiry.
        detection_weight: Fusion weight for detector confidence.
        classification_weight: Fusion weight for classifier confidence.
        ema_alpha: Stabilizer EMA alpha.
        vote_window: Stabilizer majority vote window.
        lock_frames: Stabilizer lock frames.
        stabilizer_confidence_threshold: Min confidence for stable prediction.
        quality: Quality config.
        session_log_dir: Path for session logs.
        save_logs: Whether to log sessions.
        shelf_life_enabled: Whether per-fruit shelf-life estimation runs.
        default_storage_condition: Storage assumption recorded when a request
            omits an explicit condition ("ambient" or "refrigerated").
    """

    detector_name: str = "yolo"
    detector_weights: str = "models/detection/detector/weights/best.pt"
    detector_imgsz: int = 640
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    max_detections: int = 20
    classify_every_n_frames: int = 3
    crop_expand_scale: float = 0.08
    crop_min_side: int = 32
    crop_min_area: int = 1024
    crop_target_size: int = 224
    tracker_iou_threshold: float = 0.3
    tracker_max_distance: float = 120.0
    tracker_max_lost_frames: int = 15
    detection_weight: float = 0.4
    classification_weight: float = 0.6
    ema_alpha: float = 0.2
    vote_window: int = 15
    lock_frames: int = 5
    stabilizer_confidence_threshold: float = 0.70
    quality: QualityConfig = None
    session_log_dir: str = "logs/session"
    save_logs: bool = True
    shelf_life_enabled: bool = True
    default_storage_condition: str = "ambient"

    def __post_init__(self) -> None:
        if self.quality is None:
            object.__setattr__(self, "quality", QualityConfig())


class DetectionPipeline:
    """Multi-stage fruit detection + classification pipeline.

    Pipeline stages per frame:

    1. Image quality assessment
    2. Object detection (YOLO / factory)
    3. Multi-object tracking (IoU based)
    4. Per-fruit crop + crop-quality gating
    5. Adaptive classification (every N frames)
    6. Per-fruit Phase-3 stabilization
    7. Confidence fusion (detector x classifier)
    8. Shelf-life estimation
    9. Multi-fruit result assembly

    Args:
        config: DetectionPipelineConfig.
        predictor: Pre-loaded EfficientNet Predictor (Phase 1).
    """

    def __init__(
        self,
        config: DetectionPipelineConfig,
        predictor: Optional[Predictor] = None,
    ) -> None:
        self.config = config
        cfg = config

        # Quality assessor (reused from Phase 3)
        self.quality_assessor = QualityAssessor(cfg.quality)

        # Detector (factory + abstraction)
        det_cfg = DetectorConfig(
            model_path=cfg.detector_weights,
            confidence_threshold=cfg.confidence_threshold,
            iou_threshold=cfg.iou_threshold,
            image_size=cfg.detector_imgsz,
            max_detections=cfg.max_detections,
            class_names=SUPPORTED_CLASSES,
        )
        self.detector = DetectorFactory.create(cfg.detector_name, det_cfg)

        # Tracker
        self.tracker = DetectionTracker(TrackerConfig(
            iou_threshold=cfg.tracker_iou_threshold,
            max_center_distance=cfg.tracker_max_distance,
            max_lost_frames=cfg.tracker_max_lost_frames,
        ))

        # Cropper
        self.cropper = Cropper(CropperConfig(
            expand_scale=cfg.crop_expand_scale,
            min_side=cfg.crop_min_side,
            min_area=cfg.crop_min_area,
            target_size=cfg.crop_target_size,
        ))

        # Classifier
        self.predictor = predictor

        # Confidence fusion
        self.fusion = ConfidenceFusion(FusionConfig(
            detection_weight=cfg.detection_weight,
            classification_weight=cfg.classification_weight,
        ))

        # Shelf life
        self.shelf_life = ShelfLifeEstimator(ShelfLifeConfig())

        # Per-fruit stabilizers (keyed by tracking id)
        self._stabilizers: Dict[int, Stabilizer] = {}
        self._classify_counter: Dict[int, int] = {}
        self._last_classifications: Dict[int, Tuple[str, float, float, str]] = {}
        self._last_result: Optional[MultiFruitResult] = None

    def _get_stabilizer(self, track_id: int) -> Stabilizer:
        """Get or create a per-fruit stabilizer (lazy)."""
        if track_id not in self._stabilizers:
            self._stabilizers[track_id] = Stabilizer(
                StabilizerConfig(
                    ema_alpha=self.config.ema_alpha,
                    vote_window=self.config.vote_window,
                    lock_frames=self.config.lock_frames,
                    confidence_threshold=self.config.stabilizer_confidence_threshold,
                ),
                FRESHNESS_CLASSES,
            )
            self._classify_counter[track_id] = 0
        return self._stabilizers[track_id]


    def process_frame(
        self,
        frame: np.ndarray,
        storage_condition: Optional[str] = None,
    ) -> MultiFruitResult:
        """Run the full Phase 4 pipeline on one frame.

        Args:
            frame: BGR image.
            storage_condition: Optional caller-supplied storage assumption
                (``"ambient"`` or ``"refrigerated"``). Invalid values raise
                ``ValueError`` BEFORE any inference runs, so API callers can
                map the rejection to HTTP 400.
        """
        # Fail fast on an invalid storage condition (no model work wasted).
        if storage_condition is not None:
            normalize_storage_condition(storage_condition)
        h, w = frame.shape[:2]
        # 1. Quality assessment
        quality_report = self.quality_assessor.assess(frame)

        # 2. Detection
        det_result = self.detector.detect(frame)

        # 3. Tracking
        tracked = self.tracker.update(det_result.detections)

        # 4-8. Per fruit: crop, classify, stabilize, fuse, shelf life
        fruit_results: List[FruitResult] = []
        for det in tracked:
            result = self._process_fruit(frame, det, quality_report, w, h, storage_condition)
            if result is not None:
                fruit_results.append(result)

        self._last_result = MultiFruitResult(
            fruits=fruit_results,
            frame_width=w,
            frame_height=h,
            unidentified_count=det_result.count - len(tracked),
        )
        return self._last_result

    def _process_fruit(
        self,
        frame: np.ndarray,
        det: Detection,
        quality: QualityReport,
        width: int,
        height: int,
        storage_condition: Optional[str] = None,
    ) -> Optional[FruitResult]:
        """Crop, classify, stabilize a single detected fruit."""
        # Crop
        crop_result = self.cropper.crop(frame, det)
        if not crop_result.valid:
            logger.debug(
                "Crop rejected for track #%d: %s",
                det.tracking_id,
                crop_result.rejection_reason,
            )
            return None

        crop = crop_result.cropped
        stabilizer = self._get_stabilizer(det.tracking_id)

        # Only grade freshness for fruit types with a validated freshness model.
        # The availability registry (configs/freshness_availability.json) is the
        # single source of truth. Fruits without fresh/rotten training data
        # (Kiwi, Mango, Cherry, Chickoo) are NEVER classified - the predictor is
        # not invoked on them, and freshness is reported as "data_not_available".
        supported = freshness_supported(det.label)

        # Adaptive classification: classify every N frames (and always on frame 1).
        self._classify_counter[det.tracking_id] += 1
        should_classify = (
            self._classify_counter[det.tracking_id] == 1
            or self._classify_counter[det.tracking_id] % self.config.classify_every_n_frames == 0
        )

        if (
            supported
            and should_classify
            and self.predictor is not None
            and crop is not None
        ):
            result = self.predictor.predict(crop)
            freshness = result.freshness_class
            raw_conf = result.confidence
            latency_ms = result.latency_ms
            model_version = result.model_version
            self._last_classifications[det.tracking_id] = (
                freshness,
                raw_conf,
                latency_ms,
                model_version,
            )
        else:
            # Reuse last classification (tracker fills the gap) - but only
            # for a fruit the classifier can grade. Fruits without a
            # validated freshness model always report "data_not_available"
            # and fall back to detector confidence for fusion (no ML guess).
            last = self._last_classifications.get(det.tracking_id)
            if supported and last is not None:
                freshness, raw_conf, latency_ms, model_version = last
            else:
                if not supported:
                    freshness = DATA_NOT_AVAILABLE
                else:
                    freshness = UNCERTAIN
                raw_conf = det.confidence
                latency_ms = 0.0
                model_version = "n/a"

        # Stabilize
        stabilized = stabilizer.update(freshness, raw_conf)

        # Fuse confidence
        fused = self.fusion.fuse(det.confidence, raw_conf)
        fused_conf = fused.fused_confidence

        # Determine final class using stabilized label
        # (majority vote + locking).
        if stabilized.is_uncertain:
            final_class = UNCERTAIN
            is_uncertain = True
        else:
            final_class = stabilized.label
            is_uncertain = False

        # Shelf life
        shelf = self.shelf_life.estimate(
            fruit=det.label,
            fused_confidence=fused_conf,
            freshness_class=final_class,
            storage_condition=storage_condition,
        )

        metadata = {
            "latency_ms": latency_ms,
            "model_version": model_version,
            "classify_every": self.config.classify_every_n_frames,
        }

        return FruitResult(
            detection=det,
            stabilized=stabilized,
            fused_confidence=fused_conf,
            freshness_class=final_class,
            shelf_life=shelf,
            is_uncertain=is_uncertain,
            metadata=metadata,
        )


    def initialize(self) -> None:
        """Load detector and warm up."""
        logger.info("Initializing Phase 4 detection pipeline...")
        self.detector.load()
        self.detector.warmup()
        if self.predictor is not None:
            logger.info("Classifier ready: %s", self.predictor)
        logger.info("Phase 4 detection pipeline ready.")

    def shutdown(self) -> None:
        """Release detector resources."""
        self.detector.shutdown()
        self.tracker.reset()
        self._stabilizers.clear()
        self._classify_counter.clear()
        self._last_classifications.clear()
        logger.info("Detection pipeline shut down.")

    def get_stats(self) -> dict:
        """Return pipeline stats for diagnostics."""
        return {
            "total_fruits_tracked": len(self._stabilizers),
            "active_tracks": self.tracker.get_active_tracks(),
            "detector_loaded": self.detector.is_loaded,
            "detector_backend": type(self.detector).__name__,
        }

    def __str__(self) -> str:
        stats = self.get_stats()
        return (
            f"DetectionPipeline("
            f"backend={stats['detector_backend']}, "
            f"loaded={stats['detector_loaded']}, "
            f"fruits={stats['total_fruits_tracked']}, "
            f"active={len(stats['active_tracks'])})"
        )


