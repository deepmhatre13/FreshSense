"""Main inference pipeline orchestrator for FreshSense Phase 3.

.. deprecated:: Phase 4
    This module is deprecated and represents the legacy classify-only pipeline.
    It will be preserved for backward compatibility but is no longer the
    canonical pipeline. Please use :mod:`src.inference.detection_pipeline`
    for the Phase 4 detect-track-crop-classify-fuse architecture.

This module provides the main pipeline that orchestrates real-time inference:

- Frame capture from camera
- Preprocessing using Phase 1 pipeline
- Prediction with temporal smoothing
- Image quality assessment
- Overlay rendering
- Session statistics and logging
- Display and keyboard handling

The Pipeline class is the main entry point for Phase 3 real-time inference.
"""

from __future__ import annotations

import logging
import signal
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from configs.config import Config
from src.inference.camera import Camera, CameraConfig, CameraError
from src.inference.fps import FPSMonitor, FPSConfig
from src.inference.overlay import Overlay, OverlayConfig, ColorScheme
from src.inference.predictor import Predictor, PredictionResult
from src.inference.quality import QualityAssessor, QualityConfig, QualityReport
from src.inference.stabilizer import Stabilizer, StabilizerConfig, StabilizedPrediction
from src.inference.statistics import SessionLogger, SessionStatistics
from src.inference.tracker import PredictionTracker, TrackerConfig, TrackedPrediction

logger = logging.getLogger(__name__)

__all__ = ["Pipeline", "PipelineConfig", "PipelineState"]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the inference pipeline.

    Attributes:
        camera: CameraConfig for webcam capture.
        fps: FPSConfig for performance monitoring.
        overlay: OverlayConfig for rendering.
        tracker: TrackerConfig for prediction smoothing (Phase 2).
        stabilizer: StabilizerConfig for temporal stabilization (Phase 3).
        quality: QualityConfig for image quality assessment (Phase 3).
        predictor_checkpoint: Path to best_model.pth.
        confidence_threshold: Minimum confidence to display prediction.
        save_frames: If True, save frames on 'S' key press.
        save_dir: Directory for saved frames.
        save_logs: If True, save session logs.
        session_log_dir: Directory for session logs.
    """

    camera: CameraConfig
    fps: FPSConfig
    overlay: OverlayConfig
    tracker: TrackerConfig
    predictor_checkpoint: str
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    confidence_threshold: float = 0.5
    save_frames: bool = True
    save_dir: str = "captured_frames"
    save_logs: bool = True
    session_log_dir: str = "logs/session"

    def __post_init__(self) -> None:
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0].")


class PipelineState:
    """Enumeration of pipeline states."""

    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class Pipeline:
    """Main inference pipeline orchestrator for Phase 3.

    This class coordinates:
    - Camera capture
    - Prediction engine
    - Temporal smoothing (Phase 2 tracker + Phase 3 stabilizer)
    - Image quality assessment (Phase 3)
    - Overlay rendering (Phase 3 enhanced)
    - Session statistics and logging (Phase 3)
    - FPS monitoring
    - Keyboard input handling

    The pipeline runs in a main loop until 'Q' is pressed or interrupted.

    Args:
        config: PipelineConfig instance with all settings.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.state = PipelineState.STOPPED

        # Initialize components
        self.camera = Camera(config.camera)
        self.fps_monitor = FPSMonitor(config.fps)
        self.overlay = Overlay(config.overlay)

        # Phase 2: tracker (kept for backward compatibility)
        self.tracker: Optional[PredictionTracker] = None

        # Phase 3: new components
        self.predictor: Optional[Predictor] = None
        self.stabilizer: Optional[Stabilizer] = None
        self.quality_assessor: Optional[QualityAssessor] = None
        self.statistics: Optional[SessionStatistics] = None
        self.session_logger: Optional[SessionLogger] = None

        # Display flags
        self.show_fps = True
        self.show_confidence = True
        self.show_quality = True
        self.show_stats = True

        # Frame counter
        self.frame_count = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing Phase 3 pipeline...")

        # Open camera
        self.camera.open()

        # Load predictor
        self.predictor = Predictor(self.config.predictor_checkpoint)

        # Initialize tracker (Phase 2, kept for compatibility)
        class_names = self.predictor.get_class_names()
        self.tracker = PredictionTracker(self.config.tracker, class_names)

        # Initialize stabilizer (Phase 3)
        self.stabilizer = Stabilizer(self.config.stabilizer, class_names)

        # Initialize quality assessor (Phase 3)
        self.quality_assessor = QualityAssessor(self.config.quality)

        # Initialize statistics and logging (Phase 3)
        self.statistics = SessionStatistics()
        self.session_logger = SessionLogger(
            log_dir=self.config.session_log_dir,
            save_logs=self.config.save_logs,
        )

        self.state = PipelineState.RUNNING
        logger.info("Pipeline initialized successfully.")
    def run(self) -> None:
        """Run the main inference loop."""
        if self.state != PipelineState.RUNNING:
            logger.error("Pipeline not initialized. Call initialize() first.")
            return

        logger.info("Starting Phase 3 pipeline...")
        print("\n" + "=" * 60)
        print("FreshSense Phase 3 Pipeline")
        print("=" * 60)
        print("Controls:")
        print("  Q - Quit")
        print("  F - Toggle FPS display")
        print("  C - Toggle confidence display")
        print("  S - Save frame")
        print("=" * 60 + "\n")

        # Lazy initialization for components that may have been injected manually
        if self.statistics is None:
            self.statistics = SessionStatistics()
        if self.session_logger is None:
            self.session_logger = SessionLogger(
                log_dir=self.config.session_log_dir,
                save_logs=self.config.save_logs,
            )
        if self.quality_assessor is None:
            self.quality_assessor = QualityAssessor(self.config.quality)
        if self.stabilizer is None:
            class_names = (
                self.predictor.get_class_names()
                if self.predictor is not None
                else ["fresh", "stale", "rotten"]
            )
            self.stabilizer = Stabilizer(self.config.stabilizer, class_names)

        while self.state == PipelineState.RUNNING:
            loop_start = self.fps_monitor.start_frame()

            # 1. Capture frame
            ret, frame, timestamp = self.camera.read()
            if not ret or frame is None:
                logger.warning("Frame read failed, skipping.")
                continue

            self.frame_count += 1
            self.statistics.total_frames += 1

            # 2. Assess image quality (Phase 3)
            quality_report = self.quality_assessor.assess(frame)

            # Record quality metrics
            self.statistics.record_quality(
                brightness=quality_report.brightness,
                blur_variance=quality_report.blur_variance,
            )

            # Check if quality is acceptable
            if not quality_report.is_motion_ok:
                self.statistics.record_motion_skip()
                # Display warning and skip inference
                display_frame = self.overlay.draw_quality_warning(
                    frame.copy(), "Hold fruit still", quality_report
                )
                cv2.imshow("FreshSense", display_frame)
                if not self._handle_keyboard(None, None, frame):
                    break
                self.fps_monitor.end_frame(False)
                continue

            if not quality_report.is_brightness_ok:
                self.statistics.record_lighting_warning()

            if not quality_report.is_quality_ok:
                # Display quality warnings but still run inference
                display_frame = self.overlay.draw_quality_warning(
                    frame.copy(), quality_report.warnings[0] if quality_report.warnings else "Poor Quality",
                    quality_report,
                )
            else:
                display_frame = frame.copy()

            # 3. Run inference
            inference_start = self.fps_monitor.start_frame()
            try:
                raw_result = self.predictor.predict(frame)
            except Exception as exc:
                logger.error("Inference failed: %s", exc)
                continue
            inference_end = self.fps_monitor.end_frame(True)

            # 4. Stabilize prediction (Phase 3)
            stabilized = self.stabilizer.update(
                raw_result.freshness_class,
                raw_result.confidence,
            )

            # 5. Update tracker (Phase 2, for backward compatibility)
            tracked = self.tracker.update(stabilized.label, stabilized.confidence)

            # 6. Record statistics
            current_fps = self.fps_monitor.get_stats().current_fps
            self.statistics.record_prediction(
                confidence=stabilized.confidence,
                latency_ms=raw_result.latency_ms,
                fps=current_fps,
            )

            if stabilized.is_uncertain:
                self.statistics.record_uncertain()

            # 7. Log frame
            self.session_logger.log_frame(
                prediction=stabilized.label,
                confidence=stabilized.confidence,
                fps=current_fps,
                latency_ms=raw_result.latency_ms,
                brightness=quality_report.brightness,
                blur_variance=quality_report.blur_variance,
                warnings=quality_report.warnings,
            )

            # 8. Render overlay (Phase 3 enhanced)
            display_frame = self.overlay.draw_phase3_overlay(
                frame=display_frame,
                stabilized=stabilized,
                tracked=tracked,
                raw_result=raw_result,
                quality_report=quality_report,
                fps_monitor=self.fps_monitor,
                frame_number=self.frame_count,
                predictor=self.predictor,
            )

            # 9. Display
            cv2.imshow("FreshSense", display_frame)

            # 10. Handle keyboard
            if not self._handle_keyboard(raw_result, tracked, frame):
                break

            # Print periodic status
            if self.fps_monitor.should_log():
                self._print_status(stabilized, quality_report)

        self.shutdown()
    def _handle_keyboard(self, result, tracked, frame) -> bool:
        """Handle keyboard input. Returns False if should stop."""
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            logger.info("Quit requested by user.")
            return False
        elif key == ord("f"):
            self.show_fps = not self.show_fps
            logger.debug("FPS display: %s", self.show_fps)
        elif key == ord("c"):
            self.show_confidence = not self.show_confidence
            logger.debug("Confidence display: %s", self.show_confidence)
        elif key == ord("s") and self.config.save_frames:
            import os

            os.makedirs(self.config.save_dir, exist_ok=True)
            filename = os.path.join(
                self.config.save_dir,
                f"frame_{self.frame_count:06d}.jpg",
            )
            cv2.imwrite(filename, frame)
            logger.info("Frame saved: %s", filename)
            print(f"Frame saved: {filename}")

        return True

    def _print_status(self, stabilized, quality_report) -> None:
        """Print periodic status to console."""
        stats = self.fps_monitor.get_stats()
        print(
            f"Frame {self.frame_count:6d} | "
            f"FPS: {stats.current_fps:5.1f} | "
            f"Pred: {stabilized.label:8s} | "
            f"Conf: {stabilized.confidence:5.1%} | "
            f"Quality: {quality_report.quality_score:.2f}"
        )

    def _signal_handler(self, signum, frame) -> None:
        """Handle interrupt signals for graceful shutdown."""
        logger.info("Signal %d received, shutting down...", signum)
        self.state = PipelineState.STOPPING
    def shutdown(self) -> None:
        """Shutdown pipeline and release resources."""
        logger.info("Shutting down pipeline...")
        self.state = PipelineState.STOPPED

        # Finalize statistics
        if self.statistics:
            self.statistics.finalize()

        # Save session log
        if self.session_logger:
            self.session_logger.save_summary(self.statistics)
            self.session_logger.close()

        # Release camera
        if self.camera:
            self.camera.release()

        # Close windows
        cv2.destroyAllWindows()

        # Print final stats
        stats = self.fps_monitor.get_stats()
        print("\n" + "=" * 60)
        print("Pipeline Statistics")
        print("=" * 60)
        print(f"Total frames: {stats.total_frames}")
        print(f"Elapsed time: {stats.elapsed_time:.1f}s")
        print(f"Average FPS: {stats.average_fps:.1f}")
        print(f"Average latency: {stats.avg_latency_ms:.1f}ms")
        print("=" * 60)

        # Print session statistics
        if self.statistics:
            sess_stats = self.statistics.to_dict()
            print("\nSession Statistics")
            print("=" * 60)
            for key, value in sess_stats.items():
                print(f"{key}: {value}")
            print("=" * 60)

        logger.info("Pipeline shutdown complete")


def load_pipeline_from_yaml(yaml_path: str) -> Pipeline:
    """Load pipeline configuration from YAML file.

    Args:
        yaml_path: Path to configuration YAML file.

    Returns:
        Configured Pipeline instance.
    """
    from src.inference.camera import CameraConfig
    from src.inference.fps import FPSConfig
    from src.inference.overlay import OverlayConfig
    from src.inference.quality import QualityConfig
    from src.inference.stabilizer import StabilizerConfig
    from src.inference.tracker import TrackerConfig

    # Load base config
    config = Config.from_yaml(yaml_path)

    # Create pipeline configs
    camera_config = CameraConfig(
        device_id=0,
        width=640,
        height=480,
        fps=30,
    )

    fps_config = FPSConfig()
    overlay_config = OverlayConfig()
    tracker_config = TrackerConfig()
    stabilizer_config = StabilizerConfig(
        ema_alpha=config.inference.ema_alpha,
        vote_window=config.inference.vote_window,
        lock_frames=config.inference.lock_frames,
        confidence_threshold=config.inference.stabilizer_confidence_threshold,
    )
    quality_config = QualityConfig(
        brightness_min=config.inference.brightness_min,
        brightness_max=config.inference.brightness_max,
        blur_threshold=config.inference.blur_threshold,
        contrast_min=config.inference.contrast_min,
        motion_threshold=config.inference.motion_threshold,
        use_motion_detection=config.inference.use_motion_detection,
    )

    pipeline_config = PipelineConfig(
        camera=camera_config,
        fps=fps_config,
        overlay=overlay_config,
        tracker=tracker_config,
        stabilizer=stabilizer_config,
        quality=quality_config,
        predictor_checkpoint="models/checkpoints/best_model.pth",
        confidence_threshold=0.5,
        save_logs=config.inference.save_logs,
        session_log_dir=config.inference.session_log_dir,
    )

    return Pipeline(pipeline_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("FreshSense Phase 3 Pipeline - Self Test")
    print("=" * 60)

    checkpoint_path = "models/checkpoints/best_model.pth"
    import pathlib

    if not pathlib.Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using: python -m src.main")
        exit(1)

    from src.inference.camera import CameraConfig
    from src.inference.fps import FPSConfig
    from src.inference.overlay import OverlayConfig
    from src.inference.quality import QualityConfig
    from src.inference.stabilizer import StabilizerConfig
    from src.inference.tracker import TrackerConfig

    config = PipelineConfig(
        camera=CameraConfig(device_id=0, width=640, height=480, fps=30),
        fps=FPSConfig(),
        overlay=OverlayConfig(),
        tracker=TrackerConfig(),
        stabilizer=StabilizerConfig(),
        quality=QualityConfig(),
        predictor_checkpoint=checkpoint_path,
    )

    pipeline = Pipeline(config)

    try:
        pipeline.initialize()
        pipeline.run()
    except CameraError as exc:
        print(f"\nCamera error: {exc}")
        print("Please connect a webcam and try again.")
        exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        exit(1)
