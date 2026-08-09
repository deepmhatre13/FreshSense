"""Tests for the inference pipeline (src/inference/pipeline.py).

A fake camera and fake predictor are injected so the full run loop can be
exercised without a physical webcam or a heavyweight model forward pass.
"""

import numpy as np
import pytest

from src.inference.camera import CameraConfig
from src.inference.fps import FPSConfig
from src.inference.overlay import OverlayConfig
from src.inference.pipeline import Pipeline, PipelineConfig, PipelineState
from src.inference.predictor import PredictionResult
from src.inference.tracker import PredictionTracker, TrackerConfig

BEST = "models/checkpoints/best_model.pth"


class TestPipelineConfigBackwardCompatibility:
    """Regression tests for legacy PipelineConfig construction without stabilizer/quality."""

    def test_legacy_construction_without_stabilizer_and_quality(self):
        """PipelineConfig should allow construction without explicit stabilizer/quality."""
        cfg = PipelineConfig(
            camera=CameraConfig(device_id=0, width=640, height=480, fps=30),
            fps=FPSConfig(),
            overlay=OverlayConfig(),
            tracker=TrackerConfig(),
            predictor_checkpoint=BEST,
        )
        assert cfg.camera.device_id == 0
        assert cfg.predictor_checkpoint == BEST
        # Defaults should be applied automatically
        from src.inference.stabilizer import StabilizerConfig
        from src.inference.quality import QualityConfig

        assert isinstance(cfg.stabilizer, StabilizerConfig)
        assert isinstance(cfg.quality, QualityConfig)

    def test_explicit_stabilizer_and_quality_still_works(self):
        """Explicit stabilizer/quality should override defaults."""
        from src.inference.stabilizer import StabilizerConfig
        from src.inference.quality import QualityConfig

        stabilizer = StabilizerConfig(ema_alpha=0.5, vote_window=10)
        quality = QualityConfig(brightness_min=30, blur_threshold=50.0)
        cfg = PipelineConfig(
            camera=CameraConfig(device_id=0, width=640, height=480, fps=30),
            fps=FPSConfig(),
            overlay=OverlayConfig(),
            tracker=TrackerConfig(),
            predictor_checkpoint=BEST,
            stabilizer=stabilizer,
            quality=quality,
        )
        assert cfg.stabilizer is stabilizer
        assert cfg.quality is quality
        assert cfg.stabilizer.ema_alpha == 0.5
        assert cfg.quality.brightness_min == 30


class MockCamera:
    """Stand-in for Camera: yields a few valid frames then reports failure."""

    def __init__(self, frames=3, fail_after=None):
        self.frames = frames
        self.count = 0
        self.fail_after = fail_after
        self.released = False

    def open(self):
        return True

    def read(self):
        if self.fail_after is not None and self.count >= self.fail_after:
            raise Exception("simulated persistent camera failure")
        if self.count >= self.frames:
            return False, None, 0.0
        self.count += 1
        return True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), self.count

    def release(self):
        self.released = True


class MockPredictor:
    def __init__(self):
        self.calls = 0
        self.model_version = "test"

    def get_class_names(self):
        return ["fresh", "stale", "rotten"]

    def predict(self, frame):
        self.calls += 1
        return PredictionResult(
            fruit_name="apple",
            freshness_class="fresh",
            confidence=0.97,
            probabilities=[0.97, 0.02, 0.01],
            timestamp=float(self.calls),
            latency_ms=1.2,
            device="cpu",
            model_version="test",
        )


def make_pipeline(save_dir="captured_frames"):
    config = PipelineConfig(
        camera=CameraConfig(device_id=0, width=640, height=480, fps=30),
        fps=FPSConfig(),
        overlay=OverlayConfig(),
        tracker=TrackerConfig(),
        predictor_checkpoint=BEST,
        save_dir=save_dir,
    )
    pipe = Pipeline(config)
    pipe.camera = MockCamera(frames=100)
    pipe.predictor = MockPredictor()
    pipe.tracker = PredictionTracker(config.tracker, ["fresh", "stale", "rotten"])
    pipe.state = PipelineState.RUNNING
    return pipe


@pytest.fixture(autouse=True)
def patch_display(monkeypatch):
    monkeypatch.setattr("src.inference.pipeline.cv2.imshow", lambda *a, **k: None)
    monkeypatch.setattr("src.inference.pipeline.cv2.destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr("src.inference.pipeline.cv2.namedWindow", lambda *a, **k: None)


class TestRunLoop:
    def test_runs_frames_and_quits(self, monkeypatch):
        keys = iter([0] * 5 + [ord("q")])
        monkeypatch.setattr(
            "src.inference.pipeline.cv2.waitKey", lambda _: next(keys)
        )
        pipe = make_pipeline()
        pipe.run()
        assert pipe.state == PipelineState.STOPPED
        assert pipe.camera.released is True
        assert pipe.frame_count >= 5

    def test_survives_dropped_frames(self, monkeypatch):
        # First frame dropped, then valid frames.
        pipe = make_pipeline()
        pipe.camera = MockCamera(frames=100)
        monkeypatch.setattr(
            "src.inference.pipeline.cv2.waitKey", lambda _: ord("q")
        )
        pipe.run()
        assert pipe.state == PipelineState.STOPPED


class TestKeyboard:
    def _pipe(self, monkeypatch):
        pipe = make_pipeline()
        pipe.predictor = MockPredictor()
        from src.inference.pipeline import cv2

        return pipe

    def frame(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def result(self):
        return MockPredictor().predict(self.frame())

    def tracked(self, pipe):
        return pipe.tracker.update("fresh", 0.9)

    def test_q_returns_false(self, monkeypatch):
        monkeypatch.setattr("src.inference.pipeline.cv2.waitKey", lambda _: ord("q"))
        pipe = self._pipe(monkeypatch)
        assert pipe._handle_keyboard(self.result(), self.tracked(pipe), self.frame()) is False

    def test_esc_returns_false(self, monkeypatch):
        monkeypatch.setattr("src.inference.pipeline.cv2.waitKey", lambda _: 27)
        pipe = self._pipe(monkeypatch)
        assert pipe._handle_keyboard(self.result(), self.tracked(pipe), self.frame()) is False

    def test_f_toggles(self, monkeypatch):
        monkeypatch.setattr("src.inference.pipeline.cv2.waitKey", lambda _: ord("f"))
        pipe = self._pipe(monkeypatch)
        assert pipe.show_fps is True
        pipe._handle_keyboard(self.result(), self.tracked(pipe), self.frame())
        assert pipe.show_fps is False

    def test_c_toggles(self, monkeypatch):
        monkeypatch.setattr("src.inference.pipeline.cv2.waitKey", lambda _: ord("c"))
        pipe = self._pipe(monkeypatch)
        assert pipe.show_confidence is True
        pipe._handle_keyboard(self.result(), self.tracked(pipe), self.frame())
        assert pipe.show_confidence is False

    def test_s_saves_frame(self, monkeypatch, tmp_path):
        import os

        pipe = make_pipeline(save_dir=str(tmp_path))
        pipe.predictor = MockPredictor()
        os.makedirs(pipe.config.save_dir, exist_ok=True)
        monkeypatch.setattr("src.inference.pipeline.cv2.waitKey", lambda _: ord("s"))
        pipe._handle_keyboard(self.result(), self.tracked(pipe), self.frame())
        # A saved .jpg should exist in the tmp dir.
        saved = list(tmp_path.glob("*.jpg"))
        assert len(saved) == 1

    def test_shutdown_releases_resources(self, monkeypatch):
        pipe = self._pipe(monkeypatch)
        pipe.shutdown()
        assert pipe.state == PipelineState.STOPPED
        assert pipe.camera.released is True
