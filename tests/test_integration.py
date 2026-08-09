"""End-to-end integration test: real model + fake camera through the pipeline."""

import numpy as np
import pytest

from src.inference.camera import CameraConfig
from src.inference.fps import FPSConfig
from src.inference.overlay import OverlayConfig
from src.inference.pipeline import Pipeline, PipelineConfig, PipelineState
from src.inference.predictor import Predictor
from src.inference.tracker import PredictionTracker, TrackerConfig
from tests.conftest import CHECKPOINT_PATH


class FakeCam:
    def __init__(self):
        self.count = 0
        self.released = False

    def open(self):
        return True

    def read(self):
        self.count += 1
        return True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), self.count

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def patch_display(monkeypatch):
    monkeypatch.setattr("src.inference.pipeline.cv2.imshow", lambda *a, **k: None)
    monkeypatch.setattr("src.inference.pipeline.cv2.destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr("src.inference.pipeline.cv2.namedWindow", lambda *a, **k: None)


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="checkpoint not available")
def test_end_to_end(monkeypatch):
    config = PipelineConfig(
        camera=CameraConfig(device_id=0, width=640, height=480, fps=30),
        fps=FPSConfig(),
        overlay=OverlayConfig(),
        tracker=TrackerConfig(),
        predictor_checkpoint=str(CHECKPOINT_PATH),
    )
    pipe = Pipeline(config)
    pipe.camera = FakeCam()
    pipe.predictor = Predictor(str(CHECKPOINT_PATH))  # real model
    pipe.tracker = PredictionTracker(config.tracker, pipe.predictor.get_class_names())
    pipe.state = PipelineState.RUNNING

    keys = iter([0] * 2 + [ord("q")])
    monkeypatch.setattr("src.inference.pipeline.cv2.waitKey", lambda _: next(keys))

    pipe.run()

    assert pipe.state == PipelineState.STOPPED
    assert pipe.camera.released is True
    assert pipe.frame_count >= 2
