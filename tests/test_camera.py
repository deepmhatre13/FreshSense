"""Tests for the camera module (src/inference/camera.py) using a fake capture.

No physical webcam is required: ``cv2.VideoCapture`` is patched with a fake
implementation that mimics OpenCV behaviour.
"""

import numpy as np
import pytest

import src.inference.camera as cam_mod
from src.inference.camera import Camera, CameraConfig, CameraOpenError

# Which fake "device indices" are available in this test session.
AVAILABLE = {0, 1}


class FakeCapture:
    """Minimal stand-in for cv2.VideoCapture."""

    instances = []

    def __init__(self, index=0):
        self.index = index
        self.opened = index in AVAILABLE
        self.released = False
        self.width = 640
        self.height = 480
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self.opened and not self.released

    def set(self, prop, value):
        if prop == cam_mod.cv2.CAP_PROP_FRAME_WIDTH:
            self.width = int(value)
        elif prop == cam_mod.cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = int(value)
        return True

    def get(self, prop):
        if prop == cam_mod.cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cam_mod.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop == cam_mod.cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        if not self.opened or self.released:
            return False, None
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def patch_capture(monkeypatch):
    FakeCapture.instances.clear()
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", FakeCapture)


class TestCamera:
    def test_open_and_read(self):
        cam = Camera(CameraConfig(device_id=0))
        assert cam.open() is True
        ret, frame, _ = cam.read()
        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        cam.release()
        assert cam.cap is None

    def test_release_is_idempotent(self):
        cam = Camera(CameraConfig(device_id=0))
        cam.release()
        cam.release()  # must not raise

    def test_no_cameras_raises(self, monkeypatch):
        # Force enumerate_cameras to find nothing so open() raises.
        monkeypatch.setattr(
            cam_mod.Camera,
            "enumerate_cameras",
            lambda self, max_index=10: [],
        )
        cam = Camera(CameraConfig(device_id=5))
        with pytest.raises(CameraOpenError):
            cam.open()

    def test_fallback_to_first_available(self):
        # Requested device 5 is unavailable, camera 0 is available.
        cam = Camera(CameraConfig(device_id=5))
        assert cam.open() is True
        assert cam.cap.index == 0

    def test_enumerate(self):
        cam = Camera(CameraConfig(device_id=0))
        assert cam.enumerate_cameras() == [0, 1]

    def test_config_validation(self):
        for kwargs in ({"device_id": -1}, {"width": 0}, {"fps": 0}, {"reconnect_delay": 0}):
            try:
                CameraConfig(**kwargs)
                assert False, f"should raise for {kwargs}"
            except ValueError:
                pass

    def test_get_stats(self):
        cam = Camera(CameraConfig(device_id=0))
        cam.open()
        cam.read()
        stats = cam.get_stats()
        assert stats["frame_count"] == 1
        cam.release()
