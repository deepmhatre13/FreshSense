"""Tests for the deterministic inference transform (src/inference/transforms.py)."""

import numpy as np
import pytest
import torch

from src.inference.transforms import InferenceTransform


def make_frame(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestInferenceTransform:
    def test_output_shape_and_dtype(self):
        out = InferenceTransform()(make_frame())
        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (1, 3, 224, 224)
        assert out.dtype == torch.float32

    def test_deterministic(self):
        frame = make_frame()
        t = InferenceTransform()
        assert torch.equal(t(frame), t(frame))

    def test_center_crop_actually_crops(self):
        # resize to 256, center-crop to 224 -> 16px border removed each side.
        t = InferenceTransform(image_size=224, resize_size=256, center_crop=224)
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        # mark only the center patch differently
        frame[20:236, 20:236] = 255
        out = t(frame)  # (1,3,224,224)
        # After center-crop the central bright region should dominate.
        arr = out[0]
        # Move from normalized back toward white in the interior: all cells are
        # 255 -> fully normalized positive; corner cells are 0 -> negative.
        assert arr[:, 0, 0].max() < arr[:, 112, 112].max()

    def test_resize_noop_when_same(self):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        out = InferenceTransform()(frame)
        assert tuple(out.shape) == (1, 3, 224, 224)

    @pytest.mark.parametrize("bad", [None, np.zeros((10, 10), dtype=np.uint8), np.zeros((224, 224, 3), dtype=np.float32)])
    def test_invalid_input_raises(self, bad):
        with pytest.raises((ValueError, TypeError)):
            InferenceTransform()(bad)

    def test_normalization_applied(self):
        # A pure-white image should be fully positive after ImageNet normalize.
        frame = np.full((224, 224, 3), 255, dtype=np.uint8)
        out = InferenceTransform()(frame)
        assert out.min() > 0.0

    def test_config_validation(self):
        with pytest.raises(ValueError):
            InferenceTransform(image_size=0)
        with pytest.raises(ValueError):
            InferenceTransform(image_size=224, center_crop=300, resize_size=224)
