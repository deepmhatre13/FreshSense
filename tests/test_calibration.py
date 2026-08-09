"""Tests for calibration and improved confidence fusion."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.calibration import TemperatureScaling
from src.inference.confidence_fusion import ImprovedConfidenceFusion, ConfidenceFusionConfig


def test_temperature_scaling_forward():
    ts = TemperatureScaling(init_temp=2.0)
    logits = torch.randn(4, 3)
    scaled = ts(logits)
    assert scaled.shape == logits.shape
    assert torch.allclose(scaled, logits / 2.0)


def test_temperature_scaling_calibrate():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3))
    ts = TemperatureScaling(init_temp=1.0)
    x = torch.randn(20, 4)
    y = torch.randint(0, 3, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    temp = ts.calibrate(loader, model, torch.device("cpu"))
    assert temp > 0


def test_improved_confidence_fusion_basic():
    fusion = ImprovedConfidenceFusion()
    cls, conf, info = fusion.fuse(0.9, 0.8, class_name="apple", timestamp=0.0)
    assert cls == "apple"
    assert 0.0 <= conf <= 1.0
    assert info["final_confidence"] == conf


def test_improved_confidence_fusion_stability():
    config = ConfidenceFusionConfig(stability_window=3, confidence_threshold=0.5, transition_smoothing=0.3)
    fusion = ImprovedConfidenceFusion(config)
    cls1, conf1, _ = fusion.fuse(0.6, 0.6, class_name="apple", timestamp=0.0)
    cls2, conf2, _ = fusion.fuse(0.6, 0.6, class_name="apple", timestamp=0.1)
    assert cls1 == "apple"
    assert cls2 == "apple"
    assert conf2 >= conf1 - 0.01


def test_improved_confidence_fusion_transition_penalty():
    fusion = ImprovedConfidenceFusion()
    fusion.fuse(0.9, 0.8, class_name="apple", timestamp=0.0)
    cls2, conf2, _ = fusion.fuse(0.9, 0.8, class_name="banana", timestamp=0.1)
    assert cls2 == "banana"
    assert conf2 < 1.0
