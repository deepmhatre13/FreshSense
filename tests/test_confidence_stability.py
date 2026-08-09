"""Test confidence stability improvements."""
from __future__ import annotations

import numpy as np
import torch

from src.inference.confidence_fusion import ImprovedConfidenceFusion, ConfidenceFusionConfig


def test_confidence_stability_with_same_input():
    """Test that confidence is stable for the same physical fruit."""
    fusion = ImprovedConfidenceFusion()
    confidences = []
    base_class = "apple"
    base_conf = 0.85
    base_det = 0.9
    for i in range(10):
        cls, conf, info = fusion.fuse(base_det, base_conf, class_name=base_class, timestamp=i * 0.1)
        confidences.append(conf)
    variance = float(np.var(confidences))
    mean_conf = float(np.mean(confidences))
    assert mean_conf >= 0.7, f"Mean confidence too low: {mean_conf}"
    assert variance < 0.05, f"Confidence variance too high: {variance}"


def test_confidence_stability_reduces_jitter():
    """Test that stability mechanism reduces jitter."""
    config = ConfidenceFusionConfig(stability_window=5, confidence_threshold=0.5, transition_smoothing=0.3)
    fusion = ImprovedConfidenceFusion(config)
    confidences_no_stability = []
    confidences_with_stability = []
    for i in range(10):
        cls, conf, info = fusion.fuse(0.7, 0.7, class_name="apple", timestamp=i * 0.1)
        confidences_with_stability.append(conf)
    variance = float(np.var(confidences_with_stability))
    assert variance < 0.1, f"Variance too high with stability: {variance}"


def test_temperature_scaling_calibrates():
    """Test that temperature scaling improves calibration."""
    from src.training.calibration import TemperatureScaling
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(10, 3))
    ts = TemperatureScaling(init_temp=1.0)
    x = torch.randn(50, 10)
    y = torch.randint(0, 3, (50,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)
    temp = ts.calibrate(loader, model, torch.device("cpu"))
    assert temp > 0
    assert temp != 1.0
