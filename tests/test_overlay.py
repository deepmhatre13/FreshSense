"""Tests for the overlay renderer (src/inference/overlay.py)."""

import numpy as np

from src.inference.overlay import ColorScheme, Overlay, OverlayConfig


def make_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestOverlayColor:
    def setup_method(self):
        self.overlay = Overlay(OverlayConfig(), ColorScheme())

    def test_fresh_color(self):
        assert self.overlay.get_class_color("freshapples") == ColorScheme().fresh
        assert self.overlay.get_class_color("fresh") == ColorScheme().fresh

    def test_rotten_color(self):
        assert self.overlay.get_class_color("rottenbanana") == ColorScheme().rotten

    def test_stale_color(self):
        assert self.overlay.get_class_color("staleoranges") == ColorScheme().stale

    def test_unknown_color(self):
        assert self.overlay.get_class_color("mystery") == ColorScheme().unknown

    def test_case_insensitive(self):
        assert self.overlay.get_class_color("FreshApples") == ColorScheme().fresh


class TestOverlayDrawing:
    def setup_method(self):
        self.overlay = Overlay(OverlayConfig(), ColorScheme())

    def test_draw_prediction_keeps_shape(self):
        frame = make_frame()
        out = self.overlay.draw_prediction(frame, "freshapples", 0.95, (0, 255, 0))
        assert out.shape == frame.shape
        assert out.dtype == frame.dtype

    def test_draw_prediction_with_box(self):
        frame = make_frame()
        out = self.overlay.draw_prediction(
            frame, "rottenbanana", 0.8, (0, 0, 255), box=(50, 50, 300, 400)
        )
        assert out.shape == frame.shape

    def test_draw_performance_stats(self):
        frame = make_frame()
        out = self.overlay.draw_performance_stats(frame, 28.5, 35.2, "CPU")
        assert out.shape == frame.shape

    def test_draw_confidence_bar(self):
        frame = make_frame()
        out = self.overlay.draw_confidence_bar(frame, 0.95, (0, 255, 0), "top-right")
        assert out.shape == frame.shape

    def test_draw_confidence_bar_boundary(self):
        frame = make_frame()
        for p in ("top-right", "top-left", "bottom-right", "bottom-left"):
            out = self.overlay.draw_confidence_bar(frame, 0.5, (0, 255, 0), p)
            assert out.shape == frame.shape

    def test_config_validation(self):
        for kwargs in ({"font_scale": 0}, {"alpha": 1.5}, {"confidence_threshold": -1}):
            try:
                OverlayConfig(**kwargs)
                assert False, f"should raise for {kwargs}"
            except ValueError:
                pass
