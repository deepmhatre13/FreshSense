"""Tests for the prediction tracker (src/inference/tracker.py)."""

import pytest

from src.inference.tracker import PredictionTracker, TrackerConfig

CLASS_NAMES = ["fresh", "stale", "rotten"]


class TestTracker:
    def make(self, **kw):
        return PredictionTracker(TrackerConfig(**kw), CLASS_NAMES)

    def test_majority_vote_smooths_flicker(self):
        tracker = self.make(window_size=5)
        labels = ["fresh"] * 4 + ["stale"] * 1  # stale is a flicker
        results = [tracker.update(l, 0.9) for l in labels]
        assert results[-1].label == "fresh"

    def test_window_limited_memory(self):
        tracker = self.make(window_size=5)
        for _ in range(200):
            tracker.update("fresh", 0.9)
        assert len(tracker._window) == 5  # deque maxlen enforced -> no leak

    def test_ema_confidence(self):
        tracker = self.make(window_size=3, smoothing_factor=0.3)
        tracker.update("fresh", 1.0)
        r = tracker.update("fresh", 1.0)
        # EMA should stay close to 1.0
        assert r.confidence > 0.9

    def test_consistency(self):
        tracker = self.make(window_size=5)
        tracker.update("fresh", 0.9)
        tracker.update("fresh", 0.9)
        r = tracker.update("stale", 0.9)  # 2 fresh, 1 stale
        assert r.consistency == pytest.approx(2 / 3)

    def test_stability_detection(self):
        tracker = self.make(window_size=5, confidence_threshold=0.5, min_consistency=0.6)
        for _ in range(4):
            tracker.update("fresh", 0.9)
        r = tracker.update("fresh", 0.9)  # all fresh, high conf -> stable
        assert r.is_stable is True

    def test_reset(self):
        tracker = self.make(window_size=3)
        tracker.update("fresh", 0.9)
        tracker.reset()
        assert len(tracker._window) == 0
        assert tracker.get_last_prediction() is None

    def test_confidence_threshold_blocks_stability(self):
        tracker = self.make(window_size=5, confidence_threshold=0.8, min_consistency=0.6)
        for _ in range(5):
            tracker.update("fresh", 0.5)  # consistent but low confidence
        assert tracker.get_last_prediction()[1] < 0.8


def pytest_approxim():
    import pytest as _p

    return _p
