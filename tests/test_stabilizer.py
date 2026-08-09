"""Tests for the Phase 3 prediction stabilizer (src/inference/stabilizer.py)."""

import pytest

from src.inference.stabilizer import Stabilizer, StabilizerConfig, StabilizedPrediction

CLASS_NAMES = ["fresh", "stale", "rotten"]


class TestStabilizerConfig:
    def test_default_config(self):
        config = StabilizerConfig()
        assert config.ema_alpha == 0.2
        assert config.vote_window == 15
        assert config.lock_frames == 5
        assert config.confidence_threshold == 0.70

    def test_invalid_ema_alpha(self):
        with pytest.raises(ValueError):
            StabilizerConfig(ema_alpha=1.5)

    def test_invalid_vote_window(self):
        with pytest.raises(ValueError):
            StabilizerConfig(vote_window=0)

    def test_invalid_lock_frames(self):
        with pytest.raises(ValueError):
            StabilizerConfig(lock_frames=0)

    def test_invalid_confidence_threshold(self):
        with pytest.raises(ValueError):
            StabilizerConfig(confidence_threshold=1.5)


class TestStabilizer:
    def make(self, **kw):
        config = StabilizerConfig(**kw)
        return Stabilizer(config, CLASS_NAMES)

    def test_ema_smoothing(self):
        stabilizer = self.make(ema_alpha=0.2, vote_window=3, lock_frames=2)

        # First prediction initializes EMA
        r1 = stabilizer.update("fresh", 1.0)
        assert r1.ema_confidence == pytest.approx(1.0)

        # Second prediction with lower confidence
        r2 = stabilizer.update("fresh", 0.5)
        # EMA = 0.2 * 0.5 + 0.8 * 1.0 = 0.9
        assert r2.ema_confidence == pytest.approx(0.9, abs=0.01)

    def test_majority_voting(self):
        stabilizer = self.make(vote_window=5, lock_frames=2)

        # 3 fresh, 2 stale -> fresh wins
        for _ in range(3):
            stabilizer.update("fresh", 0.9)
        for _ in range(2):
            r = stabilizer.update("stale", 0.9)

        assert r.majority_label == "fresh"

    def test_prediction_locking_prevents_instant_switch(self):
        stabilizer = self.make(vote_window=15, lock_frames=3)

        # Start with fresh
        r1 = stabilizer.update("fresh", 0.9)
        assert r1.label == "fresh"
        assert not r1.is_locked

        # One conflicting prediction - should be locked
        r2 = stabilizer.update("rotten", 0.8)
        assert r2.label == "fresh"  # Still fresh due to lock
        assert r2.is_locked
        assert r2.lock_count == 1

        # Two more conflicting - still locked
        r3 = stabilizer.update("rotten", 0.8)
        assert r3.label == "fresh"
        assert r3.is_locked
        assert r3.lock_count == 2

        # Third conflicting - lock released, switch to rotten
        r4 = stabilizer.update("rotten", 0.8)
        assert r4.label == "rotten"
        assert not r4.is_locked
        assert r4.lock_count == 0

    def test_uncertain_when_confidence_low(self):
        stabilizer = self.make(
            ema_alpha=0.5,
            vote_window=3,
            lock_frames=2,
            confidence_threshold=0.70,
        )

        # Low confidence predictions
        r = stabilizer.update("fresh", 0.5)
        assert r.is_uncertain is True

    def test_certain_when_confidence_high(self):
        stabilizer = self.make(confidence_threshold=0.70)

        r = stabilizer.update("fresh", 0.95)
        assert r.is_uncertain is False

    def test_window_size_limit(self):
        stabilizer = self.make(vote_window=5)
        for _ in range(100):
            stabilizer.update("fresh", 0.9)
        stats = stabilizer.get_stats()
        assert stats["window_size"] == 5

    def test_reset_clears_state(self):
        stabilizer = self.make()
        stabilizer.update("fresh", 0.9)
        stabilizer.update("fresh", 0.9)
        stabilizer.reset()

        stats = stabilizer.get_stats()
        assert stats["window_size"] == 0
        assert stats["total_updates"] == 0

    def test_invalid_input_handling(self):
        stabilizer = self.make()

        # Invalid confidence
        r = stabilizer.update("fresh", 1.5)
        assert r.raw_confidence == 1.0  # Clamped

        # Empty label
        r = stabilizer.update("", 0.9)
        assert r.raw_label == "unknown"

    def test_get_stats(self):
        stabilizer = self.make()
        stabilizer.update("fresh", 0.9)
        stabilizer.update("stale", 0.8)

        stats = stabilizer.get_stats()
        assert stats["total_updates"] == 2
        assert stats["window_size"] == 2
        assert "vote_counts" in stats
