"""Tests for the FPS monitor (src/inference/fps.py)."""

import time

from src.inference.fps import FPSConfig, FPSMonitor


class TestFPSMonitor:
    def test_config_validation(self):
        assert FPSConfig(window_size=10).window_size == 10
        for kwargs in ({"window_size": 0}, {"update_interval": 0}, {"log_interval": 0}):
            try:
                FPSConfig(**kwargs)
                assert False, f"should have raised for {kwargs}"
            except ValueError:
                pass

    def test_initial_stats(self):
        m = FPSMonitor(FPSConfig())
        s = m.get_stats()
        assert s.total_frames == 0
        assert s.current_fps == 0.0

    def test_frames_tracked(self):
        m = FPSMonitor(FPSConfig())
        for _ in range(50):
            m.start_frame()
            time.sleep(0.001)
            m.end_frame(inference_complete=True)
        s = m.get_stats()
        assert s.total_frames == 50
        assert s.average_fps > 0.0

    def test_latency_recorded(self):
        m = FPSMonitor(FPSConfig())
        m.start_frame()
        time.sleep(0.005)
        m.end_frame(inference_complete=True)
        # Latency is measured internally as end - start.
        assert m.get_stats().avg_latency_ms > 0.0

    def test_reset(self):
        m = FPSMonitor(FPSConfig())
        m.start_frame()
        m.end_frame(inference_complete=True)
        m.reset()
        assert m.get_stats().total_frames == 0

    def test_no_growth_without_window(self):
        # _frame_times list is capped by window_size -> no memory growth.
        m = FPSMonitor(FPSConfig(window_size=30))
        for _ in range(500):
            m.start_frame()
            m.end_frame(inference_complete=True)
        assert len(m._frame_times) <= 30
