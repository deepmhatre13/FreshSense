"""Tests for the Phase 3 session statistics (src/inference/statistics.py)."""

import json
import time

import pytest

from src.inference.statistics import SessionStatistics, SessionLogger


class TestSessionStatistics:
    def test_initial_state(self):
        stats = SessionStatistics()
        assert stats.total_frames == 0
        assert stats.total_predictions == 0
        assert stats.uncertain_frames == 0
        assert stats.motion_skips == 0
        assert stats.lighting_warnings == 0

    def test_record_prediction(self):
        stats = SessionStatistics()
        stats.record_prediction(confidence=0.85, latency_ms=20.0, fps=28.0)

        assert stats.total_predictions == 1
        assert len(stats.confidences) == 1
        assert stats.confidences[0] == 0.85
        assert len(stats.latency_values) == 1
        assert len(stats.fps_values) == 1

    def test_record_uncertain(self):
        stats = SessionStatistics()
        stats.record_uncertain()
        assert stats.uncertain_frames == 1

    def test_record_motion_skip(self):
        stats = SessionStatistics()
        stats.record_motion_skip()
        assert stats.motion_skips == 1

    def test_record_lighting_warning(self):
        stats = SessionStatistics()
        stats.record_lighting_warning()
        assert stats.lighting_warnings == 1

    def test_record_quality(self):
        stats = SessionStatistics()
        stats.record_quality(brightness=128.0, blur_variance=150.0)
        assert len(stats.brightness_values) == 1
        assert len(stats.blur_values) == 1

    def test_average_confidence(self):
        stats = SessionStatistics()
        stats.record_prediction(0.8, 20.0, 30.0)
        stats.record_prediction(0.9, 20.0, 30.0)
        stats.record_prediction(0.7, 20.0, 30.0)

        assert stats.average_confidence == pytest.approx(0.8)

    def test_average_fps(self):
        stats = SessionStatistics()
        stats.record_prediction(0.9, 20.0, 28.0)
        stats.record_prediction(0.9, 20.0, 32.0)

        assert stats.average_fps == pytest.approx(30.0)

    def test_average_latency(self):
        stats = SessionStatistics()
        stats.record_prediction(0.9, 20.0, 30.0)
        stats.record_prediction(0.9, 30.0, 30.0)

        assert stats.average_latency == pytest.approx(25.0)

    def test_elapsed_time(self):
        stats = SessionStatistics()
        time.sleep(0.01)
        elapsed = stats.elapsed_time
        assert elapsed >= 0.01

    def test_finalize_sets_end_time(self):
        stats = SessionStatistics()
        time.sleep(0.01)
        stats.finalize()
        assert stats.end_time > stats.start_time

    def test_to_dict(self):
        stats = SessionStatistics()
        stats.record_prediction(0.85, 20.0, 30.0)
        stats.record_quality(120.0, 150.0)
        stats.finalize()

        d = stats.to_dict()
        assert "session_duration_seconds" in d
        assert "total_predictions" in d
        assert "average_confidence" in d
        assert d["total_predictions"] == 1


class TestSessionLogger:
    def test_default_init(self, tmp_path):
        logger = SessionLogger(log_dir=str(tmp_path), save_logs=True)
        assert logger.session_id is not None
        assert logger.csv_path is not None
        logger.close()

    def test_log_frame(self, tmp_path):
        logger = SessionLogger(log_dir=str(tmp_path), save_logs=True)

        logger.log_frame(
            prediction="fresh",
            confidence=0.95,
            fps=29.0,
            latency_ms=18.0,
            brightness=120.0,
            blur_variance=200.0,
            warnings=[],
        )

        logger.close()

        # Verify CSV was written
        assert logger.csv_path.exists()
        with open(logger.csv_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2  # Header + 1 data row

    def test_log_frame_with_warnings(self, tmp_path):
        logger = SessionLogger(log_dir=str(tmp_path), save_logs=True)

        logger.log_frame(
            prediction="fresh",
            confidence=0.85,
            fps=29.0,
            latency_ms=18.0,
            brightness=120.0,
            blur_variance=200.0,
            warnings=["Lighting Too Low", "Image Blurry"],
        )

        logger.close()

        with open(logger.csv_path, "r") as f:
            content = f.read()
        assert "Lighting Too Low" in content
        assert "Image Blurry" in content

    def test_save_summary(self, tmp_path):
        logger = SessionLogger(log_dir=str(tmp_path), save_logs=True)

        stats = SessionStatistics()
        stats.record_prediction(0.9, 20.0, 30.0)
        stats.finalize()

        summary_path = logger.save_summary(stats)
        assert summary_path is not None
        assert summary_path.exists()

        with open(summary_path, "r") as f:
            summary = json.load(f)

        assert summary["total_predictions"] == 1
        assert "session_id" in summary

    def test_disabled_logging(self, tmp_path):
        logger = SessionLogger(log_dir=str(tmp_path), save_logs=False)
        assert logger.csv_path is None

        logger.log_frame(
            prediction="fresh",
            confidence=0.9,
            fps=30.0,
            latency_ms=15.0,
            brightness=120.0,
            blur_variance=200.0,
        )

        # No files should be created
        files = list(tmp_path.glob("*"))
        assert len(files) == 0

        logger.close()
