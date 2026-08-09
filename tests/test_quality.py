"""Tests for the Phase 3 image quality assessor (src/inference/quality.py)."""

import numpy as np
import pytest

from src.inference.quality import QualityAssessor, QualityConfig, QualityReport


class TestQualityConfig:
    def test_default_config(self):
        config = QualityConfig()
        assert config.brightness_min == 40
        assert config.brightness_max == 220
        assert config.blur_threshold == 100.0
        assert config.contrast_min == 20.0
        assert config.motion_threshold == 35.0

    def test_invalid_brightness_range(self):
        with pytest.raises(ValueError):
            QualityConfig(brightness_min=300)

    def test_inverted_brightness_range(self):
        with pytest.raises(ValueError):
            QualityConfig(brightness_min=200, brightness_max=100)

    def test_invalid_blur_threshold(self):
        with pytest.raises(ValueError):
            QualityConfig(blur_threshold=0)


class TestQualityAssessor:
    def make(self, **kw):
        config = QualityConfig(**kw)
        return QualityAssessor(config)

    def test_brightness_detection(self):
        assessor = self.make(brightness_min=50, brightness_max=200)

        # Bright frame
        bright_frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
        report = assessor.assess(bright_frame)
        assert report.is_brightness_ok is True

        # Dark frame
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        report = assessor.assess(dark_frame)
        assert report.is_brightness_ok is False
        assert "Lighting Too Low" in report.warnings

    def test_brightness_too_bright(self):
        assessor = self.make(brightness_min=40, brightness_max=150)

        bright_frame = np.ones((100, 100, 3), dtype=np.uint8) * 240
        report = assessor.assess(bright_frame)
        assert report.is_brightness_ok is False
        assert "Lighting Too Bright" in report.warnings

    def test_blur_detection(self):
        assessor = self.make(blur_threshold=100.0)

        # Sharp frame (random noise has high variance)
        sharp_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        report = assessor.assess(sharp_frame)
        # Random noise typically has high Laplacian variance
        assert report.blur_variance > 0

        # Blurry frame (uniform color has low variance)
        blurry_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        report = assessor.assess(blurry_frame)
        assert report.is_blur_ok is False
        assert "Image Blurry" in report.warnings

    def test_motion_detection(self):
        assessor = self.make(motion_threshold=10.0, use_motion_detection=True)

        # First frame - no motion possible
        frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        report1 = assessor.assess(frame1)
        assert report1.motion_detected is False

        # Second frame identical - no motion
        frame2 = frame1.copy()
        report2 = assessor.assess(frame2)
        assert report2.motion_detected is False
        assert report2.is_motion_ok is True

        # Third frame very different - motion detected
        frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 200
        report3 = assessor.assess(frame3)
        assert report3.motion_detected is True
        assert "Hold fruit still" in report3.warnings

    def test_motion_detection_disabled(self):
        assessor = self.make(use_motion_detection=False)

        frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 200

        report1 = assessor.assess(frame1)
        report2 = assessor.assess(frame2)

        assert report1.motion_detected is False
        assert report2.motion_detected is False

    def test_contrast_detection(self):
        assessor = self.make(contrast_min=30.0)

        # High contrast frame
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[:50, :] = 0
        high_contrast[50:, :] = 255
        report = assessor.assess(high_contrast)
        assert report.is_contrast_ok is True

        # Low contrast frame
        low_contrast = np.ones((100, 100, 3), dtype=np.uint8) * 128
        report = assessor.assess(low_contrast)
        assert report.is_contrast_ok is False
        assert "Low Contrast" in report.warnings

    def test_quality_score_calculation(self):
        assessor = self.make()

        # Good quality frame
        good_frame = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        report = assessor.assess(good_frame)
        assert 0.0 <= report.quality_score <= 1.0

    def test_reset_clears_previous_frame(self):
        assessor = self.make()

        frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        assessor.assess(frame1)

        assessor.reset()

        # After reset, next frame should not compare to previous
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 200
        report = assessor.assess(frame2)
        # No motion because previous was cleared
        assert report.motion_detected is False
