"""Tests for Phase 4 data collection module."""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.collection import CollectionConfig, QualityMetrics, RealWorldCollector


class TestCollectionConfig:
    """Tests for CollectionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CollectionConfig()
        assert config.blur_threshold == 100.0
        assert config.brightness_min == 40
        assert config.brightness_max == 220
        assert config.quality_checks is True
        assert config.image_size == (224, 224)

    def test_custom_values(self):
        """Test custom configuration."""
        config = CollectionConfig(
            blur_threshold=50.0,
            quality_checks=False,
        )
        assert config.blur_threshold == 50.0
        assert config.quality_checks is False


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_default_values(self):
        """Test default quality metrics."""

class TestRealWorldCollector:
    """Tests for RealWorldCollector."""

    def test_initialization(self):
        """Test collector initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CollectionConfig(output_dir=Path(tmpdir))
            collector = RealWorldCollector(config)
            assert collector.session_id is not None
            assert collector.frame_count == 0

    def test_directory_setup(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CollectionConfig(output_dir=Path(tmpdir))
            collector = RealWorldCollector(config)
            
            assert (Path(tmpdir) / "raw").exists()
            assert (Path(tmpdir) / "accepted").exists()
            assert (Path(tmpdir) / "rejected").exists()
            assert (Path(tmpdir) / "metadata").exists()

    def test_capture_creates_files(self):
        """Test that capture creates image and metadata files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CollectionConfig(
                output_dir=Path(tmpdir),
                quality_checks=False,
            )
            collector = RealWorldCollector(config)
            collector.start_session()
            
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            sample = collector.capture(
                frame=frame,
                label="fresh",
                predicted_class="freshapples",
                predicted_confidence=0.95,
            )
            
            assert sample.sample_id is not None
            assert sample.label == "fresh"
            assert sample.accepted is True
            assert Path(sample.image_path).exists()
            assert Path(sample.metadata_path).exists()

    def test_quality_check_rejection(self):
        """Test that quality checks reject poor images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CollectionConfig(
                output_dir=Path(tmpdir),
                quality_checks=True,
                blur_threshold=10000.0,
            )
            collector = RealWorldCollector(config)
            collector.start_session()
            
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            frame = cv2.GaussianBlur(frame, (51, 51), 0)
            
            sample = collector.capture(frame=frame, label="fresh")
            assert sample.accepted is False
            assert sample.rejection_reason is not None

    def test_session_statistics(self):
        """Test session statistics tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CollectionConfig(
                output_dir=Path(tmpdir),
                quality_checks=False,
            )
            collector = RealWorldCollector(config)
            collector.start_session()
            
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            collector.capture(frame=frame, label="fresh")
            collector.capture(frame=frame, label="fresh")
            collector.capture(frame=frame, label="stale")
            
            stats = collector.end_session()
            assert stats.total_captured == 3
            assert stats.total_accepted == 3
            assert stats.per_class_counts["fresh"] == 2
            assert stats.per_class_counts["stale"] == 1

        qm = QualityMetrics()
        assert qm.blur_score == 0.0
        assert qm.brightness == 0.0
        assert qm.resolution == (0, 0)

    def test_custom_values(self):
        """Test custom quality metrics."""
        qm = QualityMetrics(
            blur_score=150.5,
            resolution=(640, 480),
        )
        assert qm.blur_score == 150.5
        assert qm.resolution == (640, 480)
