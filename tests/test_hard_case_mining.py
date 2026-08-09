"""Tests for Phase 4 hard-case mining module."""
import tempfile
from pathlib import Path

import pytest

from src.data.hard_case_mining import HardCase, HardCaseConfig, HardCaseMiner


class TestHardCaseConfig:
    """Tests for HardCaseConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = HardCaseConfig()
        assert config.low_confidence_threshold == 0.60
        assert config.high_confidence_error_threshold == 0.80
        assert config.quality_blur_threshold == 80.0
        assert config.output_dir == Path("reports")

    def test_custom_values(self):
        """Test custom configuration."""
        config = HardCaseConfig(
            low_confidence_threshold=0.70,
            output_dir=Path("custom_reports"),
        )
        assert config.low_confidence_threshold == 0.70
        assert config.output_dir == Path("custom_reports")


class TestHardCase:
    """Tests for HardCase dataclass."""

    def test_default_values(self):
        """Test default hard case values."""
        hc = HardCase(image_path="test.jpg", predicted_class="fresh")
        assert hc.image_path == "test.jpg"
        assert hc.predicted_class == "fresh"
        assert hc.confidence == 0.0
        assert hc.reasons == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hc = HardCase(
            image_path="test.jpg",
            predicted_class="fresh",
            confidence=0.45,
            reasons=["low_confidence", "near_decision_boundary"],
            metadata={"frame": 42},
        )
        result = hc.to_dict()
        assert result["image_path"] == "test.jpg"
        assert result["confidence"] == 0.45
        assert "low_confidence; near_decision_boundary" in result["reasons"]
        assert result["frame"] == 42
class TestHardCaseMiner:
    """Tests for HardCaseMiner."""

    def test_initialization(self):
        """Test miner initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HardCaseConfig(output_dir=Path(tmpdir))
            miner = HardCaseMiner(config)
            assert miner.config.output_dir == Path(tmpdir)

    def test_identify_low_confidence(self):
        """Test identification of low-confidence cases."""
        miner = HardCaseMiner(HardCaseConfig(low_confidence_threshold=0.60))
        diag = {"image_path": "test.jpg", "predicted_class": "fresh", "confidence": 0.45}
        reasons = miner._identify_reasons(diag)
        assert any("low_confidence_0.450" in r for r in reasons)

    def test_identify_high_confidence_error(self):
        """Test identification of high-confidence errors."""
        miner = HardCaseMiner(HardCaseConfig(high_confidence_error_threshold=0.80))
        diag = {
            "image_path": "test.jpg",
            "predicted_class": "fresh",
            "true_class": "rotten",
            "confidence": 0.95,
        }
        reasons = miner._identify_reasons(diag)
        assert any("high_confidence_error_0.950" in r for r in reasons)

    def test_identify_blurry(self):
        """Test identification of blurry images."""
        miner = HardCaseMiner(HardCaseConfig(quality_blur_threshold=100.0))
        diag = {
            "image_path": "test.jpg",
            "predicted_class": "fresh",
            "confidence": 0.9,
            "quality_metrics": {"blur_score": 50.0},
        }
        reasons = miner._identify_reasons(diag)
        assert any("blurry" in r for r in reasons)

    def test_identify_detector_disagreement(self):
        """Test identification of detector/classifier disagreement."""
        miner = HardCaseMiner()
        diag = {
            "image_path": "test.jpg",
            "predicted_class": "fresh",
            "confidence": 0.9,
            "detector_confidence": 0.4,
            "classifier_confidence": 0.9,
        }
        reasons = miner._identify_reasons(diag)
        assert "detector_classifier_disagreement" in reasons

    def test_identify_near_boundary(self):
        """Test identification of near-boundary predictions."""
        miner = HardCaseMiner()
        diag = {"image_path": "test.jpg", "predicted_class": "fresh", "confidence": 0.5}
        reasons = miner._identify_reasons(diag)
        assert "near_decision_boundary" in reasons

    def test_analyze_diagnostics(self):
        """Test analysis of diagnostic frames."""
        miner = HardCaseMiner(HardCaseConfig(low_confidence_threshold=0.80))
        diagnostics = [
            {"image_path": "test1.jpg", "predicted_class": "fresh", "confidence": 0.5},
            {"image_path": "test2.jpg", "predicted_class": "stale",
             "confidence": 0.95, "true_class": "fresh"},
        ]
        hard_cases = miner.analyze_diagnostics(diagnostics)
        assert len(hard_cases) == 2

    def test_save_reports(self):
        """Test saving hard-case reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            miner = HardCaseMiner(HardCaseConfig(output_dir=Path(tmpdir)))
            hard_cases = [
                HardCase(
                    image_path="test.jpg",
                    predicted_class="fresh",
                    confidence=0.45,
                    reasons=["low_confidence"],
                ),
            ]
            saved = miner.save_reports(hard_cases, prefix="test_hard_cases")
            assert "csv" in saved
            assert "json" in saved
            assert "summary" in saved
            assert saved["csv"].exists()