"""Unit and integration tests for Freshness + Shelf-Life Pipeline (Phase 13).

Tests:
    1. freshness model loading
    2. freshness inference
    3. supported fruit
    4. unsupported fruit
    5. freshness output contract
    6. shelf-life calculation
    7. invalid/boundary shelf-life values
    8. multi-fruit processing
    9. no detection frame handling
    10. complete pipeline integration
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.detection import BoundingBox, Detection
from src.inference.detection_pipeline import (
    DetectionPipeline,
    DetectionPipelineConfig,
    freshness_supported,
)
from src.inference.fruit_result import FruitResult, MultiFruitResult
from src.inference.predictor import PredictionResult, Predictor
from src.inference.shelf_life import ShelfLifeConfig, ShelfLifeEstimate, ShelfLifeEstimator


class TestFreshnessAndShelfLife:
    """Test suite for Phase 2 Freshness & Shelf-Life Productionization."""

    def test_1_freshness_model_loading(self):
        """Predictor should initialize and load EfficientNet weights when checkpoint exists."""
        mock_torch = MagicMock()
        mock_ckpt = {
            "model_state_dict": {},
            "class_names": ["freshapples", "rottenapples"],
            "classifier_type": "1280-256-6",
        }

        with patch("pathlib.Path.exists", return_value=True), \
             patch("src.inference.predictor.Predictor._load_checkpoint", return_value=mock_ckpt), \
             patch("src.inference.predictor.Predictor._build_model", return_value=MagicMock()):
            pred = Predictor("models/checkpoints/best_model.pth")
            assert pred.num_classes == 2
            assert pred.class_names == ["freshapples", "rottenapples"]

    def test_2_freshness_inference(self):
        """Predictor.predict() should return valid PredictionResult."""
        mock_ckpt = {
            "model_state_dict": {},
            "class_names": ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"],
        }
        with patch("pathlib.Path.exists", return_value=True), \
             patch("src.inference.predictor.Predictor._load_checkpoint", return_value=mock_ckpt), \
             patch("src.inference.predictor.Predictor._build_model") as mock_bm:
            mock_model = MagicMock()
            import torch
            mock_model.return_value = torch.tensor([[5.0, 0.0, 0.0, -1.0, -2.0, -3.0]])
            mock_bm.return_value = mock_model

            pred = Predictor("models/checkpoints/best_model.pth")
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            res = pred.predict(frame)

            assert isinstance(res, PredictionResult)
            assert res.fruit_name in ("apples", "apple")
            assert res.freshness_class == "fresh"
            assert res.confidence > 0.5

    def test_3_supported_fruit(self):
        """Fruits with validated freshness models must be recognized as supported."""
        assert freshness_supported("apple") is True
        assert freshness_supported("Apple") is True
        assert freshness_supported("banana") is True
        assert freshness_supported("orange") is True
        assert freshness_supported("grape") is True
        assert freshness_supported("guava") is True
        assert freshness_supported("strawberry") is True

    def test_4_unsupported_fruit(self):
        """Fruits WITHOUT validated fresh/rotten training data -> NOT supported.

        Kiwi, Mango, Cherry and Chickoo have no usable fresh/rotten training
        set, so they must NEVER produce a freshness guess. The detector can
        still detect them, but the pipeline reports ``data_not_available``.
        Jujube/Pomegranate have validated training data but are not emitted by
        the frozen detector, so the API also reports ``data_not_available``.
        """
        for fruit in ("kiwi", "mango", "cherry", "chickoo", "jujube", "pomegranate"):
            assert freshness_supported(fruit) is False, f"{fruit} must not be supported"
        assert freshness_supported("kiwi") is False

    def test_5_freshness_output_contract(self):
        """FruitResult must expose structured freshness fields."""
        det = Detection(label="Apple", confidence=0.90, bbox=BoundingBox(10, 10, 100, 100))
        stabilized = MagicMock(label="fresh", is_uncertain=False, confidence=0.88, ema_confidence=0.88, is_locked=False, lock_count=0, majority_label="fresh", vote_counts={"fresh": 1})
        res = FruitResult(
            detection=det,
            stabilized=stabilized,
            fused_confidence=0.89,
            freshness_class="fresh",
        )
        assert res.freshness_class == "fresh"
        assert res.fused_confidence == 0.89
        r_dict = res.to_dict()
        assert r_dict["freshness"] == "fresh"
        assert r_dict["confidence"] == 0.89

    def test_6_shelf_life_calculation(self):
        """ShelfLifeEstimator should calculate remaining days using metadata range and confidence."""
        estimator = ShelfLifeEstimator(ShelfLifeConfig())
        est = estimator.estimate(fruit="apple", fused_confidence=0.90, freshness_class="fresh")

        assert isinstance(est, ShelfLifeEstimate)
        assert est.fruit == "apple"
        assert est.remaining_days is not None
        assert est.remaining_days > 0
        assert est.basis == "fruit_typical_range + freshness_state + freshness_confidence"
        assert est.shelf_life_status == "estimated"

    def test_7_invalid_shelf_life_values(self):
        """Rotten or unavailable fruits must not produce invalid shelf-life days."""
        estimator = ShelfLifeEstimator(ShelfLifeConfig())
        
        # Rotten fruit -> 0 days (expired)
        rotten_est = estimator.estimate(fruit="apple", fused_confidence=0.90, freshness_class="rotten")
        assert rotten_est.remaining_days == 0
        assert rotten_est.shelf_life_status == "expired"

        # data_not_available fruit -> null days, data_not_available status (required contract)
        dna_est = estimator.estimate(fruit="mango", fused_confidence=0.0, freshness_class="data_not_available")
        assert dna_est.remaining_days is None
        assert dna_est.shelf_life_status == "data_not_available"

        # uncertain fruit -> null days, uncertain status
        unc_est = estimator.estimate(fruit="apple", fused_confidence=0.4, freshness_class="uncertain")
        assert unc_est.remaining_days is None
        assert unc_est.shelf_life_status == "uncertain"

    def test_8_multi_fruit_processing(self):
        """DetectionPipeline process_frame should maintain independent results for multiple fruits."""
        pipe = DetectionPipeline(DetectionPipelineConfig(detector_name="mock"))
        pipe.initialize()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det1 = Detection(label="apple", confidence=0.92, bbox=BoundingBox(10, 10, 100, 100), tracking_id=0)
        det2 = Detection(label="orange", confidence=0.88, bbox=BoundingBox(150, 150, 250, 250), tracking_id=1)
        pipe.detector._detections = [det1, det2]

        res = pipe.process_frame(frame)
        assert len(res.fruits) == 2
        assert res.fruits[0].detection.tracking_id == 0
        assert res.fruits[1].detection.tracking_id == 1
        pipe.shutdown()

    def test_9_no_detection(self):
        """Frame with no detections should return empty MultiFruitResult cleanly."""
        pipe = DetectionPipeline(DetectionPipelineConfig(detector_name="mock"))
        pipe.initialize()
        pipe.detector._detections = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipe.process_frame(frame)
        assert len(res.fruits) == 0
        assert res.unidentified_count == 0
        pipe.shutdown()

    def test_10_complete_pipeline_integration(self):
        """Pipeline integration end-to-end test."""
        pipe = DetectionPipeline(DetectionPipelineConfig(detector_name="mock"))
        pipe.initialize()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        det = Detection(label="banana", confidence=0.85, bbox=BoundingBox(50, 50, 150, 150))
        pipe.detector._detections = [det]

        res = pipe.process_frame(frame)
        assert len(res.fruits) == 1
        fruit = res.fruits[0]
        assert fruit.detection.label == "banana"
        assert fruit.shelf_life is not None
        pipe.shutdown()
