"""Tests for the predictor (src/inference/predictor.py).

These include both fast unit tests (class-name parsing, error handling) and
integration-style tests that load the real trained checkpoint on CPU.
"""

import numpy as np
import pytest
import torch

from src.inference.predictor import Predictor
from tests.conftest import CHECKPOINT_PATH


def _bare_predictor() -> Predictor:
    """Build a Predictor without running __init__ (for method-only tests)."""
    return Predictor.__new__(Predictor)


class TestClassParsing:
    """The checkpoint stores class names like 'freshapples' / 'rottenbanana'."""

    def test_no_underscore(self):
        p = _bare_predictor()
        assert p._parse_class_name("freshapples") == ("apples", "fresh")
        assert p._parse_class_name("rottenbanana") == ("banana", "rotten")
        assert p._parse_class_name("staleoranges") == ("oranges", "stale")

    def test_underscore(self):
        p = _bare_predictor()
        assert p._parse_class_name("fresh_apple") == ("apple", "fresh")

    def test_plain_freshness(self):
        p = _bare_predictor()
        assert p._parse_class_name("fresh") == ("fresh", "fresh")

    def test_unknown(self):
        p = _bare_predictor()
        assert p._parse_class_name("mystery") == ("mystery", "unknown")


class TestErrorHandling:
    def test_missing_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Predictor(str(tmp_path / "does_not_exist.pth"))

    def test_invalid_checkpoint_missing_state_dict(self, tmp_path):
        path = tmp_path / "bad.pth"
        torch.save({"class_names": ["a", "b"]}, path)
        with pytest.raises(RuntimeError):
            Predictor(str(path))

    def test_wrong_architecture(self, tmp_path):
        path = tmp_path / "wrong.pth"
        torch.save(
            {
                "class_names": ["freshapple", "rottenapple"],
                "model_state_dict": {},
                "classifier_type": "1280-256-999",
            },
            path,
        )
        with pytest.raises(RuntimeError):
            Predictor(str(path))


class TestRealCheckpoint:
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="checkpoint not available")
    def test_loads_and_predicts(self):
        pred = Predictor(str(CHECKPOINT_PATH))
        assert pred.num_classes == 6
        assert len(pred.class_names) == 6
        assert pred.device.type in ("cpu", "cuda", "mps")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pred.predict(frame)

        assert result.fruit_name
        assert result.freshness_class in ("fresh", "stale", "rotten", "unknown")
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 6
        assert result.latency_ms >= 0.0
        assert result.device
        assert result.model_version
