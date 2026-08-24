"""Tests for the predictor (src/inference/predictor.py).

These include both fast unit tests (class-name parsing, error handling) and
integration-style tests that load the real trained checkpoint on CPU.
"""

import numpy as np
import pytest
import torch

from src.inference.predictor import Predictor
from tests.conftest import CHECKPOINT_PATH, FRESHNESS_CHECKPOINT_PATH


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

    def test_16class_taxonomy_suffix(self):
        """The 16-class format is 'Fruit_fresh' / 'Fruit_rotten' (suffix)."""
        p = _bare_predictor()
        assert p._parse_class_name("Apple_fresh") == ("Apple", "fresh")
        assert p._parse_class_name("Apple_rotten") == ("Apple", "rotten")
        assert p._parse_class_name("banana_fresh") == ("banana", "fresh")
        assert p._parse_class_name("banana_rotten") == ("banana", "rotten")
        assert p._parse_class_name("Grape_fresh") == ("Grape", "fresh")
        assert p._parse_class_name("guava_rotten") == ("guava", "rotten")
        assert p._parse_class_name("Jujube_fresh") == ("Jujube", "fresh")
        assert p._parse_class_name("Orange_rotten") == ("Orange", "rotten")
        assert p._parse_class_name("Pomegranate_fresh") == ("Pomegranate", "fresh")
        assert p._parse_class_name("Strawberry_rotten") == ("Strawberry", "rotten")

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

    def test_16class_parse_no_unknown(self):
        """Every canonical class name must parse to a real fruit+state pair."""
        import json
        from pathlib import Path
        from tests.conftest import REPO_ROOT
        cm = json.loads((REPO_ROOT / "data" / "freshness" / "class_mapping.json").read_text())
        ordered = [None] * len(cm)
        for k, v in cm.items():
            ordered[int(k)] = v
        p = _bare_predictor()
        for c in ordered:
            fruit, state = p._parse_class_name(c)
            assert state in ("fresh", "rotten", "stale"), f"{c} -> {state}"
        assert fruit, f"{c} produced empty fruit"

    @pytest.mark.skipif(
        not FRESHNESS_CHECKPOINT_PATH.exists(),
        reason="16-class checkpoint not available",
    )
    def test_16class_loads_and_predicts(self):
        """The 16-class checkpoint should load and produce valid 16-class output."""
        from tests.conftest import FRESHNESS_CHECKPOINT_PATH
        pred = Predictor(str(FRESHNESS_CHECKPOINT_PATH))
        assert pred.num_classes == 16
        assert len(pred.class_names) == 16
        # Verify the taxonomy matches the canonical 16 classes
        assert "Apple_fresh" in pred.class_names
        assert "Apple_rotten" in pred.class_names
        assert "banana_fresh" in pred.class_names
        assert "Strawberry_rotten" in pred.class_names
        # Uncertainty threshold should be set (calibrated from validation)
        assert pred.uncertainty_threshold > 0.0
        assert pred.uncertainty_threshold <= 1.0

        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = pred.predict(frame)

        assert result.fruit_name
        assert result.freshness_class in ("fresh", "rotten", "uncertain", "unknown")
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 16
        assert result.latency_ms >= 0.0
        # The uncertainty flag must be consistent with the calibrated policy:
        # uncertain exactly when top-1 confidence < threshold. Random noise may
        # legitimately be uncertain — that is the desired conservative behavior.
        assert result.is_uncertain == (
            result.confidence < pred.uncertainty_threshold
        )
        if result.is_uncertain:
            assert result.freshness_class == "uncertain"
        assert result.uncertainty_threshold > 0.0
        # Diagnostic fields must be populated
        assert result.raw_logits is not None
        assert result.top2_probabilities is not None
        assert len(result.top2_probabilities) == 2
        assert result.predicted_class_index is not None



def _built_class_mapping():
    import json
    from pathlib import Path
    from tests.conftest import REPO_ROOT
    cm = json.loads((REPO_ROOT / "data" / "freshness" / "class_mapping.json").read_text())
    ordered = [None] * len(cm)
    for k, v in cm.items():
        ordered[int(k)] = v
    return [c for c in ordered if c]
