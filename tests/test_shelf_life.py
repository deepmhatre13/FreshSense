import json
from pathlib import Path
import pytest

from src.inference.fruit_metadata import FruitMetadataDatabase
from src.inference.shelf_life import ShelfLifeConfig, ShelfLifeEstimator

def test_shelf_life_with_metadata(tmp_path):
    db_file = tmp_path / "fruit_database.json"
    data = {
        "apple": {
            "typical_shelf_life_days": [14, 30]
        }
    }
    db_file.write_text(json.dumps(data))
    
    db = FruitMetadataDatabase(str(db_file))
    estimator = ShelfLifeEstimator(ShelfLifeConfig(), db)
    
    # Test valid fruit, fresh
    result = estimator.estimate("apple", fused_confidence=1.0, is_fresh=True)
    assert result.shelf_life_status == "estimated"
    assert result.remaining_days == 30
    assert result.typical_min_days == 14
    assert result.typical_max_days == 30
    
    # Test valid fruit, rotten
    result_rotten = estimator.estimate("apple", fused_confidence=1.0, is_fresh=False)
    assert result_rotten.shelf_life_status == "expired"
    assert result_rotten.remaining_days == 0
    assert result_rotten.typical_min_days == 14
    assert result_rotten.typical_max_days == 30

    # Test valid fruit, unsupported freshness
    result_unsupported = estimator.estimate("apple", fused_confidence=1.0, freshness_class="unsupported")
    assert result_unsupported.shelf_life_status == "unsupported"
    assert result_unsupported.remaining_days is None
    assert result_unsupported.typical_min_days == 14
    assert result_unsupported.typical_max_days == 30

def test_shelf_life_without_file(tmp_path):
    db_file = tmp_path / "missing_db.json"
    
    db = FruitMetadataDatabase(str(db_file))
    estimator = ShelfLifeEstimator(ShelfLifeConfig(), db)
    
    result = estimator.estimate("apple", fused_confidence=1.0, is_fresh=True)
    assert result.shelf_life_status == "unsupported"
    assert result.remaining_days is None
    assert result.typical_min_days is None
    assert result.typical_max_days is None

def test_shelf_life_missing_fruit_entry(tmp_path):
    db_file = tmp_path / "fruit_database.json"
    data = {
        "apple": {
            "typical_shelf_life_days": [14, 30]
        }
    }
    db_file.write_text(json.dumps(data))
    
    db = FruitMetadataDatabase(str(db_file))
    estimator = ShelfLifeEstimator(ShelfLifeConfig(), db)
    
    result = estimator.estimate("grape", fused_confidence=1.0, is_fresh=True)
    assert result.shelf_life_status == "unsupported"
    assert result.remaining_days is None
    assert result.typical_min_days is None
    assert result.typical_max_days is None

def test_shelf_life_confidence_scaling(tmp_path):
    db_file = tmp_path / "fruit_database.json"
    data = {
        "apple": {
            "typical_shelf_life_days": [14, 30]
        }
    }
    db_file.write_text(json.dumps(data))
    
    db = FruitMetadataDatabase(str(db_file))
    estimator = ShelfLifeEstimator(ShelfLifeConfig(), db)
    
    # Low confidence scaling
    result_low = estimator.estimate("apple", fused_confidence=0.5, freshness_class="fresh")
    assert result_low.shelf_life_status == "estimated"
    assert result_low.remaining_days == 15
    assert result_low.typical_min_days == 14
    assert result_low.typical_max_days == 30

def test_shelf_life_bounds(tmp_path):
    db_file = tmp_path / "fruit_database.json"
    data = {
        "apple": {
            "typical_shelf_life_days": [14, 30]
        }
    }
    db_file.write_text(json.dumps(data))
    
    db = FruitMetadataDatabase(str(db_file))
    estimator = ShelfLifeEstimator(ShelfLifeConfig(), db)
    
    # 0 confidence
    result_0 = estimator.estimate("apple", fused_confidence=0.0, freshness_class="fresh")
    assert result_0.remaining_days == 1  # Should clamp to > 0 if fresh
    
    # > 1.0 confidence
    result_high = estimator.estimate("apple", fused_confidence=1.5, freshness_class="fresh")
    assert result_high.remaining_days == 30  # Should clamp max days


# ============================================================================
# Production hardening suite (Phases 3-12, 17)
# ============================================================================

from src.inference.shelf_life import (
    ALLOWED_STORAGE_CONDITIONS,
    BASIS_HEURISTIC,
    ShelfLifeConfig,
    ShelfLifeEstimator,
    sanitize_confidence,
)

REAL_DB = FruitMetadataDatabase()

EXPECTED_RANGES = {
    "apple": (14, 30),
    "grape": (7, 14),
    "kiwi": (7, 21),
    "mango": (5, 14),
    "orange": (14, 28),
    "strawberry": (3, 7),
    "banana": (3, 7),
    "cherry": (4, 10),
    "chickoo": (3, 7),
    "guava": (2, 7),
}


@pytest.mark.parametrize("fruit,expected", sorted(EXPECTED_RANGES.items()))
def test_all_ten_fruits_have_valid_metadata(fruit, expected):
    """All 10 detected fruits must have complete, valid metadata."""
    assert REAL_DB.metadata_available is True
    assert fruit in REAL_DB.names()
    meta = REAL_DB.get(fruit)
    assert meta is not None
    lo, hi = meta.typical_shelf_life_days
    assert (lo, hi) == expected
    assert meta.scientific_name, f"{fruit} missing scientific_name"
    assert meta.optimal_storage, f"{fruit} missing optimal_storage"
    assert REAL_DB.validation_issues == []


class TestFreshnessStateMatrix:
    """Deterministic behavior for every freshness state (Phase 6)."""

    def setup_method(self):
        self.est = ShelfLifeEstimator(ShelfLifeConfig(), REAL_DB)

    def test_fresh_estimates(self):
        for fruit, (_, tmax) in EXPECTED_RANGES.items():
            r = self.est.estimate(fruit, 0.88, freshness_class="fresh")
            assert r.shelf_life_status == "estimated"
            assert r.remaining_days == max(1, min(tmax, int(round(tmax * 0.88))))

    def test_rotten(self):
        r = self.est.estimate("apple", 0.9, freshness_class="rotten")
        assert r.shelf_life_status == "expired" and r.remaining_days == 0

    def test_stale(self):
        r = self.est.estimate("apple", 0.9, freshness_class="stale")
        assert r.shelf_life_status == "expired" and r.remaining_days == 0
        assert "stale" in r.explanation.lower()

    def test_unknown(self):
        r = self.est.estimate("kiwi", 0.5, freshness_class="unknown")
        assert r.shelf_life_status == "unknown" and r.remaining_days is None

    def test_uncertain(self):
        r = self.est.estimate("mango", 0.4, freshness_class="uncertain")
        assert r.shelf_life_status == "uncertain" and r.remaining_days is None

    def test_unsupported(self):
        r = self.est.estimate("guava", 0.8, freshness_class="unsupported")
        assert r.shelf_life_status == "unsupported" and r.remaining_days is None
        assert "unsupported" in r.basis

    def test_legacy_is_fresh_false_maps_to_rotten(self):
        r = self.est.estimate("apple", 1.0, is_fresh=False)
        assert r.shelf_life_status == "expired" and r.remaining_days == 0

    def test_state_dominates_malformed_confidence(self):
        """Rotten stays expired even when confidence is garbage."""
        for bad in (None, float("nan"), float("inf"), "oops"):
            r = self.est.estimate("apple", bad, freshness_class="rotten")
            assert r.shelf_life_status == "expired" and r.remaining_days == 0


class TestConfidenceBoundaries:
    """Phase 7/8: bounds, malformed values, semantic honesty."""

    def setup_method(self):
        self.est = ShelfLifeEstimator(ShelfLifeConfig(), REAL_DB)

    @pytest.mark.parametrize("conf,expected", [(0.0, 1), (0.5, 15), (1.0, 30)])
    def test_valid_confidence_scaling(self, conf, expected):
        r = self.est.estimate("apple", conf, freshness_class="fresh")
        assert r.remaining_days == expected
        assert r.basis == BASIS_HEURISTIC

    @pytest.mark.parametrize("bad", [None, True, False, float("nan"),
                                     float("inf"), -float("inf"), "0.9", "", [1]])
    def test_malformed_confidence_never_estimates(self, bad):
        r = self.est.estimate("apple", bad, freshness_class="fresh")
        assert r.shelf_life_status == "uncertain"
        assert r.remaining_days is None
        assert r.freshness_confidence is None
        assert r.basis == "freshness_confidence_unusable"

    def test_out_of_range_clamped(self):
        assert self.est.estimate("apple", -2.5, freshness_class="fresh").remaining_days == 1
        assert self.est.estimate("apple", 99, freshness_class="fresh").remaining_days == 30

    def test_sanitize_helper(self):
        import math
        assert sanitize_confidence(None) is None
        assert sanitize_confidence(True) is None
        assert sanitize_confidence(float("nan")) is None
        assert sanitize_confidence(float("inf")) is None
        assert sanitize_confidence("x") is None
        assert sanitize_confidence(-1) == 0.0
        assert sanitize_confidence(2) == 1.0
        assert sanitize_confidence(0.42) == 0.42

    def test_estimate_is_not_probability_claim(self):
        """Documented heuristic: days scale linearly with confidence."""
        lo = self.est.estimate("apple", 0.5, freshness_class="fresh").remaining_days
        hi = self.est.estimate("apple", 0.9, freshness_class="fresh").remaining_days
        assert hi > lo



class TestStorageCondition:
    """Phase 4/5/10: explicit, strictly validated, context-only."""

    def setup_method(self):
        self.est = ShelfLifeEstimator(ShelfLifeConfig(), REAL_DB)

    @pytest.mark.parametrize("condition", ["ambient", "refrigerated"])
    def test_supported_conditions_recorded(self, condition):
        r = self.est.estimate("apple", 0.88, freshness_class="fresh",
                              storage_condition=condition)
        assert r.storage_condition == condition
        assert f"assumes {condition} storage" in r.explanation

    def test_case_and_whitespace_normalized(self):
        r = self.est.estimate("apple", 0.9, freshness_class="fresh",
                              storage_condition="  REFRIGERATED ")
        assert r.storage_condition == "refrigerated"

    @pytest.mark.parametrize("bad", ["freezer", "Fridge", "", "ambient hot", 123])
    def test_invalid_condition_raises(self, bad):
        with pytest.raises(ValueError):
            self.est.estimate("apple", 0.9, freshness_class="fresh",
                              storage_condition=bad)

    def test_condition_is_context_only(self):
        """No condition-specific durations exist in the database, so the
        numeric estimate must NOT change between conditions (honesty)."""
        a = self.est.estimate("mango", 0.8, freshness_class="fresh",
                              storage_condition="ambient")
        c = self.est.estimate("mango", 0.8, freshness_class="fresh",
                              storage_condition="refrigerated")
        assert a.remaining_days == c.remaining_days
        assert ALLOWED_STORAGE_CONDITIONS == ("ambient", "refrigerated")

    def test_invalid_config_rejected_eagerly(self):
        with pytest.raises(ValueError):
            ShelfLifeConfig(default_storage_condition="freezer")


class TestRangeSemantics:
    """Phase 9: typical ranges are typicals, never guarantees."""

    def setup_method(self):
        self.est = ShelfLifeEstimator(ShelfLifeConfig(), REAL_DB)

    @pytest.mark.parametrize("conf", [0.0, 0.25, 0.5, 0.75, 1.0, -1, 5])
    def test_never_exceeds_typical_max_and_never_below_one(self, conf):
        for fruit, (_, tmax) in EXPECTED_RANGES.items():
            r = self.est.estimate(fruit, conf, freshness_class="fresh")
            assert 1 <= r.remaining_days <= tmax
            assert isinstance(r.remaining_days, int)

    def test_expired_only_state_with_zero(self):
        for cls in ("rotten", "stale"):
            r = self.est.estimate("apple", 0.7, freshness_class=cls)
            assert r.remaining_days == 0
        for cls in ("unknown", "uncertain", "unsupported"):
            r = self.est.estimate("apple", 0.7, freshness_class=cls)
            assert r.remaining_days is None



class TestMetadataFailureModes:
    """Phases 3/11/12: no fabricated defaults, ever."""

    @staticmethod
    def _db(tmp_path, content):
        path = tmp_path / "fruit_database.json"
        if content is not None:
            path.write_text(json.dumps(content))
        return FruitMetadataDatabase(str(path))

    def test_missing_database_file_is_safe(self, tmp_path):
        db = FruitMetadataDatabase(str(tmp_path / "missing.json"))
        est = ShelfLifeEstimator(ShelfLifeConfig(), db)
        r = est.estimate("apple", 0.9, freshness_class="fresh")
        assert r.shelf_life_status == "unsupported"
        assert r.remaining_days is None
        assert r.typical_min_days is None and r.typical_max_days is None
        assert db.metadata_available is False
        assert db.names() == []

    def test_malformed_database_json_is_safe(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json!!")
        db = FruitMetadataDatabase(str(path))
        est = ShelfLifeEstimator(ShelfLifeConfig(), db)
        r = est.estimate("apple", 0.9, freshness_class="fresh")
        assert r.shelf_life_status == "unsupported" and r.remaining_days is None

    def test_non_object_database_is_safe(self, tmp_path):
        db = self._db(tmp_path, ["apple"])
        assert db.metadata_available is False
        assert "top-level JSON" in db.validation_issues[0]

    def test_missing_range_not_fabricated(self, tmp_path):
        db = self._db(tmp_path, {"apple": {"scientific_name": "Malus domestica"}})
        est = ShelfLifeEstimator(ShelfLifeConfig(), db)
        r = est.estimate("apple", 0.9, freshness_class="fresh")
        assert r.shelf_life_status == "unsupported"
        assert r.basis == "metadata_invalid"
        assert r.typical_min_days is None and r.typical_max_days is None
        assert any("apple" in i for i in db.validation_issues)

    @pytest.mark.parametrize("bad_range", [
        [], [14], [14, 30, 45], [30, 14], [-5, 10], [0, 0],
        ["a", "b"], [3.5, 9], 30, "14-30", True,
    ])
    def test_invalid_ranges_explicitly_unsupported(self, tmp_path, bad_range):
        db = self._db(tmp_path, {"apple": {"typical_shelf_life_days": bad_range}})
        est = ShelfLifeEstimator(ShelfLifeConfig(), db)
        r = est.estimate("apple", 0.9, freshness_class="fresh")
        assert r.shelf_life_status == "unsupported"
        assert r.remaining_days is None
        assert r.basis == "metadata_invalid"


class TestSerialization:
    """Phase 17: typed fields; null stays null, zero means expired."""

    def test_to_dict_types(self):
        est = ShelfLifeEstimator(ShelfLifeConfig(), REAL_DB)
        d = est.estimate("apple", 0.88, freshness_class="fresh").to_dict()
        assert set(d) == {
            "fruit", "freshness_class", "freshness_confidence",
            "shelf_life_status", "remaining_days", "typical_min_days",
            "typical_max_days", "unit", "basis", "storage_condition",
            "explanation",
        }
        assert isinstance(d["remaining_days"], int)
        assert isinstance(d["freshness_confidence"], float)
        json.dumps(d)

    def test_null_vs_zero_semantics(self):
        est = ShelfLifeEstimator(ShelfLifeConfig(), REAL_DB)
        expired = est.estimate("apple", 0.9, freshness_class="rotten").to_dict()
        unknown = est.estimate("kiwi", 0.9, freshness_class="unknown").to_dict()
        unsupported = est.estimate("zzz", 0.9, freshness_class="fresh").to_dict()
        malformed = est.estimate("apple", "bad", freshness_class="fresh").to_dict()
        assert expired["remaining_days"] == 0          # semantic: expired
        assert unknown["remaining_days"] is None       # semantic: not estimated
        assert unsupported["remaining_days"] is None   # semantic: not estimated
        assert malformed["remaining_days"] is None     # semantic: not estimated
        assert malformed["freshness_confidence"] is None
        for d in (expired, unknown, unsupported, malformed):
            json.dumps(d)

    def test_disabled_status(self):
        est = ShelfLifeEstimator(ShelfLifeConfig(enabled=False), REAL_DB)
        r = est.estimate("apple", 0.9, freshness_class="fresh")
        assert r.shelf_life_status == "disabled" and r.remaining_days is None

