import json
from pathlib import Path

import pytest
from src.inference.fruit_metadata import FruitMetadataDatabase
from src.inference.shelf_life import ShelfLifeConfig, ShelfLifeEstimator

def test_shelf_life_with_metadata(tmp_path):
    # a) loading with the file present and complete
    db_file = tmp_path / "fruit_database.json"
    data = {
        "apple": {
            "typical_shelf_life_days": [14, 30]
        }
    }
    db_file.write_text(json.dumps(data))
    
    db = FruitMetadataDatabase(str(db_file))
    estimator = ShelfLifeEstimator(ShelfLifeConfig(), db)
    
    # Test valid fruit
    result = estimator.estimate("apple", fused_confidence=1.0, is_fresh=True)
    assert result.basis_type == "metadata_heuristic"
    assert result.min_days == 14
    assert result.max_days == 30

def test_shelf_life_without_file(tmp_path):
    # b) loading with the file absent - assert the system does NOT silently produce a generic-looking shelf-life number
    db_file = tmp_path / "missing_db.json"
    
    db = FruitMetadataDatabase(str(db_file))
    estimator = ShelfLifeEstimator(ShelfLifeConfig(), db)
    
    result = estimator.estimate("apple", fused_confidence=1.0, is_fresh=True)
    assert result.basis_type == "unavailable"
    assert result.min_days == 0
    assert result.max_days == 0

def test_shelf_life_missing_fruit_entry(tmp_path):
    # c) a fruit class with no metadata entry - same assertion
    db_file = tmp_path / "fruit_database.json"
    data = {
        "apple": {
            "typical_shelf_life_days": [14, 30]
        }
    }
    db_file.write_text(json.dumps(data))
    
    db = FruitMetadataDatabase(str(db_file))
    estimator = ShelfLifeEstimator(ShelfLifeConfig(), db)
    
    # Test missing fruit
    result = estimator.estimate("grape", fused_confidence=1.0, is_fresh=True)
    assert result.basis_type == "unavailable"
    assert result.min_days == 0
    assert result.max_days == 0
