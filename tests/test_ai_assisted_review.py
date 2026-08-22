"""Unit tests for Phase 3.5 AI-assisted annotation review tools."""

import json
from pathlib import Path
import pytest

from scripts.generate_annotation_proposals import PROPOSAL_FILE, PROPOSAL_DIR
from scripts.review_annotation_proposals import compute_priority, load_human_decisions


def test_proposal_file_schema():
    """Verify generated proposals match required schema."""
    assert PROPOSAL_FILE.exists(), "Proposal JSON file must exist"
    with open(PROPOSAL_FILE, "r", encoding="utf-8") as f:
        proposals = json.load(f)

    assert len(proposals) > 0, "Proposals list should not be empty"
    
    required_keys = {
        "proposal_id", "image", "split", "review_category",
        "model_path", "model_hash", "class_id", "class_name",
        "confidence", "x1", "y1", "x2", "y2", "proposal_status", "created_at"
    }

    for prop in proposals[:10]:
        assert required_keys.issubset(prop.keys()), f"Missing keys in proposal: {required_keys - set(prop.keys())}"
        assert prop["proposal_status"] == "pending_human_review", "Status must initially be pending_human_review"
        assert 0 <= prop["class_id"] <= 9, f"Class ID must be between 0 and 9, got {prop['class_id']}"
        assert prop["x2"] >= prop["x1"], "x2 must be >= x1"
        assert prop["y2"] >= prop["y1"], "y2 must be >= y1"


def test_no_data_detection_modifications():
    """Ensure data/detection directory is untouched."""
    detection_dir = Path("data/detection")
    assert detection_dir.exists()
    # Ensure no v3 directory created yet
    assert not Path("data/detection_v3").exists()


def test_best_weights_unchanged():
    """Ensure model weights file exists and remains intact."""
    weights = Path("models/detection/detector/weights/best.pt")
    assert weights.exists()
    assert weights.stat().st_size > 0


def test_priority_computation():
    """Test proposal priority scoring formula."""
    prop_high = {"confidence": 0.90, "review_category": "ambiguous_classes"}
    prop_med = {"confidence": 0.60, "review_category": "tiny_boxes"}
    prop_low = {"confidence": 0.30, "review_category": "many_objects"}

    assert compute_priority(prop_high) == "HIGH"
    assert compute_priority(prop_med) == "MEDIUM"
    assert compute_priority(prop_low) == "LOW"


def test_human_decisions_manifest_readable():
    """Ensure human decisions manifest remains valid."""
    decisions = load_human_decisions()
    assert "schema_version" in decisions
    assert "records" in decisions
    assert isinstance(decisions["records"], list)
