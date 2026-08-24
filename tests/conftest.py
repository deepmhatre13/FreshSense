"""Shared pytest fixtures and path setup for the FreshSense test suite."""

import sys
from pathlib import Path

# Ensure the repository root is importable so `src.*` and `configs.*` work.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Path to the trained model checkpoint, used by integration-style tests.
CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "best_model.pth"

# Path to the 16-class freshness checkpoint (covers all fruits with valid
# training data: Apple, Banana, Grape, Guava, Jujube, Orange, Pomegranate,
# Strawberry).
FRESHNESS_CHECKPOINT_PATH = (
    REPO_ROOT / "models" / "checkpoints" / "freshness_efficientnet_b0_16class.pth"
)
