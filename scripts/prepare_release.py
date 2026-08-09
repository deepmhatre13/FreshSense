"""Release preparation script for FreshSense.

This script prepares the repository for release by:
- Cleaning caches and temporary files
- Verifying repository structure
- Checking for expected exclusions (datasets/checkpoints/logs)
- Printing repository size and estimated release ZIP size
- Verifying ZIP readiness

Run this before creating a release ZIP or committing to ensure
no large files or sensitive data are included.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure checkmark/warning symbols print correctly on non-UTF-8 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

__all__ = ["prepare_release"]


# Directories that are intentionally excluded from the release ZIP.
# These are expected to exist locally but must NOT be included.
EXCLUDED_DIRS = {
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    ".vscode",
    ".idea",
    ".git",
    ".ipynb_checkpoints",
    "logs",
    "runs",
    "data",
    "sample_data",
    "tmp",
    "temp",
    "dist",
    "build",
    "models/checkpoints",
    "models/evaluation",
    "models/metrics",
}

# File extensions that are intentionally excluded from the release ZIP.
EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pth",
    ".pt",
    ".onnx",
    ".log",
    ".pem",
    ".key",
    ".DS_Store",
}

# Files that should exist in the release
REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "configs/config.py",
    "configs/settings.yaml",
    "src/main.py",
    "src/models/efficientnet.py",
    "src/preprocessing/dataset.py",
    "src/preprocessing/augmentation.py",
    "src/preprocessing/preprocess.py",
    "src/preprocessing/quality.py",
    "src/training/trainer.py",
    "src/training/evaluate.py",
    "src/training/losses.py",
    "src/utils/logger.py",
    "src/utils/metrics.py",
    "src/utils/visualization.py",
    "src/inference/predict.py",
    "src/inference/camera.py",
    "src/inference/fps.py",
    "src/inference/overlay.py",
    "src/inference/tracker.py",
    "src/inference/predictor.py",
    "src/inference/pipeline.py",
    "scripts/validate_project.py",
    "scripts/prepare_release.py",
    "COLAB_SETUP.md",
    "LICENSE",
    "CHANGELOG.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "TESTING.md",
]

# Directories that should exist in the release.
# Note: excluded dirs (data/, logs/, models/checkpoints/, models/metrics/)
# are intentionally NOT part of the release and are not required here.
REQUIRED_DIRS = [
    "configs",
    "src",
    "src/models",
    "src/preprocessing",
    "src/training",
    "src/utils",
    "src/inference",
    "scripts",
    "models",
]


def _is_excluded(path: Path) -> bool:
    """Check if a path should be excluded from the release.

    Args:
        path: Path to check.

    Returns:
        True if the path should be excluded.
    """
    # Check if any parent directory is excluded
    for parent in path.parents:
        try:
            rel = parent.relative_to(project_root)
            if str(rel) in EXCLUDED_DIRS:
                return True
        except ValueError:
            pass

    # Check if the path itself is an excluded directory
    try:
        rel = path.relative_to(project_root)
        if str(rel) in EXCLUDED_DIRS:
            return True
    except ValueError:
        pass

    # Check file extension
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    return False


def _iter_pruned_files():
    """Walk the project tree, yielding releaseable files.

    Excluded directories (venv/, data/, logs/, checkpoints, caches, ...) are
    pruned during the walk, so we never descend into large folders. This
    keeps the scan fast and guarantees only files that will actually be
    released are reported on.

    Yields:
        (file_path, is_excluded) tuples for every file encountered.
    """
    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)
        # Do not descend into excluded directories.
        dirs[:] = [
            d for d in dirs
            if not _is_excluded(root_path / d)
        ]
        for name in files:
            path = root_path / name
            yield path, _is_excluded(path)


def clean_caches() -> List[str]:
    """Clean Python caches and temporary files.

    Returns:
        List of cleaned files/directories.
    """
    cleaned = []

    # Large folders we never descend into while cleaning.
    pruned = {
        "venv", ".venv", "env", "data", "logs", "runs", "sample_data",
        ".git", "dist", "build", "tmp", "temp", ".idea", ".vscode",
        ".ipynb_checkpoints", "htmlcov", ".coverage", "models",
    }

    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)

        # Remove __pycache__ directories.
        for name in list(dirs):
            if name == "__pycache__":
                shutil.rmtree(root_path / name, ignore_errors=True)
                cleaned.append(str(root_path / name))

        dirs[:] = [
            d for d in dirs
            if d not in pruned and d != "__pycache__"
        ]

        # Remove Python bytecode/cache files.
        for name in files:
            path = root_path / name
            if path.suffix.lower() in {".pyc", ".pyo", ".pyd"}:
                path.unlink(missing_ok=True)
                cleaned.append(str(path))

    logger.info(f"Cleaned {len(cleaned)} cache files/directories")
    return cleaned


def check_for_expected_exclusions() -> Tuple[List[str], List[str], List[str]]:
    """Check for expected exclusions (datasets, checkpoints, logs).

    These are expected to exist locally but are intentionally excluded
    from the release ZIP. They are NOT problems.

    Returns:
        Tuple of (datasets_found, checkpoints_found, logs_found)
    """
    datasets_found = []
    checkpoints_found = []
    logs_found = []

    # Check for data directory
    data_dir = project_root / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        datasets_found.append("data/")

    # Check for checkpoints
    checkpoint_dir = project_root / "models" / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            checkpoints_found.append(f"models/checkpoints/ ({len(checkpoints)} files)")

    # Check for logs
    logs_dir = project_root / "logs"
    if logs_dir.exists() and any(logs_dir.iterdir()):
        logs_found.append("logs/")

    # Check for evaluation outputs
    eval_dir = project_root / "models" / "evaluation"
    if eval_dir.exists() and any(eval_dir.iterdir()):
        checkpoints_found.append("models/evaluation/")

    # Check for metrics
    metrics_dir = project_root / "models" / "metrics"
    if metrics_dir.exists() and any(metrics_dir.iterdir()):
        checkpoints_found.append("models/metrics/")

    return datasets_found, checkpoints_found, logs_found


def verify_structure() -> List[str]:
    """Verify repository structure is correct.

    Returns:
        List of missing required files/directories.
    """
    missing = []

    # Check required files
    for file in REQUIRED_FILES:
        if not (project_root / file).exists():
            missing.append(f"Missing file: {file}")

    # Check required directories
    for directory in REQUIRED_DIRS:
        if not (project_root / directory).exists():
            missing.append(f"Missing directory: {directory}")

    return missing


def calculate_sizes() -> Tuple[int, int, List[Tuple[str, int]]]:
    """Calculate repository size and estimated release ZIP size.

    Repository size is the full on-disk footprint of the repository.
    Release ZIP size only counts files that will actually be released
    (excluded folders such as venv/, data/, logs/ and checkpoints are
    ignored). Large-file detection is limited to files that WILL be inside
    the release ZIP, so intentionally-excluded content is never flagged.

    Returns:
        Tuple of (total_size_bytes, release_size_bytes, large_release_files)
    """
    total_size = 0
    release_size = 0
    large_release_files = []

    # Only files that will actually be released are reported on - excluded
    # dirs (venv/, data/, logs/, checkpoints, caches, ...) are pruned during
    # the walk, so the scan is fast and their contents are never counted.
    for file_path, is_excluded in _iter_pruned_files():
        if not file_path.is_file():
            continue

        size = file_path.stat().st_size

        # Count only files that will be inside the release.
        if is_excluded:
            continue

        total_size += size
        release_size += size

        # Track large files that WILL be in the release (>10MB).
        # Intentionally-excluded content (venv, dataset, checkpoints) is never
        # flagged here because those folders are pruned from the walk.
        if size > 10 * 1024 * 1024:
            large_release_files.append(
                (str(file_path.relative_to(project_root)), size)
            )

    # Sort by size descending
    large_release_files.sort(key=lambda x: x[1], reverse=True)

    return total_size, release_size, large_release_files


def check_gitignore() -> List[str]:
    """Verify .gitignore covers all necessary patterns.

    Returns:
        List of missing patterns.
    """
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        return [".gitignore not found"]

    gitignore_content = gitignore_path.read_text()
    missing_patterns = []

    # Check for critical patterns
    critical_patterns = [
        "venv/",
        "__pycache__/",
        "*.pyc",
        "logs/",
        "models/checkpoints/*.pth",
        "data/",
        ".ipynb_checkpoints/",
    ]

    for pattern in critical_patterns:
        if pattern not in gitignore_content:
            missing_patterns.append(pattern)

    return missing_patterns


def _format_size(size_bytes: int) -> str:
    """Format a byte size into a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted size string (e.g. "18 MB", "2.6 GB").
    """
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def _collect_ignored(
    datasets: List[str], checkpoints: List[str], logs: List[str]
) -> List[str]:
    """Build the display names of excluded categories present on disk.

    These are intentionally left out of the release ZIP, so their existence
    is expected and never a release problem.

    Returns:
        List of excluded category display names.
    """
    ignored = []

    if any(project_root.glob(name) for name in ("venv", ".venv", "env")):
        ignored.append("venv")
    if datasets:
        ignored.append("data")
    if logs:
        ignored.append("logs")
    if checkpoints:
        ignored.append("checkpoints")
    if (project_root / ".git").exists():
        ignored.append(".git")

    return ignored


def prepare_release() -> bool:
    """Analyse the repository and report release readiness.

    Returns:
        True if the release is ready, False otherwise.
    """
    print("=" * 60)
    print("FreshSense Release Validation")
    print("=" * 60)
    print()

    # Clean caches (best effort, never blocks the release).
    clean_caches()

    # Expected exclusions - reported as expected, never as failures.
    datasets, checkpoints, logs = check_for_expected_exclusions()

    if datasets:
        print("Dataset detected")
        print("  ✓ expected")
    if checkpoints:
        print("Checkpoint detected")
        print("  ✓ expected")
    if logs:
        print("Logs detected")
        print("  ✓ expected")
    if not datasets and not checkpoints and not logs:
        print("No excluded content found")
    print()

    # Sizes: repository size is the full on-disk footprint, the estimated
    # ZIP size is the size of exactly the files that will be released.
    total_size, release_size, large_release_files = calculate_sizes()
    print("Repository Size")
    print(f"  {_format_size(total_size)}")
    print("Release ZIP Size")
    print(f"  {_format_size(release_size)}")
    print()

    ignored = _collect_ignored(datasets, checkpoints, logs)
    if ignored:
        print("Ignored")
        for item in ignored:
            print(f"  {item}")
        print()

    # Actual problems (NOT expected exclusions).
    missing = verify_structure()
    missing_patterns = check_gitignore()

    issues = []
    if missing:
        issues.append(f"Missing required files/directories ({len(missing)})")
    if missing_patterns:
        issues.append(f"Missing .gitignore patterns ({len(missing_patterns)})")
    if large_release_files:
        issues.append(f"Large files in release ({len(large_release_files)})")

    if large_release_files:
        print(f"⚠ Large files in release (>10 MB):")
        for file_path, size in large_release_files[:10]:
            print(f"  - {file_path}: {_format_size(size)}")
        print()

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("✗ Release NOT Ready")
    else:
        print("✓ Release Ready")

    print("=" * 60)
    return not issues


if __name__ == "__main__":
    ready = prepare_release()
    sys.exit(0 if ready else 1)