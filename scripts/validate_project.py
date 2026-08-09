"""Project validation script for FreshSense Phase 2.

This script validates the entire repository before training or deployment.
Run this before starting training to ensure everything is correctly configured.

Checks:
- Imports (all modules load correctly)
- Paths (all paths are relative, no hardcoded absolute paths)
- Dataset (data directory exists and has correct structure)
- Model (architecture is correct: 1280-256-6)
- Config (settings.yaml is valid and consistent)
- Checkpoint compatibility (checkpoint matches expected architecture)
- Preprocessing (preprocessing pipeline works)
- Transforms (augmentation pipelines build correctly)
- Output folders (all required directories exist or can be created)
- Requirements (all packages are installed with correct versions)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

__all__ = ["ValidationResult", "validate_project"]


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation check.

    Attributes:
        name: Name of the check.
        passed: Whether the check passed.
        message: Detailed message about the result.
        critical: If True, failure means the project cannot proceed.
    """

    name: str
    passed: bool
    message: str
    critical: bool = True


def check_imports() -> ValidationResult:
    """Verify all modules can be imported."""
    try:
        import configs.config  # noqa: F401
        import src.models.efficientnet  # noqa: F401
        import src.preprocessing.augmentation  # noqa: F401
        import src.preprocessing.dataset  # noqa: F401
        import src.preprocessing.preprocess  # noqa: F401
        import src.preprocessing.quality  # noqa: F401
        import src.training.trainer  # noqa: F401
        import src.training.losses  # noqa: F401
        import src.training.evaluate  # noqa: F401
        import src.utils.logger  # noqa: F401
        import src.utils.metrics  # noqa: F401
        import src.utils.visualization  # noqa: F401
        import src.inference.predict  # noqa: F401
        import src.inference.camera  # noqa: F401
        import src.inference.fps  # noqa: F401
        import src.inference.overlay  # noqa: F401
        import src.inference.tracker  # noqa: F401
        import src.inference.predictor  # noqa: F401
        import src.inference.pipeline  # noqa: F401

        return ValidationResult(
            name="Imports",
            passed=True,
            message="All modules imported successfully",
        )
    except Exception as exc:
        return ValidationResult(
            name="Imports",
            passed=False,
            message=f"Import failed: {exc}",
        )


def check_paths() -> ValidationResult:
    """Verify no hardcoded absolute paths."""
    issues = []
    py_files = list(project_root.rglob("*.py"))

    for file_path in py_files:
        if "validate_project.py" in str(file_path):
            continue

        content = file_path.read_text()
        # Check for Windows absolute paths
        if "D:\\" in content or "C:\\" in content:
            issues.append(f"{file_path.name}: Contains Windows absolute path")
        # Check for Unix absolute paths (except /usr, /tmp, etc.)
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("/") and not stripped.startswith(
                ("#!/", "//", "/usr", "/tmp", "/home")
            ):
                if "Path(" not in stripped and "pathlib" not in stripped:
                    issues.append(f"{file_path.name}: {stripped[:80]}")
                    break

    if issues:
        return ValidationResult(
            name="Paths",
            passed=False,
            message=f"Found {len(issues)} hardcoded paths:\n" + "\n".join(issues[:10]),
        )
    return ValidationResult(
        name="Paths",
        passed=True,
        message="No hardcoded absolute paths found",
    )


def check_dataset() -> ValidationResult:
    """Verify dataset directory exists and has correct structure."""
    data_dir = project_root / "data" / "raw"

    if not data_dir.exists():
        return ValidationResult(
            name="Dataset",
            passed=False,
            message=f"Data directory not found: {data_dir}",
            critical=False,
        )

    # Check for class subdirectories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        return ValidationResult(
            name="Dataset",
            passed=False,
            message="No class directories found in data/raw/",
            critical=False,
        )

    # Check for images in at least one class directory
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    has_images = False
    for class_dir in class_dirs:
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
        if images:
            has_images = True
            break

    if not has_images:
        return ValidationResult(
            name="Dataset",
            passed=False,
            message="No images found in class directories",
            critical=False,
        )

    return ValidationResult(
        name="Dataset",
        passed=True,
        message=f"Dataset found: {len(class_dirs)} classes",
    )


def check_model_architecture() -> ValidationResult:
    """Verify model architecture is correct (1280-256-6)."""
    try:
        import torch
        from src.models.efficientnet import FreshSenseEfficientNet

        # Build model with expected architecture
        model = FreshSenseEfficientNet(
            num_classes=6,
            pretrained=False,
            freeze_backbone=False,
            dropout=0.3,
            classifier_hidden=256,
        )

        # Verify classifier structure
        classifier_layers = list(model.model.classifier.children())
        if len(classifier_layers) != 4:  # Dropout, Linear(1280->256), ReLU, Linear(256->6)
            return ValidationResult(
                name="Model Architecture",
                passed=False,
                message=f"Expected 4 classifier layers, found {len(classifier_layers)}",
            )

        # Check layer dimensions
        linear1 = classifier_layers[1]  # First linear layer
        linear2 = classifier_layers[3]  # Second linear layer

        if linear1.in_features != 1280 or linear1.out_features != 256:
            return ValidationResult(
                name="Model Architecture",
                passed=False,
                message=f"Expected Linear(1280, 256), got Linear({linear1.in_features}, {linear1.out_features})",
            )

        if linear2.in_features != 256 or linear2.out_features != 6:
            return ValidationResult(
                name="Model Architecture",
                passed=False,
                message=f"Expected Linear(256, 6), got Linear({linear2.in_features}, {linear2.out_features})",
            )

        return ValidationResult(
            name="Model Architecture",
            passed=True,
            message="Architecture verified: 1280-256-6",
        )
    except Exception as exc:
        return ValidationResult(
            name="Model Architecture",
            passed=False,
            message=f"Architecture check failed: {exc}",
        )


def check_config() -> ValidationResult:
    """Verify configuration is valid and consistent."""
    try:
        config_path = project_root / "configs" / "settings.yaml"
        if not config_path.exists():
            return ValidationResult(
                name="Configuration",
                passed=False,
                message="configs/settings.yaml not found",
            )

        from configs.config import Config

        config = Config.from_yaml(config_path)

        # Verify key settings
        if config.data.image_size != 224:
            return ValidationResult(
                name="Configuration",
                passed=False,
                message=f"Expected image_size=224, got {config.data.image_size}",
            )

        if config.model.classifier_hidden != 256:
            return ValidationResult(
                name="Configuration",
                passed=False,
                message=f"Expected classifier_hidden=256, got {config.model.classifier_hidden}",
            )

        return ValidationResult(
            name="Configuration",
            passed=True,
            message=f"Config valid: {config.project_name}",
        )
    except Exception as exc:
        return ValidationResult(
            name="Configuration",
            passed=False,
            message=f"Config validation failed: {exc}",
        )


def check_checkpoint_compatibility(checkpoint_path: str = None) -> ValidationResult:
    """Verify checkpoint compatibility with current architecture."""
    if checkpoint_path is None:
        checkpoint_path = project_root / "models" / "checkpoints" / "best_model.pth"

    if not Path(checkpoint_path).exists():
        return ValidationResult(
            name="Checkpoint",
            passed=False,
            message=f"Checkpoint not found: {checkpoint_path}",
            critical=False,
        )

    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Check versioning
        checkpoint_version = checkpoint.get("checkpoint_version", 1)
        classifier_type = checkpoint.get("classifier_type", "unknown")
        architecture_version = checkpoint.get("architecture_version", "unknown")

        expected_classifier = "1280-256-6"

        if classifier_type != expected_classifier:
            return ValidationResult(
                name="Checkpoint",
                passed=False,
                message=f"Checkpoint architecture mismatch: {classifier_type} (expected {expected_classifier})",
            )

        return ValidationResult(
            name="Checkpoint",
            passed=True,
            message=f"Checkpoint compatible: v{checkpoint_version}, {architecture_version}, {classifier_type}",
        )
    except Exception as exc:
        return ValidationResult(
            name="Checkpoint",
            passed=False,
            message=f"Checkpoint check failed: {exc}",
        )


def check_preprocessing() -> ValidationResult:
    """Verify preprocessing pipeline works."""
    try:
        import numpy as np
        from src.preprocessing.preprocess import ImagePreprocessor
        from src.preprocessing.augmentation import AugmentationPipeline

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test preprocessor
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        processed = preprocessor.preprocess(dummy_image)
        assert processed.shape == (224, 224, 3), f"Expected (224, 224, 3), got {processed.shape}"

        # Test augmentation pipeline
        pipeline = AugmentationPipeline(image_size=(224, 224))
        train_transforms = pipeline.train_transforms()
        val_transforms = pipeline.validation_transforms()

        # Test transforms
        transformed = val_transforms(image=processed)
        assert "image" in transformed, "Transforms did not return 'image' key"

        return ValidationResult(
            name="Preprocessing",
            passed=True,
            message="Preprocessing pipeline verified",
        )
    except Exception as exc:
        return ValidationResult(
            name="Preprocessing",
            passed=False,
            message=f"Preprocessing check failed: {exc}",
        )


def check_output_folders() -> ValidationResult:
    """Verify output folders exist or can be created."""
    folders = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "models/metrics",
        "logs",
    ]

    missing = []
    for folder in folders:
        folder_path = project_root / folder
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            missing.append(f"{folder}: {exc}")

    if missing:
        return ValidationResult(
            name="Output Folders",
            passed=False,
            message=f"Cannot create folders:\n" + "\n".join(missing),
        )

    return ValidationResult(
        name="Output Folders",
        passed=True,
        message="All output folders ready",
    )


def check_requirements() -> ValidationResult:
    """Verify required packages are installed."""
    required = {
        "torch": None,
        "torchvision": None,
        "opencv-python": None,
        "albumentations": None,
        "numpy": None,
        "scikit-learn": None,
        "tqdm": None,
        "pyyaml": None,
        "matplotlib": None,
    }

    missing = []
    for package in required:
        try:
            if package == "opencv-python":
                import cv2
                required[package] = cv2.__version__
            elif package == "scikit-learn":
                import sklearn
                required[package] = sklearn.__version__
            elif package == "pyyaml":
                required[package] = yaml.__version__
            else:
                module = __import__(package)
                required[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            missing.append(package)

    if missing:
        return ValidationResult(
            name="Requirements",
            passed=False,
            message=f"Missing packages: {', '.join(missing)}",
        )

    versions = "\n".join(f"  {pkg}: {ver}" for pkg, ver in required.items() if ver)
    return ValidationResult(
        name="Requirements",
        passed=True,
        message=f"All packages installed:\n{versions}",
    )


def validate_project() -> List[ValidationResult]:
    """Run all validation checks.

    Returns:
        List of ValidationResult objects.
    """
    results = []

    logger.info("=" * 60)
    logger.info("FreshSense Project Validation")
    logger.info("=" * 60)
    logger.info("")

    checks = [
        ("Imports", check_imports),
        ("Paths", check_paths),
        ("Dataset", check_dataset),
        ("Model Architecture", check_model_architecture),
        ("Configuration", check_config),
        ("Checkpoint", check_checkpoint_compatibility),
        ("Preprocessing", check_preprocessing),
        ("Output Folders", check_output_folders),
        ("Requirements", check_requirements),
    ]

    for name, check_func in checks:
        logger.info(f"Checking {name}...")
        result = check_func()
        results.append(result)

        if result.passed:
            logger.info(f"✓ {result.message}")
        else:
            status = "CRITICAL" if result.critical else "WARNING"
            logger.error(f"✗ [{status}] {result.message}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    critical_failed = sum(1 for r in results if not r.passed and r.critical)

    logger.info(f"Passed: {passed}/{len(results)}")
    logger.info(f"Failed: {failed}/{len(results)}")
    logger.info(f"Critical failures: {critical_failed}")

    if critical_failed > 0:
        logger.error("")
        logger.error("CRITICAL ISSUES FOUND - Project cannot proceed:")
        for result in results:
            if not result.passed and result.critical:
                logger.error(f"  - {result.name}: {result.message}")
        logger.error("")
        logger.error("Please fix these issues before training.")
    elif failed > 0:
        logger.warning("")
        logger.warning("Non-critical issues found:")
        for result in results:
            if not result.passed and not result.critical:
                logger.warning(f"  - {result.name}: {result.message}")
    else:
        logger.info("")
        logger.info("✓ All checks passed! Project is ready for training.")

    return results


if __name__ == "__main__":
    results = validate_project()

    # Exit with error code if critical checks failed
    critical_failed = any(not r.passed and r.critical for r in results)
    sys.exit(1 if critical_failed else 0)