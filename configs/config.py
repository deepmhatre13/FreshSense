"""FreshSense AI - Central configuration.

This module defines the single source of truth for all Phase 1
configuration: data paths, model architecture, training hyperparameters,
and reproducibility settings.

The configuration is a frozen dataclass hierarchy so that:

- It is immutable at runtime (prevents accidental mutation).
- It is type-hinted (IDE support, static analysis).
- It can be serialized/deserialized to YAML for experiment tracking.
- It validates its own values at construction time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "PathsConfig",
    "TrainingConfig",
    "DEFAULT_CONFIG",
]


# ---------------------------------------------------------------------------
# Nested configuration groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used by the project."""

    root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    best_model_path: Path = field(init=False)
    last_model_path: Path = field(init=False)
    history_csv_path: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "data_dir", self.root / "data")
        object.__setattr__(self, "raw_data_dir", self.data_dir / "raw")
        object.__setattr__(self, "processed_data_dir", self.data_dir / "processed")
        object.__setattr__(self, "model_dir", self.root / "models")
        object.__setattr__(self, "checkpoint_dir", self.model_dir / "checkpoints")
        object.__setattr__(self, "best_model_path", self.checkpoint_dir / "best_model.pth")
        object.__setattr__(self, "last_model_path", self.checkpoint_dir / "last_model.pth")
        object.__setattr__(self, "history_csv_path", self.checkpoint_dir / "training_history.csv")
        object.__setattr__(self, "metrics_dir", self.model_dir / "metrics")
        object.__setattr__(self, "logs_dir", self.root / "logs")

    def ensure_directories(self) -> None:
        """Create all required directories if they do not exist."""
        for directory in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.metrics_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """Dataset and DataLoader settings."""

    image_size: int = 224
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 4  # Overridden at runtime: 2 on Windows, 4 on Linux/macOS.
    pin_memory: bool = True  # Overridden at runtime: False on CPU.
    persistent_workers: bool = True  # Only effective when num_workers > 0.
    prefetch_factor: int = 2  # Only used when num_workers > 0.
    drop_last: bool = False
    min_width: int = 224
    min_height: int = 224
    blur_threshold: float = 100.0
    dark_threshold: int = 40
    bright_threshold: int = 220
    quality_report_dir: str = "models/quality"

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1.")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError(
                "persistent_workers=True requires num_workers >= 1. "
                "Set num_workers>=1 or persistent_workers=False."
            )
        total_split = self.train_split + self.val_split + self.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError(
                f"Splits must sum to 1.0, got {total_split:.4f}."
            )
        if self.min_width <= 0 or self.min_height <= 0:
            raise ValueError("min_width and min_height must be positive.")


@dataclass(frozen=True)
class ModelConfig:
    """EfficientNet-B0 model settings."""

    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.30
    classifier_hidden: Optional[int] = 256  # Two-layer head: 768 -> 256 -> C

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1].")
        if self.classifier_hidden is not None and self.classifier_hidden <= 0:
            raise ValueError("classifier_hidden must be positive or None.")


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    factor: float = 0.1
    lr_patience: int = 2
    grad_clip: float = 1.0
    mixed_precision: bool = True
    label_smoothing: float = 0.0
    save_checkpoint_every: int = 5  # Every 5 epochs by default.
    print_every: int = 1
    resume_from: Optional[Path] = None
    unfreeze_epoch: Optional[int] = 10  # Unfreeze backbone after this epoch.
    warmup_epochs: int = 0  # Classifier-only warmup before unfreezing.

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if not 0.0 < self.factor <= 1.0:
            raise ValueError("factor must be in (0, 1].")
        if self.lr_patience <= 0:
            raise ValueError("lr_patience must be positive.")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.save_checkpoint_every <= 0:
            raise ValueError("save_checkpoint_every must be positive.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")
        if self.unfreeze_epoch is not None and self.unfreeze_epoch < 0:
            raise ValueError("unfreeze_epoch must be positive or None.")
        # Normalise resume_from to a Path if a string was passed via YAML.
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from))


@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating all sub-configs."""

    project_name: str = "FreshSense AI"
    random_seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ------------------------------------------------------------------
    # Device (computed lazily, then cached)
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Return the best available device (CUDA if present, else CPU).

        Results are cached so repeated calls are cheap.
        """
        if not hasattr(self, "_device"):
            object.__setattr__(
                self, "_device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        return self._device

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def seed_everything(self) -> None:
        """Seed all random number generators for deterministic training."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        # Make cuDNN deterministic (slightly slower but reproducible).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Warn rather than error on non-deterministic ops (e.g. grid_sample).
        torch.use_deterministic_algorithms(True, warn_only=True)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert the config to a plain dict (for YAML / logging / checkpoints)."""
        return {
            "project_name": self.project_name,
            "random_seed": self.random_seed,
            "paths": {
                "root": str(self.paths.root),
                "data_dir": str(self.paths.data_dir),
                "raw_data_dir": str(self.paths.raw_data_dir),
                "processed_data_dir": str(self.paths.processed_data_dir),
                "model_dir": str(self.paths.model_dir),
                "checkpoint_dir": str(self.paths.checkpoint_dir),
                "metrics_dir": str(self.paths.metrics_dir),
                "logs_dir": str(self.paths.logs_dir),
            },
            "data": {
                "image_size": self.data.image_size,
                "train_split": self.data.train_split,
                "val_split": self.data.val_split,
                "test_split": self.data.test_split,
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
                "pin_memory": self.data.pin_memory,
                "persistent_workers": self.data.persistent_workers,
                "prefetch_factor": self.data.prefetch_factor,
                "drop_last": self.data.drop_last,
                "min_width": self.data.min_width,
                "min_height": self.data.min_height,
                "blur_threshold": self.data.blur_threshold,
                "dark_threshold": self.data.dark_threshold,
                "bright_threshold": self.data.bright_threshold,
                "quality_report_dir": self.data.quality_report_dir,
            },
            "model": {
                "model_name": self.model.model_name,
                "pretrained": self.model.pretrained,
                "freeze_backbone": self.model.freeze_backbone,
                "dropout": self.model.dropout,
                "classifier_hidden": self.model.classifier_hidden,
            },
            "training": {
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "patience": self.training.patience,
                "factor": self.training.factor,
                "lr_patience": self.training.lr_patience,
                "grad_clip": self.training.grad_clip,
                "mixed_precision": self.training.mixed_precision,
                "label_smoothing": self.training.label_smoothing,
                "save_checkpoint_every": self.training.save_checkpoint_every,
                "print_every": self.training.print_every,
                "resume_from": (
                    str(self.training.resume_from)
                    if self.training.resume_from
                    else None
                ),
                "unfreeze_epoch": self.training.unfreeze_epoch,
                "warmup_epochs": self.training.warmup_epochs,
            },
        }

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load a Config from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            A fully-validated Config instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config YAML not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        data = raw.get("data", {})
        model = raw.get("model", {})
        training = raw.get("training", {})

        return cls(
            project_name=raw.get("project_name", "FreshSense AI"),
            random_seed=raw.get("random_seed", 42),
            data=DataConfig(**data),
            model=ModelConfig(**model),
            training=TrainingConfig(**training),
        )

    def save_yaml(self, path: Path | str) -> None:
        """Serialize this config to a YAML file.

        Args:
            path: Destination YAML path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Module-level singleton (convenience, but immutable)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = Config()