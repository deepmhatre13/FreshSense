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
    "InferenceConfig",
    "DetectionConfig",
    "DEFAULT_CONFIG",
]


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
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
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
        if not 0.0 <= self.train_split <= 1.0:
            raise ValueError("train_split must be in [0.0, 1.0].")
        if not 0.0 <= self.val_split <= 1.0:
            raise ValueError("val_split must be in [0.0, 1.0].")
        if not 0.0 <= self.test_split <= 1.0:
            raise ValueError("test_split must be in [0.0, 1.0].")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be >= 1.")
        total_split = self.train_split + self.val_split + self.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total_split:.4f}.")
        if self.min_width <= 0 or self.min_height <= 0:
            raise ValueError("min_width and min_height must be positive.")


@dataclass(frozen=True)
class ModelConfig:
    """EfficientNet-B0 model settings."""

    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout: float = 0.30
    classifier_hidden: Optional[int] = 256

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
    save_checkpoint_every: int = 5
    print_every: int = 1
    resume_from: Optional[Path] = None
    unfreeze_epoch: Optional[int] = 10
    warmup_epochs: int = 0

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
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from))


@dataclass(frozen=True)
class InferenceConfig:
    """Phase 3 real-time inference settings."""

    ema_alpha: float = 0.2
    vote_window: int = 15
    lock_frames: int = 5
    stabilizer_confidence_threshold: float = 0.70
    brightness_min: int = 40
    brightness_max: int = 220
    blur_threshold: float = 100.0
    contrast_min: float = 20.0
    motion_threshold: float = 35.0
    use_motion_detection: bool = True
    save_logs: bool = True
    session_log_dir: str = "logs/session"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0.0, 1.0].")
        if self.vote_window <= 0:
            raise ValueError("vote_window must be positive.")
        if self.lock_frames <= 0:
            raise ValueError("lock_frames must be positive.")
        if not 0.0 <= self.stabilizer_confidence_threshold <= 1.0:
            raise ValueError("stabilizer_confidence_threshold must be in [0.0, 1.0].")
        if not 0 <= self.brightness_min <= 255:
            raise ValueError("brightness_min must be in [0, 255].")
        if not 0 <= self.brightness_max <= 255:
            raise ValueError("brightness_max must be in [0, 255].")
        if self.brightness_min >= self.brightness_max:
            raise ValueError("brightness_min must be less than brightness_max.")
        if self.blur_threshold <= 0:
            raise ValueError("blur_threshold must be positive.")
        if self.contrast_min < 0:
            raise ValueError("contrast_min must be non-negative.")
        if self.motion_threshold < 0:
            raise ValueError("motion_threshold must be non-negative.")


@dataclass(frozen=True)
class DetectionDatasetConfig:
    """Roboflow detection dataset configuration.

    Attributes:
        roboflow_workspace: Roboflow workspace name.
        roboflow_project: Roboflow project name.
        roboflow_version: Dataset version number.
        detection_data_dir: Local directory for detection dataset.
        detector_model: Base YOLO model for training.
        detector_epochs: Number of training epochs.
        detector_batch: Training batch size.
        detector_imgsz: Training image size.
        detector_output_dir: Output directory for trained models.
        detector_device: Device for training ("cuda", "cpu", "auto").
        detector_workers: Number of data loader workers.
        detector_patience: Early stopping patience.
    """

    roboflow_workspace: str = "smartfresh-ai"
    roboflow_project: str = "fruits-test"
    roboflow_version: int = 1
    detection_data_dir: Path = field(default_factory=lambda: Path("data/detection"))
    detector_model: str = "yolo11n.pt"
    detector_epochs: int = 50
    detector_batch: int = 16
    detector_imgsz: int = 640
    detector_output_dir: Path = field(default_factory=lambda: Path("models/detection"))
    detector_device: str = "auto"
    detector_workers: int = 4
    detector_patience: int = 10

    def __post_init__(self) -> None:
        if self.roboflow_version <= 0:
            raise ValueError("roboflow_version must be positive.")
        if self.detector_epochs <= 0:
            raise ValueError("detector_epochs must be positive.")
        if self.detector_batch <= 0:
            raise ValueError("detector_batch must be positive.")
        if self.detector_imgsz <= 0:
            raise ValueError("detector_imgsz must be positive.")
        if self.detector_patience <= 0:
            raise ValueError("detector_patience must be positive.")


@dataclass(frozen=True)
class DetectionConfig:
    """Phase 4 object detection settings."""

    # Detector backend
    detector_backend: str = "yolo"
    detector_weights: str = "yolo11n.pt"
    detection_confidence: float = 0.45
    detection_iou: float = 0.45
    max_detections: int = 20

    # Cropper
    crop_expand_scale: float = 0.08
    crop_min_side: int = 32
    crop_min_area: int = 1024
    crop_target_size: int = 224

    # Tracker
    tracker_iou_threshold: float = 0.3
    tracker_max_distance: float = 120.0
    tracker_max_lost_frames: int = 15

    # Confidence fusion
    detection_weight: float = 0.4
    classification_weight: float = 0.6

    # Classification cadence (adaptive)
    classify_every_n_frames: int = 3

    # Shelf life
    fresh_bonus: float = 0.5

    # Explainability
    gradcam_enabled: bool = False

    # Quality gates
    quality_warning: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be in [0.0, 1.0].")
        if not 0.0 <= self.detection_iou <= 1.0:
            raise ValueError("detection_iou must be in [0.0, 1.0].")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive.")
        if self.classify_every_n_frames <= 0:
            raise ValueError("classify_every_n_frames must be positive.")
        if self.detection_weight < 0 or self.classification_weight < 0:
            raise ValueError("Weights must be non-negative.")


@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating all sub-configs."""

    project_name: str = "FreshSense AI"
    random_seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    detection_dataset: DetectionDatasetConfig = field(default_factory=DetectionDatasetConfig)

    @property
    def device(self) -> torch.device:
        if not hasattr(self, "_device"):
            object.__setattr__(
                self, "_device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        return self._device

    def seed_everything(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def to_dict(self) -> dict:
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
                "resume_from": str(self.training.resume_from) if self.training.resume_from else None,
                "unfreeze_epoch": self.training.unfreeze_epoch,
                "warmup_epochs": self.training.warmup_epochs,
            },
            "inference": {
                "ema_alpha": self.inference.ema_alpha,
                "vote_window": self.inference.vote_window,
                "lock_frames": self.inference.lock_frames,
                "stabilizer_confidence_threshold": self.inference.stabilizer_confidence_threshold,
                "brightness_min": self.inference.brightness_min,
                "brightness_max": self.inference.brightness_max,
                "blur_threshold": self.inference.blur_threshold,
                "contrast_min": self.inference.contrast_min,
                "motion_threshold": self.inference.motion_threshold,
                "use_motion_detection": self.inference.use_motion_detection,
                "save_logs": self.inference.save_logs,
                "session_log_dir": self.inference.session_log_dir,
            },
            "detection": {
                "detector_backend": self.detection.detector_backend,
                "detector_weights": self.detection.detector_weights,
                "detection_confidence": self.detection.detection_confidence,
                "detection_iou": self.detection.detection_iou,
                "max_detections": self.detection.max_detections,
                "crop_expand_scale": self.detection.crop_expand_scale,
                "crop_min_side": self.detection.crop_min_side,
                "crop_min_area": self.detection.crop_min_area,
                "crop_target_size": self.detection.crop_target_size,
                "tracker_iou_threshold": self.detection.tracker_iou_threshold,
                "tracker_max_distance": self.detection.tracker_max_distance,
                "tracker_max_lost_frames": self.detection.tracker_max_lost_frames,
                "detection_weight": self.detection.detection_weight,
                "classification_weight": self.detection.classification_weight,
                "classify_every_n_frames": self.detection.classify_every_n_frames,
                "fresh_bonus": self.detection.fresh_bonus,
                "gradcam_enabled": self.detection.gradcam_enabled,
                "quality_warning": self.detection.quality_warning,
            },
            "detection_dataset": {
                "roboflow_workspace": self.detection_dataset.roboflow_workspace,
                "roboflow_project": self.detection_dataset.roboflow_project,
                "roboflow_version": self.detection_dataset.roboflow_version,
                "detection_data_dir": str(self.detection_dataset.detection_data_dir),
                "detector_model": self.detection_dataset.detector_model,
                "detector_epochs": self.detection_dataset.detector_epochs,
                "detector_batch": self.detection_dataset.detector_batch,
                "detector_imgsz": self.detection_dataset.detector_imgsz,
                "detector_output_dir": str(self.detection_dataset.detector_output_dir),
                "detector_device": self.detection_dataset.detector_device,
                "detector_workers": self.detection_dataset.detector_workers,
                "detector_patience": self.detection_dataset.detector_patience,
            },
        }

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config YAML not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data = raw.get("data", {})
        model = raw.get("model", {})
        training = raw.get("training", {})
        inference = raw.get("inference", {})
        return cls(
            project_name=raw.get("project_name", "FreshSense AI"),
            random_seed=raw.get("random_seed", 42),
            data=DataConfig(**data),
            model=ModelConfig(**model),
            training=TrainingConfig(**training),
            inference=InferenceConfig(**inference),
            detection=DetectionConfig(**raw.get("detection", {})),
            detection_dataset=DetectionDatasetConfig(**raw.get("detection_dataset", {})),
        )

    def save_yaml(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)


DEFAULT_CONFIG = Config()
