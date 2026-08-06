"""FreshSense AI - Phase 1 entry point.

Pipeline:

    Configuration
      -> Logging setup
      -> Dataset verification & statistics
      -> Model initialization
      -> Optimizer & scheduler
      -> Training
      -> Evaluation
      -> Save model & generate reports
      -> Exit

Everything is driven by ``configs/settings.yaml``; no hardcoded values live
in this module. Exceptions are logged to ``logs/errors.log`` and re-raised
so the process exit code is non-zero on failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.config import Config
from src.models.efficientnet import FreshSenseEfficientNet
from src.preprocessing.augmentation import AugmentationPipeline
from src.preprocessing.dataset import FreshSenseDatasetLoader
from src.training.evaluate import Evaluator
from src.training.losses import build_criterion
from src.training.trainer import Trainer
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def _apply_runtime_optimizations(config: Config) -> Config:
    """Apply OS- and device-specific optimizations to the config.

    - Windows: cap num_workers at 2 to avoid spawn overhead.
    - CPU: disable pin_memory and persistent_workers.
    - Ensure prefetch_factor is only used when workers > 0.
    - Set OpenCV threads to 1 to prevent thread oversubscription.
    """
    import platform

    # OpenCV: prevent thread oversubscription with DataLoader workers.
    cv2.setNumThreads(1)

    # Windows: spawn-based multiprocessing is heavy; cap workers.
    num_workers = config.data.num_workers
    if platform.system() == "Windows":
        num_workers = min(num_workers, 2)

    # CPU: pin_memory and persistent_workers are no-ops/warnings on CPU.
    is_cuda = config.device.type == "cuda"
    pin_memory = config.data.pin_memory and is_cuda
    persistent_workers = config.data.persistent_workers and num_workers > 0
    prefetch_factor = config.data.prefetch_factor if num_workers > 0 else 2

    # Build an updated config dict and recreate the frozen dataclass.
    config_dict = config.to_dict()
    config_dict["data"]["num_workers"] = num_workers
    config_dict["data"]["pin_memory"] = pin_memory
    config_dict["data"]["persistent_workers"] = persistent_workers
    config_dict["data"]["prefetch_factor"] = prefetch_factor

    return Config(
        project_name=config_dict["project_name"],
        random_seed=config_dict["random_seed"],
        paths=config.paths,
        data=config.data.__class__(**config_dict["data"]),
        model=config.model.__class__(**config_dict["model"]),
        training=config.training.__class__(**config_dict["training"]),
    )


def _log_config(config: Config) -> None:
    """Log the active configuration for reproducibility."""
    logger.info("Project   : %s", config.project_name)
    logger.info("Seed      : %d", config.random_seed)
    logger.info("Device    : %s", config.device)
    logger.info("Data      : %s", config.data)
    logger.info("Model     : %s", config.model)
    logger.info("Training  : %s", config.training)


def build_dataloaders(
    config: Config,
) -> tuple:
    """Build the augmentation pipeline and dataset loaders.

    Args:
        config: Active configuration.

    Returns:
        ``(train_loader, val_loader, test_loader, dataset_info)``
    """
    pipeline = AugmentationPipeline(
        image_size=(config.data.image_size, config.data.image_size)
    )

    loader = FreshSenseDatasetLoader(
        dataset_path=config.paths.raw_data_dir,
        train_transform=pipeline.get_transforms("train"),
        val_transform=pipeline.get_transforms("val"),
        test_transform=pipeline.get_transforms("test"),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=config.data.drop_last,
        test_size=config.data.test_split,
        val_size=config.data.val_split,
        random_state=config.random_seed,
    )

    # Verify the dataset loads one batch from every split before training.
    loader.verify()

    train_loader, val_loader, test_loader, dataset_info = loader.create_dataloaders()
    return train_loader, val_loader, test_loader, dataset_info


def build_model(config: Config, num_classes: int) -> FreshSenseEfficientNet:
    """Build and move the model to the configured device.

    Args:
        config: Active configuration.
        num_classes: Number of output classes from the dataset.

    Returns:
        The prepared model.
    """
    model = FreshSenseEfficientNet(
        num_classes=num_classes,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
        dropout=config.model.dropout,
        classifier_hidden=config.model.classifier_hidden,
    )
    model = model.to(config.device)

    logger.info("Model: %s", repr(model))
    logger.info(
        "Trainable params: %d | Frozen: %d | Total: %d",
        model.trainable_parameters(),
        model.frozen_parameters(),
        model.total_parameters(),
    )
    return model


def build_optimizer_and_scheduler(
    config: Config, model: FreshSenseEfficientNet
) -> tuple:
    """Build the optimizer and LR scheduler.

    Uses differential learning rates: backbone lr = 0.1x classifier lr
    (a transfer-learning best practice).

    Args:
        config: Active configuration.
        model: The model.

    Returns:
        ``(optimizer, scheduler)``
    """
    param_groups = model.get_parameter_groups(
        backbone_lr=config.training.learning_rate * 0.1,
        classifier_lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    optimizer = optim.AdamW(param_groups)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.factor,
        patience=config.training.lr_patience,
    )
    return optimizer, scheduler


def run_pipeline(config: Config) -> None:
    """Run the full Phase 1 training pipeline.

    Args:
        config: Active configuration.

    Raises:
        Exception: Any pipeline failure is logged to errors.log and re-raised.
    """
    try:
        # ------------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------------
        train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

        logger.info("Classes: %s", dataset_info.class_names)
        logger.info("Class distribution: %s", dataset_info.class_distribution)
        logger.info(
            "Valid images: %d | Skipped: %d",
            dataset_info.valid_count,
            dataset_info.skipped_count,
        )
        logger.info("\n%s", dataset_info.class_imbalance_report())

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        model = build_model(config, len(dataset_info.class_names))

        # ------------------------------------------------------------------
        # Loss, Optimizer, Scheduler
        # ------------------------------------------------------------------
        criterion = build_criterion(
            num_classes=len(dataset_info.class_names),
            label_smoothing=config.training.label_smoothing,
        )
        optimizer, scheduler = build_optimizer_and_scheduler(config, model)

        # ------------------------------------------------------------------
        # Trainer
        # ------------------------------------------------------------------
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device,
            epochs=config.training.epochs,
            checkpoint_dir=config.paths.checkpoint_dir,
            patience=config.training.patience,
            grad_clip=config.training.grad_clip,
            mixed_precision=config.training.mixed_precision,
            save_checkpoint_every=config.training.save_checkpoint_every,
            history_csv_path=config.paths.history_csv_path,
            resume_from=config.training.resume_from,
            class_names=dataset_info.class_names,
            config=config,
        )

        history = trainer.fit()

        logger.info(
            "Best epoch: %d | Best val loss: %.4f | Best val acc: %.2f%%",
            history.best_epoch,
            history.best_val_loss,
            history.best_val_acc,
        )

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            device=config.device,
            class_names=dataset_info.class_names,
            output_dir=config.paths.metrics_dir,
        )

        results = evaluator.evaluate()
        evaluator.print_results(results)

        # Save metrics JSON + all plots.
        saved_paths = evaluator.save_all(results)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Training completed successfully.")
        logger.info("Best model: %s", config.paths.best_model_path)
        logger.info("Last model: %s", config.paths.last_model_path)
        logger.info("History CSV: %s", config.paths.history_csv_path)
        logger.info("Metrics directory: %s", config.paths.metrics_dir)
        logger.info(
            "Artifacts written: %d (JSON + plots)", len(saved_paths) + 1
        )
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        raise

    except Exception:
        logger.exception("Pipeline failed. See errors.log for details.")
        raise


def main() -> int:
    """Entry point for ``python -m src.main``.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    try:
        # Load config first so paths exist before logging is configured.
        config = Config.from_yaml(Path("configs/settings.yaml"))
        config.paths.ensure_directories()

        # Production logging: console + logs/training.log + logs/errors.log.
        setup_logging(config.paths.logs_dir)

        # Apply OS- and device-specific runtime optimizations.
        config = _apply_runtime_optimizations(config)

        config.seed_everything()
        _log_config(config)

        run_pipeline(config)
        return 0
    except KeyboardInterrupt:
        # Logging may not be configured yet; use a minimal fallback.
        logging.warning("Pipeline interrupted by user.")
        return 130
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
