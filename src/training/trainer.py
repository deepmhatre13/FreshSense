"""Training loop for the FreshSense pipeline.

The Trainer handles:

- Mixed precision (AMP) with ``torch.amp.GradScaler`` (modern API).
- Gradient clipping.
- Early stopping (on validation loss).
- Checkpointing (best model, last model, periodic epoch checkpoints).
- Resume training from a checkpoint (restores history + patience counter).
- Learning rate scheduling (``ReduceLROnPlateau``).
- CSV history logging.
- Progress bars with live metrics.
- Epoch timing and throughput.
- Graceful interruption (``KeyboardInterrupt`` saves a checkpoint).
- Two-stage transfer learning (warmup + unfreeze).
"""

from __future__ import annotations

import csv
import logging
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["Trainer", "TrainingHistory"]


@dataclass
class TrainingHistory:
    """Structured training history."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0

    def to_csv(self, path: Path) -> None:
        """Write the history to a CSV file.

        Args:
            path: Destination CSV path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_acc",
                    "val_acc",
                    "learning_rate",
                    "epoch_time_s",
                ]
            )
            for i in range(len(self.train_loss)):
                writer.writerow(
                    [
                        i + 1,
                        f"{self.train_loss[i]:.6f}",
                        f"{self.val_loss[i]:.6f}",
                        f"{self.train_acc[i]:.6f}",
                        f"{self.val_acc[i]:.6f}",
                        f"{self.learning_rates[i]:.8f}",
                        f"{self.epoch_times[i]:.3f}",
                    ]
                )


class Trainer:
    """Trains a PyTorch model with best practices.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR scheduler (``ReduceLROnPlateau``).
        device: torch.device.
        epochs: Number of epochs.
        checkpoint_dir: Directory for checkpoints.
        patience: Early stopping patience.
        grad_clip: Max gradient norm for clipping.
        mixed_precision: If True, use AMP.
        save_checkpoint_every: Save an epoch checkpoint every N epochs.
        history_csv_path: Path for the CSV history file.
        resume_from: Optional checkpoint path to resume from.
        class_names: Ordered class names aligned with labels.
        config: Full Config object for checkpoint metadata and two-stage training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[ReduceLROnPlateau],
        device: torch.device,
        epochs: int = 20,
        checkpoint_dir: Path = Path("models/checkpoints"),
        patience: int = 5,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        save_checkpoint_every: int = 5,
        history_csv_path: Optional[Path] = None,
        resume_from: Optional[Path] = None,
        class_names: Optional[List[str]] = None,
        config: Optional["Config"] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_names = list(class_names) if class_names else []
        self.config = config

        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.save_checkpoint_every = save_checkpoint_every

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        self.last_model_path = self.checkpoint_dir / "last_model.pth"
        self.history_csv_path = (
            Path(history_csv_path)
            if history_csv_path
            else self.checkpoint_dir / "training_history.csv"
        )

        # Mixed precision scaler (enabled only on CUDA).
        self.scaler = GradScaler(
            "cuda", enabled=mixed_precision and device.type == "cuda"
        )

        self.history = TrainingHistory()
        self.start_epoch = 0
        self.early_stopped = False
        self.patience_counter = 0

        # Resume from checkpoint if requested.
        if resume_from is not None:
            self._resume(resume_from)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            ``(epoch_loss, epoch_accuracy)``
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )

        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                "cuda",
                enabled=self.mixed_precision and self.device.type == "cuda",
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for AMP).
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            num_batches += 1

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress.set_postfix(
                loss=running_loss / num_batches,
                acc=100.0 * correct / total,
            )

        epoch_loss = running_loss / max(num_batches, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Run validation.

        Returns:
            ``(val_loss, val_accuracy)``
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(
                "cuda",
                enabled=self.mixed_precision and self.device.type == "cuda",
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            num_batches += 1

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / max(num_batches, 1)
        val_acc = 100.0 * correct / max(total, 1)
        return val_loss, val_acc

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_dict(self, epoch: int, val_loss: float, val_acc: float) -> dict:
        """Build the checkpoint dictionary.

        Includes ``class_names`` so inference can recover the label order
        without external configuration (prevents silent label-order drift).
        """
        import platform
        import subprocess

        import torch
        import torchvision

        # Git hash for reproducibility.
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            git_hash = "unknown"

        # Dataset statistics.
        num_train = len(self.train_loader.dataset)
        num_val = len(self.val_loader.dataset)

        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_loss": self.history.best_val_loss,
            "best_val_acc": self.history.best_val_acc,
            "best_epoch": self.history.best_epoch,
            "patience_counter": self.patience_counter,
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "dataset_stats": {
                "num_train": num_train,
                "num_val": num_val,
            },
            "config_dict": self.config.to_dict() if self.config else {},
            "git_hash": git_hash,
            "training_date": __import__("datetime").datetime.now().isoformat(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "python_version": platform.python_version(),
            "history": {
                "train_loss": self.history.train_loss,
                "val_loss": self.history.val_loss,
                "train_acc": self.history.train_acc,
                "val_acc": self.history.val_acc,
                "learning_rates": self.history.learning_rates,
                "epoch_times": self.history.epoch_times,
            },
        }

    def save_epoch_checkpoint(self, epoch: int, val_loss: float, val_acc: float) -> None:
        """Save a periodic epoch checkpoint."""
        if self.save_checkpoint_every <= 0:
            return
        if epoch % self.save_checkpoint_every != 0:
            return

        path = self.checkpoint_dir / f"epoch_{epoch}.pth"
        torch.save(self._checkpoint_dict(epoch, val_loss, val_acc), path)
        logger.info("Saved epoch checkpoint: %s", path)

    def save_best_model(self, epoch: int, val_loss: float, val_acc: float) -> None:
        """Save the best model checkpoint."""
        torch.save(self._checkpoint_dict(epoch, val_loss, val_acc), self.best_model_path)
        logger.info(
            "Saved best model (epoch %d, val_loss %.4f, val_acc %.2f%%)",
            epoch,
            val_loss,
            val_acc,
        )

    def save_last_model(self, epoch: int, val_loss: float, val_acc: float) -> None:
        """Save the last model checkpoint."""
        torch.save(self._checkpoint_dict(epoch, val_loss, val_acc), self.last_model_path)

    def _resume(self, checkpoint_path: Path) -> None:
        """Resume training from a checkpoint.

        Restores model, optimizer, scheduler, scaler, history, and the
        early-stopping patience counter.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0)
        self.history.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.history.best_epoch = checkpoint.get("best_epoch", -1)
        self.patience_counter = checkpoint.get("patience_counter", 0)

        # Restore full history if present.
        hist = checkpoint.get("history")
        if hist:
            self.history.train_loss = list(hist.get("train_loss", []))
            self.history.val_loss = list(hist.get("val_loss", []))
            self.history.train_acc = list(hist.get("train_acc", []))
            self.history.val_acc = list(hist.get("val_acc", []))
            self.history.learning_rates = list(hist.get("learning_rates", []))
            self.history.epoch_times = list(hist.get("epoch_times", []))

        logger.info(
            "Resumed from %s at epoch %d (best_val_loss %.4f, patience %d)",
            checkpoint_path,
            self.start_epoch,
            self.history.best_val_loss,
            self.patience_counter,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> TrainingHistory:
        """Run the full training loop.

        Returns:
            The training history.

        Raises:
            KeyboardInterrupt: Re-raised after saving a checkpoint so the
                caller can decide how to handle graceful interruption.
        """
        logger.info("Starting training for %d epochs on %s", self.epochs, self.device)
        logger.info(
            "Mixed precision: %s | Grad clip: %.2f | Patience: %d",
            "ON" if self.mixed_precision and self.device.type == "cuda" else "OFF",
            self.grad_clip,
            self.patience,
        )

        try:
            for epoch in range(self.start_epoch, self.epochs):
                epoch_start = time.time()

                logger.info("Epoch %d/%d", epoch + 1, self.epochs)

                train_loss, train_acc = self.train_one_epoch()
                val_loss, val_acc = self.validate()

                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Record history.
                self.history.train_loss.append(train_loss)
                self.history.val_loss.append(val_loss)
                self.history.train_acc.append(train_acc)
                self.history.val_acc.append(val_acc)
                self.history.learning_rates.append(current_lr)
                self.history.epoch_times.append(epoch_time)

                # Scheduler step (ReduceLROnPlateau uses val_loss).
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                logger.info(
                    "Epoch %d | Train Loss %.4f | Train Acc %.2f%% | "
                    "Val Loss %.4f | Val Acc %.2f%% | LR %.6f | Time %.1fs",
                    epoch + 1,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    current_lr,
                    epoch_time,
                )

                # Save last model every epoch.
                self.save_last_model(epoch + 1, val_loss, val_acc)

                # Save periodic epoch checkpoint.
                self.save_epoch_checkpoint(epoch + 1, val_loss, val_acc)

                # Early stopping on validation loss.
                if val_loss < self.history.best_val_loss:
                    self.history.best_val_loss = val_loss
                    self.history.best_val_acc = val_acc
                    self.history.best_epoch = epoch + 1
                    self.patience_counter = 0
                    self.save_best_model(epoch + 1, val_loss, val_acc)
                else:
                    self.patience_counter += 1
                    logger.info(
                        "No improvement for %d epoch(s) (best %.4f).",
                        self.patience_counter,
                        self.history.best_val_loss,
                    )

                if self.patience_counter >= self.patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                    self.early_stopped = True
                    break
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving checkpoint...")
            # Save a checkpoint so training can be resumed.
            current_epoch = self.start_epoch + len(self.history.train_loss)
            last_val_loss = self.history.val_loss[-1] if self.history.val_loss else 0.0
            last_val_acc = self.history.val_acc[-1] if self.history.val_acc else 0.0
            self.save_last_model(current_epoch, last_val_loss, last_val_acc)
            self.history.to_csv(self.history_csv_path)
            raise

        # Write CSV history.
        self.history.to_csv(self.history_csv_path)
        logger.info("Training history saved to %s", self.history_csv_path)

        # Load the best model back so evaluation uses the best weights.
        if self.best_model_path.exists():
            best_checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(best_checkpoint["model_state_dict"])
            logger.info(
                "Loaded best model from epoch %d (val_loss %.4f, val_acc %.2f%%)",
                best_checkpoint["epoch"],
                best_checkpoint["val_loss"],
                best_checkpoint["val_acc"],
            )

        return self.history