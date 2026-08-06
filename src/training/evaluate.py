"""Model evaluation for the FreshSense pipeline.

The Evaluator computes a comprehensive set of classification metrics:

- Top-1 accuracy
- Weighted precision / recall / F1
- Per-class accuracy, precision, recall, F1
- Confusion matrix
- Classification report
- ROC AUC (one-vs-rest, multi-class)
- PR AUC (one-vs-rest, multi-class)
- Misclassified image list + grid plot
- Prediction confidence distribution + histogram

Results are returned as a structured :class:`EvaluationResults` dataclass and
can be saved to disk (JSON metrics + plots). The Evaluator caches the raw
prediction arrays from :meth:`evaluate` internally so :meth:`save_all` can
render every plot without re-running inference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.utils.metrics import (
    per_class_accuracy,
    ovr_pr_auc,
    ovr_roc_auc,
    top_k_accuracy,
)
from src.utils.visualization import (
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_misclassified_grid,
    plot_pr_curves,
    plot_roc_curves,
)

logger = logging.getLogger(__name__)

__all__ = ["Evaluator", "EvaluationResults", "MisclassifiedSample"]


@dataclass
class MisclassifiedSample:
    """A single misclassified image."""

    image_path: str
    true_label: int
    true_class: str
    predicted_label: int
    predicted_class: str
    confidence: float


@dataclass
class EvaluationResults:
    """Structured evaluation results."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    top1_accuracy: float
    per_class_accuracy: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    roc_auc: Optional[Dict[str, float]] = None
    pr_auc: Optional[Dict[str, float]] = None
    misclassified: List[MisclassifiedSample] = field(default_factory=list)
    num_samples: int = 0

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "top1_accuracy": self.top1_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "per_class_accuracy": self.per_class_accuracy,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "misclassified": [
                {
                    "image_path": m.image_path,
                    "true_label": m.true_label,
                    "true_class": m.true_class,
                    "predicted_label": m.predicted_label,
                    "predicted_class": m.predicted_class,
                    "confidence": m.confidence,
                }
                for m in self.misclassified
            ],
        }

    def save(self, output_dir: Path) -> List[Path]:
        """Save metrics to JSON.

        Args:
            output_dir: Directory to save results into.

        Returns:
            List of file paths written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved evaluation metrics to %s", metrics_path)
        return [metrics_path]


class Evaluator:
    """Evaluates a trained model on a test set.

    Args:
        model: The trained model (in eval mode).
        test_loader: Test DataLoader.
        device: torch.device.
        class_names: List of class names (index-aligned with labels).
        output_dir: Optional directory for saving all plots.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        class_names: List[str],
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.output_dir = Path(output_dir) if output_dir else None

        # Cached raw arrays from the most recent evaluate() call.
        self._last_y_true: Optional[np.ndarray] = None
        self._last_y_prob: Optional[np.ndarray] = None

    @torch.no_grad()
    def evaluate(self) -> EvaluationResults:
        """Run evaluation and compute all metrics.

        Returns:
            An :class:`EvaluationResults` dataclass.
        """
        self.model.eval()

        y_true: List[int] = []
        y_pred: List[int] = []
        y_prob: List[np.ndarray] = []
        image_paths: List[str] = []

        # Track the starting index for image path collection.
        start_idx = 0
        dataset = self.test_loader.dataset
        has_paths = hasattr(dataset, "image_paths")

        for images, labels in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy())

            # Collect image paths from the dataset if available.
            if has_paths:
                for idx in range(labels.size(0)):
                    image_paths.append(str(dataset.image_paths[start_idx + idx]))

            start_idx += labels.size(0)

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_prob_arr = np.array(y_prob)

        # Cache for save_all() so plots can be rendered without re-inference.
        self._last_y_true = y_true_arr
        self._last_y_prob = y_prob_arr

        # Core metrics.
        accuracy = accuracy_score(y_true_arr, y_pred_arr)
        top1 = top_k_accuracy(y_true_arr, y_prob_arr, k=1)
        precision = precision_score(
            y_true_arr, y_pred_arr, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_true_arr, y_pred_arr, average="weighted", zero_division=0
        )
        f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

        # Per-class metrics.
        per_class_acc = per_class_accuracy(y_true_arr, y_pred_arr, self.class_names)
        per_class_precision = {
            self.class_names[i]: float(v)
            for i, v in enumerate(
                precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
            )
        }
        per_class_recall = {
            self.class_names[i]: float(v)
            for i, v in enumerate(
                recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
            )
        }
        per_class_f1 = {
            self.class_names[i]: float(v)
            for i, v in enumerate(
                f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
            )
        }

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        report = classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=self.class_names,
            zero_division=0,
        )

        # ROC / PR AUC (one-vs-rest) — only if >1 class.
        roc_auc: Optional[Dict[str, float]] = None
        pr_auc: Optional[Dict[str, float]] = None
        if len(self.class_names) > 1:
            roc_auc = ovr_roc_auc(y_true_arr, y_prob_arr, self.class_names)
            pr_auc = ovr_pr_auc(y_true_arr, y_prob_arr, self.class_names)

        # Misclassified samples.
        misclassified: List[MisclassifiedSample] = []
        for i in range(len(y_true_arr)):
            if y_true_arr[i] != y_pred_arr[i]:
                misclassified.append(
                    MisclassifiedSample(
                        image_path=image_paths[i] if i < len(image_paths) else "unknown",
                        true_label=int(y_true_arr[i]),
                        true_class=self.class_names[y_true_arr[i]],
                        predicted_label=int(y_pred_arr[i]),
                        predicted_class=self.class_names[y_pred_arr[i]],
                        confidence=float(y_prob_arr[i, y_pred_arr[i]]),
                    )
                )

        logger.info(
            "Evaluation complete: acc=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, "
            "misclassified=%d/%d",
            accuracy,
            precision,
            recall,
            f1,
            len(misclassified),
            len(y_true_arr),
        )

        results = EvaluationResults(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            top1_accuracy=float(top1),
            per_class_accuracy=per_class_acc,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            confusion_matrix=cm,
            classification_report=report,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            misclassified=misclassified,
            num_samples=len(y_true_arr),
        )

        # Save all plots if an output directory was provided.
        if self.output_dir is not None:
            self._save_plots(results, y_true_arr, y_prob_arr)

        return results

    # ------------------------------------------------------------------
    # Plotting / export
    # ------------------------------------------------------------------

    def _save_plots(
        self,
        results: EvaluationResults,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> List[Path]:
        """Save all evaluation plots.

        Args:
            results: Computed evaluation results.
            y_true: True label array.
            y_prob: Probability array.

        Returns:
            List of written paths.
        """
        if self.output_dir is None:
            return []

        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []

        saved.append(
            plot_confusion_matrix(
                results.confusion_matrix,
                self.class_names,
                output_dir / "confusion_matrix.png",
            )
        )

        if len(self.class_names) > 1:
            saved.append(
                plot_roc_curves(
                    y_true, y_prob, self.class_names, output_dir / "roc_curves.png"
                )
            )
            saved.append(
                plot_pr_curves(
                    y_true, y_prob, self.class_names, output_dir / "pr_curves.png"
                )
            )

        saved.append(
            plot_confidence_distribution(
                y_prob, output_dir / "confidence_distribution.png"
            )
        )

        if results.misclassified:
            saved.append(
                plot_misclassified_grid(
                    [m.image_path for m in results.misclassified],
                    [m.true_label for m in results.misclassified],
                    [m.predicted_label for m in results.misclassified],
                    [m.confidence for m in results.misclassified],
                    self.class_names,
                    output_dir / "misclassified.png",
                )
            )

        return saved

    def save_all(self, results: EvaluationResults) -> List[Path]:
        """Save the metrics JSON and all plots.

        Must be called after :meth:`evaluate`, which caches the prediction
        arrays needed for plotting.

        Args:
            results: The EvaluationResults to save.

        Returns:
            List of written paths.

        Raises:
            RuntimeError: If ``evaluate`` was not called first.
        """
        if self._last_y_true is None or self._last_y_prob is None:
            raise RuntimeError(
                "save_all() requires evaluate() to be called first."
            )

        logger.info("Saving evaluation artifacts...")
        output_dir = self.output_dir or Path("models/metrics")
        saved = results.save(output_dir)
        saved.extend(self._save_plots(results, self._last_y_true, self._last_y_prob))
        return saved

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_results(self, results: EvaluationResults) -> None:
        """Log a human-readable summary of the results.

        Args:
            results: The EvaluationResults to print.
        """
        lines = [
            "=" * 60,
            "Evaluation Results",
            "=" * 60,
            f"Samples     : {results.num_samples}",
            f"Top-1 Acc   : {results.accuracy:.4f}",
            f"Precision   : {results.precision:.4f}",
            f"Recall      : {results.recall:.4f}",
            f"F1 Score    : {results.f1:.4f}",
            "",
            "Per-class metrics:",
        ]
        for cls in self.class_names:
            lines.append(
                f"  {cls:20s} Acc={results.per_class_accuracy[cls]:.4f} "
                f"P={results.per_class_precision[cls]:.4f} "
                f"R={results.per_class_recall[cls]:.4f} "
                f"F1={results.per_class_f1[cls]:.4f}"
            )
        lines.append("")

        if results.roc_auc:
            lines.append("ROC AUC (one-vs-rest):")
            for cls, val in results.roc_auc.items():
                lines.append(f"  {cls:20s} {val:.4f}")
            lines.append("")

        if results.pr_auc:
            lines.append("PR AUC (one-vs-rest):")
            for cls, val in results.pr_auc.items():
                lines.append(f"  {cls:20s} {val:.4f}")
            lines.append("")

        lines.append("Classification Report")
        lines.append(results.classification_report)
        lines.append("")
        lines.append("Confusion Matrix")
        lines.append(str(results.confusion_matrix))
        lines.append("")

        if results.misclassified:
            lines.append(f"Misclassified ({len(results.misclassified)}):")
            for m in results.misclassified[:10]:  # Show first 10.
                lines.append(
                    f"  {m.image_path} | true={m.true_class} "
                    f"pred={m.predicted_class} conf={m.confidence:.3f}"
                )
            if len(results.misclassified) > 10:
                lines.append(f"  ... and {len(results.misclassified) - 10} more.")

        logger.info("\n".join(lines))