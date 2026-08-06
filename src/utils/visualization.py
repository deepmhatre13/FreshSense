"""Plotting helpers for the FreshSense evaluation pipeline.

All functions use the non-interactive ``Agg`` backend so they are safe in
headless environments (CI, Docker, remote training). Each function writes a
PNG to disk and returns the path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_confidence_distribution",
    "plot_misclassified_grid",
]


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    """Save a figure and close it to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Render and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2D array).
        class_names: Class names index-aligned with rows/columns.
        path: Destination PNG path.
        title: Plot title.

    Returns:
        The saved path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells.
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return _save_fig(fig, path)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    path: Path,
) -> Path:
    """Plot one-vs-rest ROC curves for each class.

    Args:
        y_true: 1D array of true integer labels.
        y_prob: 2D array of class probabilities.
        class_names: Class names index-aligned with labels.
        path: Destination PNG path.

    Returns:
        The saved path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
        ax.plot(fpr, tpr, label=name)

    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, path)


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    path: Path,
) -> Path:
    """Plot one-vs-rest precision-recall curves for each class.

    Args:
        y_true: 1D array of true integer labels.
        y_prob: 2D array of class probabilities.
        class_names: Class names index-aligned with labels.
        path: Destination PNG path.

    Returns:
        The saved path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, y_prob[:, i])
        ax.plot(recall, precision, label=name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, path)


def plot_confidence_distribution(
    y_prob: np.ndarray,
    path: Path,
    num_bins: int = 20,
) -> Path:
    """Plot a histogram of maximum softmax confidence.

    Args:
        y_prob: 2D array of class probabilities.
        path: Destination PNG path.
        num_bins: Number of histogram bins.

    Returns:
        The saved path.
    """
    confidences = np.max(y_prob, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=num_bins, range=(0.0, 1.0), color="steelblue", edgecolor="white")
    ax.set_xlabel("Max Softmax Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, path)


def plot_misclassified_grid(
    image_paths: List[str],
    true_labels: List[int],
    pred_labels: List[int],
    confidences: List[float],
    class_names: List[str],
    path: Path,
    max_images: int = 16,
) -> Path:
    """Render a grid of misclassified images with true/pred labels.

    Args:
        image_paths: Paths to the misclassified images.
        true_labels: True integer labels.
        pred_labels: Predicted integer labels.
        confidences: Prediction confidences.
        class_names: Class names index-aligned with labels.
        path: Destination PNG path.
        max_images: Maximum number of images to show.

    Returns:
        The saved path.
    """
    import cv2

    n = min(len(image_paths), max_images)
    if n == 0:
        logger.info("No misclassified images to plot.")
        return path

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(n):
        ax = axes[idx]
        # Load the image if it exists and is readable; otherwise show a
        # placeholder (path collection may not be available for every dataset).
        img = cv2.imread(image_paths[idx]) if Path(image_paths[idx]).exists() else None
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(
                0.5,
                0.5,
                "unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
        ax.set_title(
            f"T:{class_names[true_labels[idx]]}\n"
            f"P:{class_names[pred_labels[idx]]} ({confidences[idx]:.2f})",
            fontsize=8,
        )
        ax.axis("off")

    # Hide unused axes.
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Misclassified Images", fontsize=14)
    fig.tight_layout()
    return _save_fig(fig, path)