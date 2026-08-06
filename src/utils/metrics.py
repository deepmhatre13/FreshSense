"""Metrics helpers for the FreshSense evaluation pipeline.

Pure functions that compute classification metrics from arrays of true
labels, predicted labels, and predicted probabilities. Kept dependency-light
so they can be unit-tested in isolation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

__all__ = [
    "top_k_accuracy",
    "per_class_accuracy",
    "confidence_distribution",
    "ovr_roc_auc",
    "ovr_pr_auc",
]


def top_k_accuracy(
    y_true: np.ndarray, y_prob: np.ndarray, k: int = 1
) -> float:
    """Return the top-k accuracy.

    Args:
        y_true: 1D array of true integer labels.
        y_prob: 2D array of class probabilities (shape ``(N, C)``).
        k: Number of top predictions to consider.

    Returns:
        Fraction of samples whose true label is in the top-k predictions.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if k == 1:
        return float(accuracy_score(y_true, np.argmax(y_prob, axis=1)))

    top_k_indices = np.argsort(y_prob, axis=1)[:, -k:][:, ::-1]
    correct = sum(1 for i in range(len(y_true)) if y_true[i] in top_k_indices[i])
    return float(correct / max(len(y_true), 1))


def per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Dict[str, float]:
    """Return per-class accuracy as ``{class_name: accuracy}``.

    Args:
        y_true: 1D array of true integer labels.
        y_pred: 1D array of predicted integer labels.
        class_names: Class names index-aligned with labels.

    Returns:
        Dict mapping each class name to its accuracy.
    """
    result: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            result[name] = 0.0
        else:
            result[name] = float((y_pred[mask] == i).mean())
    return result


def confidence_distribution(
    y_prob: np.ndarray, num_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a histogram of maximum softmax confidence.

    Args:
        y_prob: 2D array of class probabilities (shape ``(N, C)``).
        num_bins: Number of histogram bins.

    Returns:
        ``(hist, bin_edges)`` as returned by ``np.histogram``.
    """
    confidences = np.max(y_prob, axis=1)
    hist, bin_edges = np.histogram(
        confidences, bins=num_bins, range=(0.0, 1.0)
    )
    return hist, bin_edges


def ovr_roc_auc(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]
) -> Dict[str, float]:
    """Compute one-vs-rest ROC AUC per class.

    Args:
        y_true: 1D array of true integer labels.
        y_prob: 2D array of class probabilities.
        class_names: Class names index-aligned with labels.

    Returns:
        ``{class_name: roc_auc}``. A value of 0.0 means the metric could not
        be computed (e.g. only one class present in the true labels).
    """
    result: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        try:
            result[name] = float(roc_auc_score(binary_true, y_prob[:, i]))
        except ValueError:
            # Single-class fold: AUC is undefined.
            result[name] = 0.0
    return result


def ovr_pr_auc(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]
) -> Dict[str, float]:
    """Compute one-vs-rest precision-recall AUC per class.

    Args:
        y_true: 1D array of true integer labels.
        y_prob: 2D array of class probabilities.
        class_names: Class names index-aligned with labels.

    Returns:
        ``{class_name: pr_auc}``.
    """
    result: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, y_prob[:, i])
        result[name] = float(auc(recall, precision))
    return result