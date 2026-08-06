"""Loss functions for the FreshSense pipeline.

Phase 1 uses ``CrossEntropyLoss`` with optional label smoothing, which is a
well-established production technique for classification: it regularises the
model against over-confidence and generally improves generalisation on
imbalanced/real-world datasets.

A factory function (:func:`build_criterion`) is provided so the choice of
loss comes from configuration rather than being hardcoded in the training
entry point.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["LabelSmoothingCrossEntropy", "build_criterion"]


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing.

    Args:
        num_classes: Number of classes.
        smoothing: Smoothing factor in ``[0, 1)``. ``0.0`` is equivalent to
            plain cross-entropy.

    Raises:
        ValueError: If ``smoothing`` is not in ``[0, 1)`` or classes <= 0.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.0) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")

        self.num_classes = num_classes
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the label-smoothed cross-entropy loss.

        Args:
            logits: Raw model logits of shape ``(N, C)``.
            targets: Integer class indices of shape ``(N,)``.

        Returns:
            Scalar loss.
        """
        log_probs = self.log_softmax(logits)

        # Build smoothed targets.
        smoothed = torch.full_like(log_probs, self.smoothing / self.num_classes)
        smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smoothed * log_probs).sum(dim=1).mean()
        return loss


def build_criterion(
    num_classes: int, label_smoothing: float = 0.0
) -> nn.Module:
    """Build the loss function based on configuration.

    Args:
        num_classes: Number of output classes.
        label_smoothing: Smoothing factor. ``0.0`` (default) gives plain
            ``CrossEntropyLoss``.

    Returns:
        A loss module.
    """
    if label_smoothing > 0.0:
        criterion: nn.Module = LabelSmoothingCrossEntropy(
            num_classes=num_classes, smoothing=label_smoothing
        )
        logger.info("Using LabelSmoothingCrossEntropy (smoothing=%.2f)", label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss")
    return criterion