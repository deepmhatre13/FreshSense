"""EfficientNet-B0 model for the FreshSense pipeline.

Wraps torchvision's EfficientNet-B0 with transfer learning support:

- Pretrained ImageNet weights (``IMAGENET1K_V1`` for EfficientNet-B0).
- Optional backbone freezing / unfreezing.
- Configurable dropout and classifier head (single or two-layer).
- Parameter accounting (trainable / frozen / total).
- Parameter groups for differential learning rates.
- Optional torchinfo summary.

Design notes:

- The ``freeze_backbone`` **setting** is stored on ``self.freeze_backbone_``
  so it never shadows the :meth:`freeze_backbone` **method** (a classic
  bool/function name clash that silently breaks training).
- Freezing is applied to ``model.features.parameters()`` only; the classifier
  head always stays trainable.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger(__name__)

__all__ = ["FreshSenseEfficientNet"]


class FreshSenseEfficientNet(nn.Module):
    """EfficientNet-B0 classifier for freshness detection.

    Args:
        num_classes: Number of output classes.
        pretrained: If True, load ImageNet-pretrained weights.
        freeze_backbone: If True, freeze all backbone (features) parameters.
        dropout: Dropout probability in the classifier head.
        classifier_hidden: Optional hidden layer size for a two-layer head.
            If ``None``, a single linear layer is used.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        classifier_hidden: Optional[int] = 256,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1].")
        if classifier_hidden is not None and classifier_hidden <= 0:
            raise ValueError("classifier_hidden must be positive or None.")

        self.num_classes = num_classes
        self.pretrained = pretrained
        # NOTE: trailing underscore avoids shadowing the freeze_backbone() method.
        self.freeze_backbone_ = freeze_backbone
        self.dropout = dropout
        self.classifier_hidden = classifier_hidden

        # Load the backbone (with or without pretrained weights).
        # EfficientNet-B0's DEFAULT weights are IMAGENET1K_V1.
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        # Replace the classifier head BEFORE freezing so the head is never
        # touched by the freeze loop (it must always be trainable).
        in_features = self.model.classifier[1].in_features
        self.model.classifier = self._build_classifier(in_features)

        # Freeze backbone if requested.
        if self.freeze_backbone_:
            self.freeze_backbone()

    def _build_classifier(self, in_features: int) -> nn.Sequential:
        """Build the classifier head.

        Args:
            in_features: Number of input features from the backbone.

        Returns:
            A Sequential classifier head.
        """
        layers: List[nn.Module] = [nn.Dropout(self.dropout)]

        if self.classifier_hidden is not None:
            layers.append(nn.Linear(in_features, self.classifier_hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.classifier_hidden, self.num_classes))
        else:
            layers.append(nn.Linear(in_features, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)`` float32 in [0, 1].

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        return self.model(x)

    # ------------------------------------------------------------------
    # Freezing / unfreezing
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> "FreshSenseEfficientNet":
        """Freeze all backbone (features) parameters.

        Returns:
            self (for chaining).
        """
        for param in self.model.features.parameters():
            param.requires_grad = False
        logger.info(
            "Backbone frozen: %d trainable params remaining.",
            self.trainable_parameters(),
        )
        return self

    def unfreeze_backbone(self) -> "FreshSenseEfficientNet":
        """Unfreeze all backbone (features) parameters.

        Returns:
            self (for chaining).
        """
        for param in self.model.features.parameters():
            param.requires_grad = True
        self.freeze_backbone_ = False
        logger.info(
            "Backbone unfrozen: %d trainable params.",
            self.trainable_parameters(),
        )
        return self

    # ------------------------------------------------------------------
    # Parameter accounting
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        """Return the number of frozen (non-trainable) parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def total_parameters(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_parameter_groups(
        self,
        backbone_lr: float,
        classifier_lr: float,
        weight_decay: float,
    ) -> List[dict]:
        """Return parameter groups for differential learning rates.

        This is a transfer-learning best practice: the (possibly frozen)
        backbone gets a lower LR than the freshly-initialized classifier.

        Args:
            backbone_lr: Learning rate for backbone parameters.
            classifier_lr: Learning rate for classifier parameters.
            weight_decay: Weight decay for all parameters.

        Returns:
            A list of optimizer parameter groups.
        """
        backbone_params = [
            p for p in self.model.features.parameters() if p.requires_grad
        ]
        classifier_params = [
            p for p in self.model.classifier.parameters() if p.requires_grad
        ]

        groups: List[dict] = []
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                }
            )
        if classifier_params:
            groups.append(
                {
                    "params": classifier_params,
                    "lr": classifier_lr,
                    "weight_decay": weight_decay,
                }
            )
        return groups

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, input_size: Tuple[int, int, int] = (3, 224, 224)) -> str:
        """Return a model summary string.

        Uses torchinfo if available; otherwise falls back to a simple
        parameter count.

        Args:
            input_size: Input tensor shape ``(C, H, W)``.

        Returns:
            A human-readable model summary.
        """
        try:
            from torchinfo import summary as torchinfo_summary

            return str(
                torchinfo_summary(
                    self,
                    input_size=(1, *input_size),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    verbose=0,
                )
            )
        except ImportError:
            return (
                f"FreshSenseEfficientNet(num_classes={self.num_classes}, "
                f"pretrained={self.pretrained}, freeze_backbone={self.freeze_backbone_})\n"
                f"Trainable: {self.trainable_parameters():,} | "
                f"Frozen: {self.frozen_parameters():,} | "
                f"Total: {self.total_parameters():,}"
            )

    def __repr__(self) -> str:
        return (
            f"FreshSenseEfficientNet(num_classes={self.num_classes}, "
            f"pretrained={self.pretrained}, freeze_backbone={self.freeze_backbone_}, "
            f"dropout={self.dropout}, classifier_hidden={self.classifier_hidden})"
        )


if __name__ == "__main__":
    model = FreshSenseEfficientNet(
        num_classes=3,
        pretrained=True,
        freeze_backbone=True,
    )

    print("=" * 60)
    print("FreshSense EfficientNet-B0")
    print("=" * 60)
    print(repr(model))
    print()
    print(model.summary())
    print()
    print(f"Trainable Parameters : {model.trainable_parameters():,}")
    print(f"Frozen Parameters    : {model.frozen_parameters():,}")
    print(f"Total Parameters     : {model.total_parameters():,}")

    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print()
    print("Output Shape :", tuple(output.shape))