"""Grad-CAM explainability for FreshSense Phase 4.

Produces a class-activation heatmap on the classified crop to highlight which
regions drove the prediction (e.g. bruised/damaged spots).

Targets the last convolutional layer of the EfficientNet backbone.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["GradCAM"]


def _find_target_layer(model: torch.nn.Module):
    """Locate the last convolutional module in the backbone."""
    features = getattr(model, "model", model).features
    target = None
    for name, module in features.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target = module
    if target is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return target


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Args:
        model: A ``FreshSenseEfficientNet`` (or any module exposing ``.model.features``).
        target_layer: Optional layer to hook. Defaults to the last Conv2d.
        device: Optional torch device. Auto-detected when None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.target_layer = target_layer or _find_target_layer(model)

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for an input batch.

        Args:
            input_tensor: Preprocessed tensor of shape (1, C, H, W).
            class_idx: Target class index; defaults to the argmax class.

        Returns:
            Normalised heatmap as a float32 array of shape (H, W) in [0, 1].
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())

        self.model.zero_grad()
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1,1,H,W)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1].
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay(
        self,
        image_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay the heatmap onto a BGR crop.

        Args:
            image_bgr: BGR crop image (already resized to heatmap size).
            heatmap: Heatmap in [0, 1] matching image height/width.
            alpha: Blend strength.

        Returns:
            Blended BGR image.
        """
        heatmap_resized = cv2.resize(
            heatmap, (image_bgr.shape[1], image_bgr.shape[0])
        )
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
        return blended

    def close(self) -> None:
        """Remove registered hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()
