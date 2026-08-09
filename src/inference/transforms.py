"""Deterministic inference preprocessing for FreshSense Phase 2.

Inference must NOT depend on training augmentations (no RandomCrop,
ColorJitter, Blur, CoarseDropout, etc.). This module provides an independent,
deterministic preprocessing pipeline used by :class:`~src.inference.predictor.Predictor`
so that the real-time inference layer stays lightweight and only relies on
numpy / OpenCV / PyTorch.

Pipeline (matches the model's expected 224x224 ImageNet-normalized input):

    BGR frame
      -> validate (uint8, 3-channel)
      -> BGR -> RGB
      -> Resize
      -> CenterCrop
      -> /255 -> float32
      -> ImageNet Normalize
      -> HWC -> CHW -> (1, 3, H, W) tensor

Normalization uses the same ImageNet mean/std as the Phase 1 training pipeline
so that inference inputs are distributionally identical to validation images.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["InferenceTransform"]

# ImageNet statistics (matches training / pretrained EfficientNet-B0).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InferenceTransform:
    """Deterministic transform that turns a BGR webcam frame into a model tensor.

    Only deterministic geometric/color transforms are applied. There is no
    random augmentation of any kind.

    Args:
        image_size: Final (square) model input size in pixels.
        resize_size: Target size for the resize step. Defaults to ``image_size``
            so the transform behaves as ``Resize(size) -> CenterCrop(size)``.
        center_crop: Crop size for the center-crop step. Defaults to
            ``image_size``. When ``resize_size > center_crop`` the center-crop
            trims a border; when equal the crop is a no-op.
        mean: RGB normalization mean.
        std: RGB normalization standard deviation.
    """

    def __init__(
        self,
        image_size: int = 224,
        resize_size: Optional[int] = None,
        center_crop: Optional[int] = None,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        self.image_size = int(image_size)
        self.resize_size = int(resize_size) if resize_size else self.image_size
        self.center_crop = int(center_crop) if center_crop else self.image_size

        if self.resize_size <= 0 or self.center_crop <= 0:
            raise ValueError("resize_size and center_crop must be positive.")
        if self.center_crop > self.resize_size:
            raise ValueError("center_crop cannot exceed resize_size.")

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    # ------------------------------------------------------------------
    # Preprocessing steps
    # ------------------------------------------------------------------

    def _validate(self, image: np.ndarray) -> None:
        """Validate a BGR webcam frame."""
        if image is None:
            raise ValueError("Input frame is None.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR image, got shape {image.shape}."
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 image, got dtype {image.dtype}."
            )

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to the configured square size."""
        height, width = image.shape[:2]
        target = self.resize_size
        if width == target and height == target:
            return image
        interpolation = (
            cv2.INTER_AREA
            if width > target or height > target
            else cv2.INTER_LINEAR
        )
        return cv2.resize(image, (target, target), interpolation=interpolation)

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        """Center-crop the image to the configured crop size."""
        if self.center_crop >= self.resize_size:
            return image
        start = (self.resize_size - self.center_crop) // 2
        end = start + self.center_crop
        return image[start:end, start:end]

    # ------------------------------------------------------------------
    # Main callable
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Transform a BGR frame into a normalized model tensor.

        Args:
            image: BGR uint8 frame (e.g. from ``cv2.VideoCapture.read``).

        Returns:
            ``torch.Tensor`` of shape ``(1, 3, H, W)`` float32 on CPU,
            normalized with ImageNet stats. Move it to the target device
            (``tensor.to(device)``) before inference.
        """
        self._validate(image)

        # BGR -> RGB (OpenCV loads BGR; PyTorch models expect RGB).
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Deterministic geometric transforms only.
        rgb = self._resize(rgb)
        rgb = self._center_crop(rgb)

        # To float [0,1] then normalize with ImageNet stats.
        img = rgb.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # HWC -> CHW -> add batch dimension.
        tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
        return tensor.contiguous()


if __name__ == "__main__":
    # Quick self-test on a synthetic frame.
    import time

    logging.basicConfig(level=logging.INFO)
    transform = InferenceTransform(image_size=224)

    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    start = time.perf_counter()
    out = transform(dummy)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"Input shape : {dummy.shape} (BGR uint8)")
    print(f"Output shape: {tuple(out.shape)} (float32, CHW, batched)")
    print(f"Dtype       : {out.dtype}")
    print(f"Value range : [{float(out.min()):.3f}, {float(out.max()):.3f}]")
    print(f"Transform   : {elapsed_ms:.2f} ms")
    print()
    print("Deterministic? (two calls equal):", bool((out == transform(dummy)).all()))

