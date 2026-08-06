"""Albumentations-based augmentation for the FreshSense pipeline.

Written for **Albumentations 2.0.8** (verified against the installed
signatures at write time). Key 2.x API changes reflected here:

- ``RandomResizedCrop`` now takes ``size=(height, width)`` (the old
  ``height``/``width`` keyword arguments were removed).
- ``GaussNoise`` now uses ``std_range`` (``var_limit`` was removed).
- ``Affine`` uses ``border_mode`` / ``fill`` (the old ``mode``/``cval``
  aliases were removed).

Augmentation strategy for freshness classification:

- ``RandomResizedCrop``: framing/zoom variation. The single most effective
  augmentation for CNNs; forces robustness to object scale and position.
- ``HorizontalFlip``: mirroring a photo of produce is realistic.
- ``Affine``: small shifts/scales/rotations simulate camera angle and
  framing variation. ``border_mode=cv2.BORDER_REFLECT_101`` avoids black
  borders (which, after ImageNet normalization, inject spurious dark edges).
- ``RandomBrightnessContrast``: lighting variation.
- ``HueSaturationValue``: small color-cast variation (kept small so color
  cues are preserved).
- ``GaussianBlur`` (light, p=0.2): mild focus variation. Kept VERY light so
  texture cues (wrinkles, spots, mold) remain visible.
- ``GaussNoise`` (light, p=0.2): mild sensor noise. Kept VERY light so it
  does not destroy the texture signal.

Validation/test are deterministic: ``Resize`` (matching the preprocessor
size), ImageNet ``Normalize``, ``ToTensorV2``.

Normalization uses ImageNet mean/std to match pretrained EfficientNet-B0.
"""

from __future__ import annotations

from typing import Tuple

import cv2
from albumentations import (
    Affine,
    Compose,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    RandomResizedCrop,
    Resize,
)
from albumentations.pytorch import ToTensorV2

__all__ = ["AugmentationPipeline"]

# ImageNet statistics for pretrained EfficientNet-B0.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AugmentationPipeline:
    """Builds train/validation/test transform pipelines.

    Args:
        image_size: Target image size (width, height). Used as the
            ``size`` argument for ``RandomResizedCrop`` and the ``height`` /
            ``width`` arguments for ``Resize``.
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224)) -> None:
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size must be positive.")
        self.image_size = tuple(int(v) for v in image_size)

    def train_transforms(self) -> Compose:
        """Return the training augmentation pipeline.

        Transforms and rationale:

        - ``RandomResizedCrop``: scale/zoom + crop variation (2.0.8 API:
          ``size=(height, width)``).
        - ``HorizontalFlip``: mirroring produce photos is realistic.
        - ``Affine``: small camera-angle / framing variation.
        - ``RandomBrightnessContrast``: lighting variation.
        - ``HueSaturationValue``: small color-cast variation.
        - ``GaussianBlur`` / ``GaussNoise``: light texture/sensor noise (kept
          mild so freshness cues are preserved).
        - ``Normalize`` + ``ToTensorV2``: ImageNet normalization + CHW tensor.
        """
        return Compose(
            [
                RandomResizedCrop(
                    size=(
                        self.image_size[1],
                        self.image_size[0],
                    ),
                    scale=(0.7, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),
                HorizontalFlip(p=0.5),
                Affine(
                    scale=(0.90, 1.10),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-15, 15),
                    border_mode=cv2.BORDER_REFLECT_101,
                    fit_output=False,
                    keep_ratio=False,
                    p=0.5,
                ),
                RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.4,
                ),
                GaussianBlur(
                    blur_limit=(3, 5),
                    sigma_limit=0.5,
                    p=0.2,
                ),
                GaussNoise(
                    std_range=(0.01, 0.02),
                    mean_range=(0.0, 0.0),
                    per_channel=False,
                    p=0.2,
                ),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    def validation_transforms(self) -> Compose:
        """Return the validation pipeline (Resize + normalize only)."""
        return Compose(
            [
                Resize(
                    height=self.image_size[1],
                    width=self.image_size[0],
                ),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    def test_transforms(self) -> Compose:
        """Return the test pipeline (identical to validation)."""
        return self.validation_transforms()

    def get_transforms(self, split: str) -> Compose:
        """Return the transform pipeline for a given split.

        Args:
            split: One of "train", "val", "test".

        Returns:
            The corresponding Compose pipeline.

        Raises:
            ValueError: If split is not recognized.
        """
        split = split.lower()
        if split == "train":
            return self.train_transforms()
        if split in ("val", "validation"):
            return self.validation_transforms()
        if split == "test":
            return self.test_transforms()
        raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")


if __name__ == "__main__":
    pipeline = AugmentationPipeline()
    for split in ("train", "val", "test"):
        transforms = pipeline.get_transforms(split)
        print(f"--- {split.upper()} ---")
        print(transforms)
        print()