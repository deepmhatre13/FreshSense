"""OpenCV-based image preprocessing for the FreshSense pipeline.

The :class:`ImagePreprocessor` performs *minimal* geometric/color preprocessing
that must happen before augmentation:

1. Validate the input image (3-channel BGR array).
2. Resize to a fixed size (so batching is consistent with the model input).
3. Convert BGR -> RGB (OpenCV loads BGR; Albumentations/PyTorch expect RGB).
4. Optionally apply CLAHE on the L channel of LAB (off by default).

Normalization (ImageNet mean/std) is intentionally NOT done here. It belongs
in the Albumentations pipeline so it is applied consistently to train, val,
and test sets, and matches the pretrained EfficientNet-B0 weights.

Why minimal preprocessing?

- Heavy filtering (denoising, sharpening, contrast stretching) destroys the
  texture and color cues that distinguish fresh vs. stale vs. rotten produce.
- CLAHE is optional and OFF by default because it can wash out the subtle
  color differences that matter most for this task.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

__all__ = ["ImagePreprocessor"]


class ImagePreprocessor:
    """Resize and color-convert images for the FreshSense pipeline.

    Args:
        image_size: Target ``(width, height)`` in pixels.
        use_clahe: If True, apply CLAHE on the L channel of LAB.
        clahe_clip_limit: CLAHE contrast clip limit.
        clahe_tile_grid_size: CLAHE tile grid size.

    Raises:
        ValueError: If ``image_size`` is non-positive or the clip limit <= 0.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size must be positive.")
        if clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive.")

        self.image_size = tuple(int(v) for v in image_size)
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        # NOTE: The CLAHE object is created lazily in apply_clahe() rather
        # than here. cv2.CLAHE is NOT picklable, and holding it as an
        # attribute breaks DataLoader multiprocessing on Windows (spawn),
        # which pickles the dataset (and thus the preprocessor) for each
        # worker. Lazy creation keeps this object picklable.
        self._clahe = None

    def _validate_input(self, image: np.ndarray) -> None:
        """Validate that the input is a 3-channel BGR uint8 image.

        Args:
            image: Image as loaded by ``cv2.imread``.

        Raises:
            ValueError: If the image is None, not 3-channel, or unexpected dtype.
        """
        if image is None:
            raise ValueError("Input image is None.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR image, got shape {image.shape}."
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 image, got dtype {image.dtype}. "
                "Normalization is handled by Albumentations; pass raw uint8."
            )

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize the image to the configured size if needed.

        Uses ``INTER_AREA`` when downscaling (smoother, fewer aliasing
        artifacts) and ``INTER_LINEAR`` when upscaling (better quality for
        small images). If the image is already the target size, no copy is
        made.

        Args:
            image: A BGR or RGB uint8 image.

        Returns:
            The resized image.
        """
        height, width = image.shape[:2]
        target_width, target_height = self.image_size

        if width == target_width and height == target_height:
            return image

        # INTER_AREA is best for shrinking; INTER_LINEAR for enlarging.
        interpolation = (
            cv2.INTER_AREA
            if width > target_width or height > target_height
            else cv2.INTER_LINEAR
        )
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=interpolation,
        )

    def bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR image (OpenCV default) to RGB.

        Args:
            image: BGR uint8 image.

        Returns:
            RGB uint8 image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of the LAB color space.

        CLAHE enhances local contrast while limiting noise amplification.
        It is applied only to the L (lightness) channel so that color
        information (a/b channels) is preserved — critical for freshness
        classification where color is a primary cue.

        Args:
            image: An RGB image.

        Returns:
            The CLAHE-enhanced RGB image.
        """
        # Create the CLAHE object on first use (it is stateless per-apply).
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size,
            )

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self._clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing chain on a BGR image.

        Args:
            image: A BGR image as loaded by ``cv2.imread`` (uint8).

        Returns:
            A resized RGB image (uint8, [0, 255]) ready for augmentation.

        Raises:
            ValueError: If the input is not a valid 3-channel uint8 BGR image.
        """
        self._validate_input(image)
        image = self.resize(image)
        image = self.bgr_to_rgb(image)
        if self.use_clahe:
            image = self.apply_clahe(image)
        return image


if __name__ == "__main__":
    # Quick self-test.
    sample = cv2.imread("sample.jpg")
    if sample is None:
        print("No sample.jpg found; skipping self-test.")
    else:
        processor = ImagePreprocessor()
        processed = processor.preprocess(sample)
        print(f"Input shape  : {sample.shape}")
        print(f"Output shape : {processed.shape}")
        print(f"Dtype        : {processed.dtype}")
        print(f"Value range  : [{processed.min()}, {processed.max()}]")