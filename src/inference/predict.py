"""Inference pipeline for single-image prediction.

The :class:`Predictor` mirrors the training preprocessing exactly:

    cv2.imread (BGR)
      -> ImagePreprocessor (resize + BGR->RGB)
      -> Albumentations Normalize (ImageNet) + ToTensorV2
      -> EfficientNet-B0 forward
      -> softmax probabilities

Because inference must match training transforms, normalization uses the
same ImageNet mean/std as ``AugmentationPipeline`` (no augmentation during
inference, matching the val/test pipeline).

The checkpoint is loaded by ``model_state_dict`` key, which is how the
:class:`src.training.trainer.Trainer` saves checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch

from configs.config import Config
from src.models.efficientnet import FreshSenseEfficientNet
from src.preprocessing.preprocess import ImagePreprocessor

logger = logging.getLogger(__name__)

__all__ = ["Predictor"]

# ImageNet statistics matching the training pipeline.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Predictor:
    """Loads a trained model and predicts on a single image.

    Args:
        model_path: Path to a trainer checkpoint (dict with ``model_state_dict``).
        class_names: Ordered class names aligned with training labels.
        image_size: Target input size (width, height).
        device: torch.device to run on.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError: If the checkpoint is malformed or lacks state dict.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        class_names: Optional[Sequence[str]] = None,
        image_size: int = 224,
        device: torch.device | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.device = device or Config().device
        self.preprocessor = ImagePreprocessor(
            image_size=(image_size, image_size)
        )

        # Load the checkpoint first so we can recover class names if the
        # caller did not provide them (production best practice: the label
        # order lives with the model, not in external config).
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if class_names is not None:
            self.class_names = list(class_names)
        elif "class_names" in checkpoint:
            self.class_names = list(checkpoint["class_names"])
        else:
            raise ValueError(
                "class_names must be provided because the checkpoint does not "
                "store them (older checkpoint format)."
            )

        self.num_classes = len(self.class_names)
        if self.num_classes == 0:
            raise ValueError("class_names must contain at least one class.")

        # Build model with matching architecture. pretrained=False because we
        # load our own weights. freeze_backbone=False is irrelevant at eval.
        self.model = FreshSenseEfficientNet(
            num_classes=self.num_classes,
            pretrained=False,
            freeze_backbone=False,
        )

        # Load state dict from the trainer's checkpoint wrapper.
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and all(
            k.startswith("model.") or k.startswith("features.")
            for k in checkpoint.keys()
        ):
            # Bare state dict (e.g. exported model).
            state_dict = checkpoint
        else:
            raise RuntimeError(
                f"Unrecognized checkpoint format: {model_path}. "
                "Expected a trainer checkpoint with 'model_state_dict'."
            )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Predictor ready: %d classes, device=%s, checkpoint=%s",
            self.num_classes,
            self.device,
            model_path,
        )

    # ------------------------------------------------------------------
    # Preprocessing (mirrors training pipeline exactly)
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a BGR image into a normalized model-ready tensor.

        Args:
            image: BGR uint8 image as returned by ``cv2.imread``.

        Returns:
            Tensor of shape ``(1, 3, H, W)`` float32, normalized.
        """
        # Resize + BGR->RGB (uint8 [0, 255]).
        image_rgb = self.preprocessor.preprocess(image)

        # Normalize with ImageNet stats (matches training).
        image_f = image_rgb.astype(np.float32) / 255.0
        image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> CHW -> (1, C, H, W).
        tensor = torch.from_numpy(
            np.transpose(image_f, (2, 0, 1))
        ).unsqueeze(0)
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, image_path: Union[str, Path]
    ) -> Dict[str, Union[str, float, List[float]]]:
        """Predict the class of a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with keys ``class``, ``confidence``, and ``probabilities``
            (ordered per ``class_names``).

        Raises:
            ValueError: If the image cannot be read.
            FileNotFoundError: If the file does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")

        tensor = self._preprocess(image)
        outputs = self.model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        confidence, prediction = torch.max(probabilities, dim=0)
        idx = int(prediction.item())

        result = {
            "class": self.class_names[idx],
            "confidence": float(confidence.item()),
            "probabilities": probabilities.cpu().numpy().tolist(),
        }
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    classes = ["fresh", "stale", "rotten"]

    predictor = Predictor(
        model_path="models/checkpoints/best_model.pth",
        class_names=classes,
    )

    result = predictor.predict("sample.jpg")
    print(result)