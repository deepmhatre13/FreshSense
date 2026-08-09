"""Real-time prediction engine for FreshSense Phase 2.

This module provides the prediction engine for real-time inference:

- Loads the trained model from best_model.pth
- Automatic device detection (CPU/CUDA/MPS)
- Reuses existing preprocessing pipeline (no duplication)
- Structured PredictionResult output
- LangGraph-compatible to_dict() method
- Thread-safe inference

The Predictor class wraps the Phase 1 model and provides a clean API
for real-time inference without modifying any training code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from configs.config import Config
from src.inference.transforms import InferenceTransform
from src.models.efficientnet import FreshSenseEfficientNet

logger = logging.getLogger(__name__)

__all__ = ["Predictor", "PredictionResult"]


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction result for real-time inference.

    This dataclass provides a consistent output format for the inference
    pipeline and is designed to be compatible with LangGraph integration.

    Attributes:
        fruit_name: Detected fruit/vegetable name.
        freshness_class: Freshness classification (fresh/stale/rotten).
        confidence: Prediction confidence score (0.0-1.0).
        probabilities: Class probability distribution.
        timestamp: Inference timestamp (seconds since epoch).
        latency_ms: Inference time in milliseconds.
        device: Device used for inference (CPU/CUDA/MPS).
        model_version: Model version identifier.
        ready_for_langgraph: Flag indicating LangGraph compatibility.
    """

    fruit_name: str
    freshness_class: str
    confidence: float
    probabilities: List[float]
    timestamp: float
    latency_ms: float
    device: str
    model_version: str
    ready_for_langgraph: bool = True

    def to_dict(self) -> Dict[str, Union[str, float, List[float], bool]]:
        """Convert to dictionary for LangGraph compatibility.

        Returns:
            Dictionary with all fields in LangGraph-ready format.
        """
        return {
            "fruit": self.fruit_name,
            "freshness": self.freshness_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "image_id": f"frame_{int(self.timestamp * 1000)}",
            "session_id": self.model_version,
            "device": self.device,
            "model_version": self.model_version,
            "ready_for_langgraph": self.ready_for_langgraph,
        }


class Predictor:
    """Real-time prediction engine for FreshSense.

    This class provides:
    - Model loading from checkpoint
    - Automatic device detection
    - Reuses Phase 1 preprocessing (no duplication)
    - Optimized inference with torch.no_grad()
    - Batch size = 1 for minimal latency
    - Structured PredictionResult output

    Args:
        checkpoint_path: Path to best_model.pth.
        config: Config instance (optional, uses default if not provided).
        device: Override device (auto-detected if not specified).
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None,
        calibration_path: Optional[str] = None,
    ) -> None:
        self.config = config or Config.from_yaml("configs/settings.yaml")
        self.device = device or self._detect_device()

        # Load checkpoint
        self.checkpoint_path = checkpoint_path
        self.checkpoint = self._load_checkpoint(checkpoint_path)

        # Extract metadata
        self.class_names = list(self.checkpoint.get("class_names") or [])
        if not self.class_names:
            raise ValueError(
                f"Checkpoint {self.checkpoint_path} does not contain "
                "'class_names'. Cannot build the model without knowing the "
                "class ordering."
            )
        self.num_classes = len(self.class_names)
        self.model_version = self._extract_model_version()

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

        # Load temperature scaling if provided
        self.temperature = self._load_temperature(calibration_path)

        # Independent, deterministic inference preprocessing only.
        # NO training augmentations are used at inference time.
        self.image_size = self.config.data.image_size
        self.transform = InferenceTransform(image_size=self.image_size)

        logger.info(
            "Predictor initialized: %d classes, device=%s, model=%s",
            self.num_classes,
            self.device,
            self.model_version,
        )

    def _detect_device(self) -> torch.device:
        """Automatically detect the best available device.

        Priority: CUDA > MPS > CPU

        Returns:
            torch.device for inference.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("CUDA device detected: %s", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("MPS device detected (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for inference")
        return device

    def _load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint dictionary.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint format is invalid.
        """
        import pathlib

        checkpoint_path = pathlib.Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info("Loading checkpoint from %s", path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Validate checkpoint format
        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Invalid checkpoint format: missing 'model_state_dict'. "
                "Expected a training checkpoint."
            )

        return checkpoint

    def _extract_model_version(self) -> str:
        """Extract model version from checkpoint metadata.

        Returns:
            Model version string.
        """
        # Try to get version from checkpoint
        version = self.checkpoint.get("model_version", "unknown")

        # If not present, construct from checkpoint data
        if version == "unknown":
            training_date = self.checkpoint.get("training_date", "")
            if training_date:
                version = f"v{training_date[:10]}"
            else:
                version = "v1.0.0"

        return version

    def _load_temperature(self, calibration_path: str | None) -> float | None:
        """Load temperature scaling parameter if available."""
        if not calibration_path:
            return None
        import json
        import pathlib

        path = pathlib.Path(calibration_path)
        if not path.exists():
            logger.warning("Calibration file not found: %s", calibration_path)
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        temp = data.get("temperature")
        if temp is None:
            logger.warning("Calibration file missing 'temperature'.")
            return None
        logger.info("Loaded temperature scaling: %.4f", temp)
        return float(temp)

    def _build_model(self) -> FreshSenseEfficientNet:
        """Build model architecture and load weights.

        Returns:
            FreshSenseEfficientNet model with loaded weights.

        Raises:
            RuntimeError: If checkpoint architecture is incompatible.
        """
        # Get model config from checkpoint or use defaults
        num_classes = self.num_classes
        pretrained = False  # We load our own weights
        freeze_backbone = False  # Irrelevant at inference

        # Try to extract model config from checkpoint
        config_dict = self.checkpoint.get("config_dict", {})
        if config_dict:
            model_config = config_dict.get("model", {})
            dropout = model_config.get("dropout", 0.3)
            classifier_hidden = model_config.get("classifier_hidden", 256)
        else:
            dropout = 0.3
            classifier_hidden = 256

        # Validate checkpoint architecture version
        checkpoint_version = self.checkpoint.get("checkpoint_version", 1)
        architecture_version = self.checkpoint.get("architecture_version", "unknown")
        classifier_type = self.checkpoint.get("classifier_type")

        # Current expected architecture
        expected_classifier = "1280-256-6"

        if classifier_type is not None and classifier_type != expected_classifier:
            logger.error(
                "Checkpoint architecture mismatch!\n"
                "---------------------------------------\n"
                "Checkpoint trained with: %s\n"
                "Current model expects:   %s\n"
                "Checkpoint version:      %s\n"
                "Architecture version:    %s\n"
                "---------------------------------------\n"
                "Please retrain Phase 1 with the current architecture.\n"
                "---------------------------------------",
                classifier_type,
                expected_classifier,
                checkpoint_version,
                architecture_version,
            )
            raise RuntimeError(
                f"Incompatible checkpoint architecture: {classifier_type} "
                f"(expected {expected_classifier}). Please retrain Phase 1."
            )
        elif classifier_type is None:
            # Legacy checkpoint without metadata: rely on state-dict shape
            # validation below to detect architecture mismatches.
            logger.warning(
                "Checkpoint has no 'classifier_type' metadata; architecture "
                "will be validated from the weights when loading."
            )

        # Build model
        model = FreshSenseEfficientNet(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            classifier_hidden=classifier_hidden,
        )

        # Load weights
        try:
            model.load_state_dict(self.checkpoint["model_state_dict"])
            logger.info("Model weights loaded successfully")
        except RuntimeError as e:
            logger.error("Failed to load checkpoint: %s", e)
            raise RuntimeError(
                f"Checkpoint architecture mismatch: {e}. "
                "Please retrain Phase 1 with the current architecture."
            ) from e

        return model

    def predict(self, frame: np.ndarray) -> PredictionResult:
        """Run inference on a single frame.

        This method:
        1. Preprocesses the frame (same as training)
        2. Runs inference with torch.no_grad()
        3. Applies softmax to get probabilities
        4. Returns structured PredictionResult

        Args:
            frame: BGR image as numpy array (from OpenCV).

        Returns:
            PredictionResult with structured output.
        """
        start_time = time.perf_counter()

        # Preprocess (reuse existing pipeline)
        processed = self._preprocess(frame)

        # Inference
        with torch.no_grad():
            logits = self.model(processed)
            if self.temperature is not None:
                logits = logits / self.temperature
            probabilities = torch.softmax(logits, dim=1)[0]

        # Get prediction
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_idx = int(predicted_idx.item())
        confidence = float(confidence.item())

        # Extract class name
        if predicted_idx < len(self.class_names):
            class_name = self.class_names[predicted_idx]
        else:
            class_name = "unknown"

        # Extract fruit name and freshness
        fruit_name, freshness_class = self._parse_class_name(class_name)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return PredictionResult(
            fruit_name=fruit_name,
            freshness_class=freshness_class,
            confidence=confidence,
            probabilities=probabilities.cpu().numpy().tolist(),
            timestamp=time.time(),
            latency_ms=latency_ms,
            device=str(self.device),
            model_version=self.model_version,
        )

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for inference.

        Uses the deterministic inference transform only - the same applied to
        every frame (Resize + CenterCrop + ImageNet Normalize + ToTensor).
        Training augmentations are never used at inference time.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            Preprocessed tensor already moved to the inference device.
        """
        return self.transform(frame).to(self.device)

    def _parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """Parse class name into fruit name and freshness.

        Supports both the checkpoint format ("freshapples", "rottenbanana",
        "staleoranges") and the underscore form ("fresh_apple").

        Args:
            class_name: Full class name.

        Returns:
            Tuple of (fruit_name, freshness_class).
        """
        cleaned = (class_name or "").strip().lower()

        for prefix in ("fresh", "stale", "rotten"):
            if cleaned == prefix:
                return cleaned, prefix
            if cleaned.startswith(prefix):
                fruit = cleaned[len(prefix):].strip(" _-").strip()
                if fruit:
                    return fruit, prefix

        # Underscore fallback: "fresh_apple" -> (apple, fresh)
        parts = (class_name or "").split("_", 1)
        if len(parts) == 2 and parts[0].strip().lower() in ("fresh", "stale", "rotten"):
            return parts[1].strip(), parts[0].strip().lower()

        return class_name, "unknown"

    def get_class_names(self) -> List[str]:
        """Get list of class names.

        Returns:
            List of class names.
        """
        return self.class_names.copy()

    def get_num_classes(self) -> int:
        """Get number of classes.

        Returns:
            Number of classes.
        """
        return self.num_classes

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"Predictor("
            f"classes={self.num_classes}, "
            f"device={self.device}, "
            f"model={self.model_version})"
        )


if __name__ == "__main__":
    # Quick self-test.
    logging.basicConfig(level=logging.INFO)

    # Check if checkpoint exists
    checkpoint_path = "models/checkpoints/best_model.pth"
    import pathlib

    if not pathlib.Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using: python -m src.main")
        exit(1)

    print("Loading predictor...")
    predictor = Predictor(checkpoint_path)

    print(f"\n{predictor}")
    print(f"Classes: {predictor.get_class_names()}")

    # Create a dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("\nRunning inference on dummy frame...")
    result = predictor.predict(frame)

    print(f"\nPrediction Result:")
    print(f"  Fruit: {result.fruit_name}")
    print(f"  Freshness: {result.freshness_class}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Latency: {result.latency_ms:.1f}ms")
    print(f"  Device: {result.device}")

    print(f"\nLangGraph-compatible dict:")
    print(result.to_dict())