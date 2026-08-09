"""Detector factory for FreshSense Phase 4.

The factory decouples detector creation from usage. Adding a new detector
backend (e.g. ``"rtdetr"``, ``"grounding_dino"``, ``"onnx"``, ``"tensorrt"``)
only requires registering it here; the inference pipeline never changes.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from src.detection.base_detector import BaseDetector, DetectorConfig
from src.detection.detector import MockDetector, SimpleDetector, YOLODetector

logger = logging.getLogger(__name__)

__all__ = ["DetectorFactory"]


class DetectorFactory:
    """Factory that builds detector instances by name.

    Registered backends:

    - ``"yolo"``      -> :class:`YOLODetector` (Ultralytics YOLOv11)
    - ``"simple"``    -> :class:`SimpleDetector` (HSV demo)
    - ``"mock"``      -> :class:`MockDetector` (deterministic test stub)
    """

    _registry: Dict[str, Type[BaseDetector]] = {
        "yolo": YOLODetector,
        "simple": SimpleDetector,
        "mock": MockDetector,
    }

    @classmethod
    def create(
        cls,
        name: str,
        config: DetectorConfig,
        **kwargs,
    ) -> BaseDetector:
        """Create a detector instance.

        Args:
            name: Detector backend name ("yolo", "simple", "mock").
            config: DetectorConfig with settings.
            **kwargs: Extra constructor kwargs passed to the detector class.

        Returns:
            An instance of the requested detector.

        Raises:
            ValueError: If the backend name is unknown.
        """
        key = name.strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown detector backend: {name!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        detector_cls = cls._registry[key]
        detector = detector_cls(config=config, **kwargs)
        logger.info("DetectorFactory created backend %r", key)
        return detector

    @classmethod
    def register(cls, name: str, detector_cls: Type[BaseDetector]) -> None:
        """Register a new detector backend.

        Args:
            name: Backend name.
            detector_cls: Detector subclass of :class:`BaseDetector`.
        """
        cls._registry[name.strip().lower()] = detector_cls
        logger.info("Registered detector backend %r", name)

    @classmethod
    def available_backends(cls) -> list:
        """Return sorted list of registered backend names."""
        return sorted(cls._registry)
