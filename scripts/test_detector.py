#!/usr/bin/env python3
"""Local CPU smoke test + single-image inference for the YOLO detector.

Loads the trained ``best.pt`` checkpoint (default:
``models/detection/detector/weights/best.pt``), runs inference on one or more
images, and reports detections (class, confidence, xyxy box). Optionally writes
an annotated PNG output.

The model is loaded with the existing :class:`YOLODetector` abstraction so that
background RGB/BGR conventions, bounding-box format (int xyxy), and the project
``Detection``/``DetectionResult`` types are preserved.

Usage:
    python -m scripts.test_detector
    python -m scripts.test_detector --image <path-to-image>.jpg
    python -m scripts.test_detector --model models/detection/detector/weights/best.pt
    python -m scripts.test_detector --conf 0.25 --output reports/detection_smoke_test

Exit code:
    0  — at least one image processed successfully
    1  — model or image missing, or inference failed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running directly (python scripts/<name>.py) from any CWD.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.config import Config
from src.detection.base_detector import BoundingBox, Detection, DetectorConfig
from src.detection.detector import YOLODetector
from src.detection.visualizer import color_for_label, draw_box
from src.utils.environment import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _default_model_path(config: Config) -> Path:
    """Return the trained inference checkpoint path.

    Prefers ``models/detection/detector/weights/best.pt`` (the standard
    Ultralytics output for ``output_dir=models/detection``). If absent, falls
    back to the pretrained name used for training only as a last resort and
    clearly logs that it is not the trained artifact.
    """
    default = Path("models/detection/detector/weights/best.pt")
    if default.exists():
        return default
    # Config supplies the pretrained model name used for training, not the
    # trained inference checkpoint. We only use it if best.pt is truly absent.
    pretrained = Path(config.detection_dataset.detector_model)
    logger.warning(
        "Trained checkpoint not found at %s; falling back to %s "
        "(this is the pretrained base model, not the trained detector).",
        default,
        pretrained,
    )
    return pretrained


def _class_name_to_id_map(detector: YOLODetector) -> dict:
    """Build a ``{class_name_lower: class_id}`` map from the loaded model.

    The project :class:`~src.detection.base_detector.Detection` dataclass only
    carries ``label`` (a class-name string) - it has **no** ``class_id`` field,
    so ``getattr(det, "class_id", -1)`` always fell back to ``-1``. The loaded
    Ultralytics model exposes ``names`` -> ``{id: name}``, which is the reliable
    reverse mapping to recover the original class id without inventing it.

    Returns an empty dict when the detector is not model-backed (or has not been
    loaded with a model exposing ``names``), so callers can fall back cleanly
    to ``-1`` rather than fabricating an id.
    """
    model = getattr(detector, "model", None)
    names = getattr(model, "names", None) if model is not None else None
    if isinstance(names, dict):
        return {str(v).lower(): int(k) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {str(v).lower(): int(k) for k, v in enumerate(names)}
    return {}


def run_inference(
    detector: YOLODetector,
    image_path: Path,
    conf_threshold: float = 0.45,
    verbose: bool = True,
) -> dict:
    """Run YOLO inference on a single image and return a clean detection report.

    Args:
        detector: A loaded :class:`YOLODetector`.
        image_path: Path to an image file (any format OpenCV can decode).
        conf_threshold: Minimum confidence to keep a detection (0.0-1.0).
        verbose: If True, log each kept detection.

    Returns:
        A dict suitable for printing/visualization:

        .. code-block:: python

            {
              "image": "<abs path>",
              "latency_ms": 12.34,
              "count": 2,
              "detections": [
                  {
                      "class_id": 0,
                      "class_name": "Apple",
                      "confidence": 0.91,
                      "x1": 120, "y1": 80, "x2": 220, "y2": 260,
                  },
                  ...
              ],
            }

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If the image cannot be decoded, or a detection has an
            invalid bounding-box layout (x2 < x1 or y2 < y1).
    """
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}."
            f" Provide a valid --image path, e.g. data/detection/test/images/<file>.jpg."
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(
            f"Could not decode image (invalid or unsupported format): {image_path}"
        )

    start = time.perf_counter()
    result = detector.detect(image)
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Recover the true class id. Detection objects only persist `label` (the
    # class-name string); they carry no class_id field, so the old
    # getattr(det, "class_id", -1) always returned -1. Derive it from the loaded
    # model's canonical names mapping (verified against the model, not ordered).
    class_id_map: dict = _class_name_to_id_map(detector)
    if not class_id_map and result.detections:
        logger.warning(
            "Could not read class-name mapping from detector model; "
            "class_id will fall back to -1 for each detection."
        )

    detections: list[dict] = []
    for det in result.detections:
        # Skip low-confidence detections.
        if det.confidence < conf_threshold:
            continue
        bbox = det.bbox
        if bbox.x2 < bbox.x1 or bbox.y2 < bbox.y1:
            logger.warning(
                "Skipping detection with invalid bbox %s for class %r.",
                bbox.to_tuple(),
                det.label,
            )
            continue
        entry = {
            "class_id": int(class_id_map.get(str(det.label).lower(), -1)),
            "class_name": det.label,
            "confidence": round(float(det.confidence), 4),
            "x1": int(bbox.x1),
            "y1": int(bbox.y1),
            "x2": int(bbox.x2),
            "y2": int(bbox.y2),
        }
        detections.append(entry)
        if verbose:
            logger.info(
                "  %-12s conf=%.3f xyxy=(%d, %d, %d, %d)",
                det.label,
                det.confidence,
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
            )

    return {
        "image": str(image_path),
        "latency_ms": round(latency_ms, 2),
        "count": len(detections),
        "detections": detections,
    }


def annotate_and_save(
    image_path: Path | str,
    result: dict,
    output_path: Path | str,
) -> Path:
    """Draw already-computed detections from ``result`` onto the image and save.

    This operates **only** on a pre-computed inference result (the dict returned
    by :func:`run_inference`). It performs no inference of its own and never
    accepts a :class:`YOLODetector` (the caller owns inference, this function
    only annotates). Each detection is drawn as a bounding box plus a
    ``class_name confidence`` caption.

    Args:
        image_path: Path to the source image (read from disk; the on-disk
            file is never overwritten unless ``output_path`` equals it).
        result: Detection report dict from :func:`run_inference`. Must contain a
            ``"detections"`` list whose entries each carry ``x1``/``y1``/``x2``/
            ``y2`` (and optionally ``class_name``/``confidence``).
        output_path: File path for the annotated image. Parent directories are
            created if missing.

    Returns:
        The :class:`~pathlib.Path` to the written annotated image.

    Raises:
        ValueError: If ``result`` is malformed (not a dict, missing
            ``detections``, or a detection lacks the required coordinates) or
            the source image cannot be decoded.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not isinstance(result, dict):
        raise ValueError(
            f"result must be a dict, got {type(result).__name__!r}."
        )
    if "detections" not in result:
        raise ValueError("result is missing the required 'detections' key.")
    detections = result["detections"]
    if not isinstance(detections, list):
        raise ValueError(
            f"result['detections'] must be a list, got {type(detections).__name__!r}."
        )

    # Only overwrite the source when the caller explicitly targets it.
    if output_path.resolve() == image_path.resolve():
        logger.warning(
            "output_path equals image_path; the source image will be overwritten."
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(
            f"Could not decode image (invalid or unsupported format): {image_path}"
        )

    for det in detections:
        if not isinstance(det, dict):
            raise ValueError(
                f"each detection must be a dict, got {type(det).__name__!r}."
            )
        for coord in ("x1", "y1", "x2", "y2"):
            if coord not in det:
                raise ValueError(
                    f"detection is missing required key {coord!r}: {det}"
                )
        x1, y1, x2, y2 = (
            int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"]),
        )
        class_name = str(det.get("class_name", "object"))
        confidence = float(det.get("confidence", 0.0))

        # draw_box draws the rectangle + caption in place on the in-memory
        # array only; the on-disk source image is left untouched.
        draw_box(
            image,
            BoundingBox(x1, y1, x2, y2),
            color_for_label(class_name),
            f"{class_name} {confidence:.2f}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info("Wrote annotated image: %s", output_path)
    return output_path
