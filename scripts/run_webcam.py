#!/usr/bin/env python3
"""Run the trained SmartFreshAI YOLO detector against a local webcam.

This is a **detection-only** real-time application (no freshness
classification, no tracking, no shelf-life estimation, no DetectionPipeline).
The flow for each frame is:

    OpenCV webcam frame -> YOLODetector.detect() -> DetectionResult ->
    draw bounding boxes -> display annotated frame (optional HUD)

Inference always goes through the existing :class:`YOLODetector` abstraction
loaded from the trained ``best.pt`` checkpoint. Ultralytics is never imported
directly here. The implementation is intentionally simple and synchronous
(no threading/multiprocessing/queues/async) so that actual FPS and latency are
reported honestly on the CPU-only machine.

Usage:
    python -m scripts.run_webcam --help
    python -m scripts.run_webcam --device cpu
    python -m scripts.run_webcam --model models/detection/detector/weights/best.pt
        --device cpu --conf 0.25 --save reports/webcam.mp4

Exit code:
    0  - normal quit / finite frame budget exhausted
    1  - runtime failure (model missing, camera open error, etc.)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Allow running directly (python scripts/run_webcam.py) from any CWD.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.detection.base_detector import BoundingBox, DetectionResult, DetectorConfig
from src.detection.detector import YOLODetector
from src.detection.visualizer import draw_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "models/detection/detector/weights/best.pt"
def _gui_available() -> bool:
    """Return True if OpenCV HighGUI windowing actually works in this env.

    ``import cv2`` alone is not sufficient: a headless OpenCV build can import
    cleanly while ``cv2.imshow``/``cv2.waitKey`` are unimplemented stubs that
    raise ``cv2.error`` ("The function is not implemented. Rebuild the library
    with Windows, GTK+ 2.x or Cocoa support.").

    This performs a safe minimum HighGUI exercise: create a throwaway window,
    show a 1x1 black frame, and then tear the window down. If any step raises,
    GUI is treated as unavailable. The window is always destroyed and no window
    is left open afterwards. The result is cached so it is probed exactly once.
    """
    if getattr(_gui_available, "_cached", None) is not None:
        return _gui_available._cached

    probe = "sfai_probe_gui"
    ok = True
    try:
        cv2.namedWindow(probe)
        cv2.imshow(probe, np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)
    except Exception:  # noqa: BLE001 - cv2.error surfaces here on headless builds
        ok = False
    finally:
        try:
            cv2.destroyWindow(probe)
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001 - teardown must never mask the check
            pass

    _gui_available._cached = ok
    if not ok:
        logger.warning(
            "OpenCV HighGUI is unavailable (headless build). "
            "Running webcam inference without display."
        )
    return ok


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="scripts.run_webcam",
        description=(
            "Run the trained SmartFreshAI YOLO detector on a local webcam "
            "(detection-only)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
        help="Path to trained YOLO weights (default: %(default)s).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: %(default)s).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for non-max suppression (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: %(default)s).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested camera width (default: %(default)s).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested camera height (default: %(default)s).",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=20,
        help="Maximum number of detections per frame (default: %(default)s).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output video path (e.g. reports/webcam.mp4).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (no GUI window); requires --frames > 0.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help=(
            "Total frames to process, then exit. 0 = unlimited (only allowed "
            "when display is enabled). Required (positive) when --no-display."
        ),
    )
    return parser


def _create_detector(args: argparse.Namespace) -> YOLODetector:
    """Build a :class:`YOLODetector` from CLI args (model + config)."""
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. "
            f"Provide a valid --model path (default: {_DEFAULT_MODEL})."
        )
    config = DetectorConfig(
        model_path=str(model_path),
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        max_detections=args.max_detections,
    )
    detector = YOLODetector(config, weight_name=str(model_path))
    return detector

def _open_camera(
    camera_index: int,
    width: int,
    height: int,
) -> cv2.VideoCapture:
    """Open the webcam, request a resolution, and validate it opened."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # The wrapper is unusable; release the handle before raising.
        cap.release()
        raise RuntimeError(f"Could not open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        "Camera %d opened. Requested %dx%d; actual %dx%d.",
        camera_index,
        width,
        height,
        actual_w,
        actual_h,
    )
    return cap


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    latency_ms: Optional[float],
    num_detections: int,
    device: str,
    width: int,
    height: int,
) -> None:
    """Draw a small read-only HUD in place on an annotated frame."""
    device_label = device.upper() if device != "auto" else "AUTO"
    lat = f"{latency_ms:.1f} ms" if latency_ms is not None else "n/a"
    lines = [
        f"FPS: {fps:.1f}",
        f"Inference: {lat}",
        f"Objects: {num_detections}",
        f"Device: {device_label}",
        f"Resolution: {width}x{height}",
        "q = quit   s = save snapshot",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_h = 20
    pad_x, pad_y = 8, 18
    for i, text in enumerate(lines):
        y = pad_y + i * line_h
        # Slight dark outline for readability over arbitrary backgrounds.
        cv2.putText(frame, text, (pad_x, y), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (pad_x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def run_webcam(args: argparse.Namespace) -> int:
    """Open the webcam, run detection, and process frames until exit."""
    detector = _create_detector(args)
    detector.load()
    # Real-time single-pass inference benefits from a warmup call.
    detector.warmup()
    logger.info(
        "Detector ready. Model=%s, device=%s, conf=%.2f, iou=%.2f, max_det=%d.",
        detector.weight_name,
        detector.device_str,
        args.conf,
        args.iou,
        args.max_detections,
    )

    cap = _open_camera(args.camera, args.width, args.height)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional video writer; created only when --save is supplied, using the
    # actual frame dimensions that OpenCV reports.
    writer: Optional[cv2.VideoWriter] = None
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # MP4 with the widely-supported MP4V codec.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, 20.0, (frame_w, frame_h))
        if not writer.isOpened():
            writer = None
            logger.error("Could not open video writer for %s; continuing without saving.", save_path)
        else:
            logger.info("Saving video to %s (%dx%d).", save_path, frame_w, frame_h)

    # Decide whether GUI display is possible. --no-display forces headless;
    # otherwise we probe HighGUI and gracefully fall back to headless if the
    # installed OpenCV build provides no GUI support.
    gui_enabled = not args.no_display and _gui_available()

    if args.no_display:
        logger.info("Headless mode: no display window will be opened.")
    elif not gui_enabled and not _gui_available():
        # _gui_available() already logged the headless warning; add a clear note
        # about the automatic fallback so the reason is never hidden.
        logger.warning(
            "GUI is unavailable and --no-display was not set; "
            "automatically falling back to headless processing."
        )

    # Without a display (either --no-display or unavailable HighGUI) an
    # unlimited frame budget would loop forever. Require a finite --frames.
    if not gui_enabled and args.frames <= 0:
        logger.error(
            "OpenCV HighGUI is unavailable and no display is being used.\n"
            "Cannot run unlimited webcam mode without a display.\n"
            "Re-run with:\n"
            "    --no-display --frames <N>"
        )
        raise SystemExit(1)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    frame_count: int = 0
    loop_start = time.perf_counter()
    try:
        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame from camera; stopping.")
                break

            start = time.perf_counter()
            result: DetectionResult = detector.detect(frame)
            frame_time = time.perf_counter() - start
            fps = 1.0 / frame_time if frame_time > 0 else 0.0
            latency_ms = result.latency_ms if result.latency_ms is not None else None

            # Draw every detection using the project's detection-aware utility
            # (consumes Detection objects directly; no second representation).
            annotated = draw_detections(frame, result.detections, show_tracking_id=False)

            _draw_hud(
                annotated,
                fps=fps,
                latency_ms=latency_ms,
                num_detections=len(result.detections),
                device=detector.device_str,
                width=frame_w,
                height=frame_h,
            )

            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("SmartFreshAI - YOLO Webcam", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("'q' pressed; quitting.")
                    break
                if key == ord("s"):
                    snapshot = (
                        reports_dir
                        / f"webcam_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    )
                    cv2.imwrite(str(snapshot), annotated)
                    logger.info("Saved snapshot: %s", snapshot)

            frame_count += 1
            if args.frames > 0 and frame_count >= args.frames:
                logger.info("Processed %d frames; exiting.", frame_count)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user; cleaning up.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        detector.shutdown()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - loop_start
    if frame_count > 0:
        logger.info(
            "Processed %d frames in %.2fs (avg %.2f FPS).",
            frame_count,
            elapsed,
            frame_count / elapsed,
        )
    return 0


def main() -> int:
    """Parse CLI args and run the webcam loop."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.conf < 0.0 or args.conf > 1.0:
        parser.error("--conf must be in [0.0, 1.0]")
    if args.iou < 0.0 or args.iou > 1.0:
        parser.error("--iou must be in [0.0, 1.0]")
    if args.max_detections <= 0:
        parser.error("--max-detections must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.frames < 0:
        parser.error("--frames must be >= 0")

    try:
        return run_webcam(args)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
