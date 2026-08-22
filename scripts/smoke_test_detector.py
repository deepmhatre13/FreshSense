"""Real image smoke test for frozen production YOLO detector (Phase 11).

Usage:
    python scripts/smoke_test_detector.py [--image PATH]
"""

import argparse
import sys
from pathlib import Path
import cv2

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import DetectorConfig, YOLODetector


def main():
    parser = argparse.ArgumentParser(description="Smoke test production YOLO detector on a real image.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/detection/detector/weights/best.pt",
        help="Path to trained production YOLO weights.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/detection/test/images/apple_12_jpg.rf.7f4a14511957f2baef662e3855fd513b.jpg",
        help="Path to input test image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    print("==================================================")
    print("SMARTFRESHAI — PRODUCTION YOLO DETECTOR SMOKE TEST")
    print("==================================================")
    print(f"Model path: {model_path}")
    print(f"Image path: {image_path}")

    if not model_path.exists():
        print(f"ERROR: Production weight file missing at '{model_path}'!")
        sys.exit(1)

    if not image_path.exists():
        print(f"ERROR: Test image missing at '{image_path}'!")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not read image at '{image_path}'!")
        sys.exit(1)

    cfg = DetectorConfig(
        model_path=str(model_path),
        confidence_threshold=args.conf,
        image_size=640,
    )
    detector = YOLODetector(cfg, weight_name=str(model_path))

    print("Loading YOLO detector...")
    detector.load()
    print("Detector loaded successfully.")

    print("Running inference...")
    result = detector.detect(image)

    print("\n--------------------------------------------------")
    print(f"Frame Size: {result.frame_width} x {result.frame_height}")
    print(f"Latency   : {result.latency_ms:.2f} ms")
    print(f"Detections: {result.count}")
    print("--------------------------------------------------")

    if result.count == 0:
        print("No objects detected above confidence threshold.")
    else:
        print("Detected objects:")
        for idx, det in enumerate(result.detections, 1):
            print(f"{idx}. {det.class_name:<10} confidence={det.confidence:.4f}")
            print(f"   class_id={det.class_id} bbox=(x1={det.x1}, y1={det.y1}, x2={det.x2}, y2={det.y2})")

    print("==================================================")
    print("SMOKE TEST COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    main()
