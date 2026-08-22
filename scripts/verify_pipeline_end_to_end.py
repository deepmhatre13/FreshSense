"""End-to-End validation of DetectionPipeline with production YOLO detector (Phase 12).

Usage:
    python scripts/verify_pipeline_end_to_end.py
"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.detection_pipeline import DetectionPipeline, DetectionPipelineConfig
from configs.config import Config


def main():
    print("==================================================")
    print("SMARTFRESHAI — END-TO-END PIPELINE VALIDATION")
    print("==================================================")

    yaml_config = Config.from_yaml("configs/settings.yaml")
    d = yaml_config.detection

    pipe_cfg = DetectionPipelineConfig(
        detector_name="yolo",
        detector_weights=d.detector_weights,
        detector_imgsz=d.detector_imgsz,
        confidence_threshold=d.detection_confidence,
        iou_threshold=d.detection_iou,
        max_detections=d.max_detections,
    )

    print(f"Initializing DetectionPipeline with weights: {pipe_cfg.detector_weights}")
    pipeline = DetectionPipeline(pipe_cfg, predictor=None)
    pipeline.initialize()
    print("Pipeline initialized successfully.")

    image_path = Path("data/detection/test/images/apple_12_jpg.rf.7f4a14511957f2baef662e3855fd513b.jpg")
    print(f"Processing test frame: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: Could not read image at '{image_path}'!")
        sys.exit(1)

    result = pipeline.process_frame(frame)

    print("\n--------------------------------------------------")
    print(f"Total Detected Fruits : {len(result.fruits)}")
    print(f"Unidentified Count    : {result.unidentified_count}")
    print(f"Frame Dimensions      : {result.frame_width} x {result.frame_height}")
    print("--------------------------------------------------")

    for idx, fruit in enumerate(result.fruits, 1):
        det = fruit.detection
        print(f"\nFruit #{idx} (Track #{det.tracking_id}):")
        print(f"  Class           : {det.class_name} (class_id={det.class_id})")
        print(f"  Detector Conf   : {det.confidence:.4f}")
        print(f"  BBox (x1,y1,x2,y2): ({det.x1}, {det.y1}, {det.x2}, {det.y2})")
        print(f"  Freshness State : {fruit.freshness_class}")
        print(f"  Fused Confidence: {fruit.fused_confidence:.4f}")
        if fruit.shelf_life:
            print(f"  Est. Shelf Life : {fruit.shelf_life.to_range_string()} ({fruit.shelf_life.basis_type})")

    pipeline.shutdown()
    print("\n==================================================")
    print("END-TO-END VALIDATION COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    main()
