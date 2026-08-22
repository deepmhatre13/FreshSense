"""Trace a real image end-to-end through the complete SmartFreshAI pipeline (Phase 3).

Usage:
    python scripts/trace_real_image_pipeline.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.detection_pipeline import DetectionPipeline, DetectionPipelineConfig
from src.inference.predictor import Predictor
from configs.config import Config


def main():
    print("==================================================")
    print("PHASE 3 — TRACING REAL IMAGE END-TO-END PIPELINE")
    print("==================================================")

    # 1. Load settings
    yaml_config = Config.from_yaml("configs/settings.yaml")
    d = yaml_config.detection

    # 2. Check predictor checkpoint
    ckpt_path = "models/checkpoints/best_model.pth"
    predictor = None
    if Path(ckpt_path).exists():
        print(f"Loading EfficientNet predictor from '{ckpt_path}'...")
        predictor = Predictor(ckpt_path, config=yaml_config)
    else:
        print(f"WARNING: Predictor checkpoint missing at '{ckpt_path}'")

    # 3. Initialize Pipeline
    pipe_cfg = DetectionPipelineConfig(
        detector_name="yolo",
        detector_weights=d.detector_weights,
        detector_imgsz=d.detector_imgsz,
        confidence_threshold=d.detection_confidence,
        iou_threshold=d.detection_iou,
        max_detections=d.max_detections,
    )
    pipeline = DetectionPipeline(pipe_cfg, predictor=predictor)
    pipeline.initialize()

    # 4. Load Real Image
    image_path = Path("data/detection/test/images/apple_12_jpg.rf.7f4a14511957f2baef662e3855fd513b.jpg")
    print(f"\n1. STAGE 1: INPUT IMAGE")
    print(f"   Image Path: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: Cannot read image '{image_path}'")
        sys.exit(1)
    print(f"   Shape: {frame.shape} (H x W x C)")

    # STAGE 2: Quality Assessment
    print(f"\n2. STAGE 2: QUALITY ASSESSMENT")
    q_res = pipeline.quality_assessor.assess(frame)
    print(f"   Quality Result: is_quality_ok={q_res.is_quality_ok}, warnings={q_res.warnings}")
    print(f"   Metrics: blur_variance={q_res.blur_variance:.1f}, brightness={q_res.brightness:.1f}, contrast={q_res.contrast:.1f}")

    # STAGE 3: YOLO Object Detection
    print(f"\n3. STAGE 3: YOLO OBJECT DETECTION")
    det_res = pipeline.detector.detect(frame)
    print(f"   Detections Count: {det_res.count}")
    print(f"   Latency: {det_res.latency_ms:.2f} ms")
    for idx, d_obj in enumerate(det_res.detections, 1):
        print(f"   Det #{idx}: class_id={d_obj.class_id}, label='{d_obj.label}', conf={d_obj.confidence:.4f}, bbox=({d_obj.x1},{d_obj.y1},{d_obj.x2},{d_obj.y2})")

    # STAGE 4: Multi-Object Tracking
    print(f"\n4. STAGE 4: MULTI-OBJECT TRACKING")
    tracked_dets = pipeline.tracker.update(det_res.detections)
    for idx, t_obj in enumerate(tracked_dets, 1):
        print(f"   Tracked #{idx}: track_id={t_obj.tracking_id}, label='{t_obj.label}', conf={t_obj.confidence:.4f}")

    # STAGE 5: Cropping
    print(f"\n5. STAGE 5: CROP PROCESSING")
    crops = [pipeline.cropper.crop(frame, d) for d in tracked_dets]
    for idx, (t_obj, c_res) in enumerate(zip(tracked_dets, crops), 1):
        crop_shape = c_res.cropped.shape if c_res.cropped is not None else None
        print(f"   Crop #{idx} (Track #{t_obj.tracking_id}): valid={c_res.valid}, shape={crop_shape}")

    # STAGE 6, 7, 8, 9: Full Pipeline process_frame()
    print(f"\n6. STAGES 6-9: CLASSIFICATION, STABILIZATION, FUSION, SHELF-LIFE & RESULT ASSEMBLY")
    multi_res = pipeline.process_frame(frame)

    print(f"   MultiFruitResult: total_fruits={len(multi_res.fruits)}, unidentified={multi_res.unidentified_count}")
    for idx, fruit in enumerate(multi_res.fruits, 1):
        d_info = fruit.detection
        print(f"\n   --- Fruit Instance #{idx} ---")
        print(f"   - Fruit Class        : {d_info.class_name} (class_id={d_info.class_id})")
        print(f"   - Detection Conf     : {d_info.confidence:.4f}")
        print(f"   - Bounding Box       : ({d_info.x1}, {d_info.y1}, {d_info.x2}, {d_info.y2})")
        print(f"   - Track ID           : #{d_info.tracking_id}")
        print(f"   - Classifier Output  : {fruit.metadata.get('model_version', 'N/A')}")
        print(f"   - Freshness Output   : {fruit.freshness_class}")
        print(f"   - Fused Confidence   : {fruit.fused_confidence:.4f}")
        if fruit.shelf_life:
            print(f"   - Shelf Life Output  : {fruit.shelf_life.to_range_string()}")
            print(f"   - Shelf Life Basis   : {fruit.shelf_life.basis} ({fruit.shelf_life.basis_type})")

    pipeline.shutdown()
    print("\n==================================================")
    print("END-TO-END TRACE COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    main()
