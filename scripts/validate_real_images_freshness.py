"""Validate freshness and shelf-life pipeline across 6 real fruit image species (Phase 14 & Phase 15).

Target species (YOLO-detected; 3 freshness-unsupported, 3 supported):
    1. Apple    (freshness-classifier supported)
    2. Orange   (freshness-classifier supported)
    3. banana   (freshness-classifier supported)
    4. Grape    (freshness unsupported -> must NOT fabricate a freshness)
    5. Mango    (freshness unsupported)
    6. Guava    (freshness unsupported)
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.detection_pipeline import DetectionPipeline, DetectionPipelineConfig
from src.inference.predictor import Predictor
from configs.config import Config


TEST_IMAGES = [
    ("Apple", "data/detection/test/images/apple_12_jpg.rf.7f4a14511957f2baef662e3855fd513b.jpg"),
    ("Orange", "data/detection/test/images/Curiosidades-de-las-naranjas_jpg.rf.639f81f684e97cbc5975aae4faa16826.jpg"),
    ("banana", "data/detection/test/images/ripe-bananas-in-a-wooden-bowl-royalty-free-image-1724469492_avif.rf.5b9c85292363c7945289538dde1e348e.jpg"),
    ("Grape", "data/detection/test/images/Grape-45-_jpeg.rf.3bdf30a4f5e6e9aad80c6551bc5b3804.jpg"),
    ("Mango", "data/detection/test/images/IMG_7533_jpg.rf.a230dd16211d0bd803384770fb3d0742.jpg"),
    ("Guava", "data/detection/test/images/guava-1-_jpg.rf.db1cd8ae83e739606795761d6fb433cd.jpg"),
]


def main():
    print("==================================================================")
    print("SMARTFRESHAI — REAL IMAGE FRESHNESS & SHELF-LIFE VALIDATION")
    print("==================================================================")

    yaml_config = Config.from_yaml("configs/settings.yaml")
    d = yaml_config.detection

    ckpt_path = "models/checkpoints/best_model.pth"
    predictor = Predictor(ckpt_path, config=yaml_config) if Path(ckpt_path).exists() else None

    pipe_cfg = DetectionPipelineConfig(
        detector_name="yolo",
        detector_weights=d.detector_weights,
        detector_imgsz=d.detector_imgsz,
        confidence_threshold=d.detection_confidence,
        iou_threshold=d.detection_iou,
    )

    pipeline = DetectionPipeline(pipe_cfg, predictor=predictor)
    pipeline.initialize()

    latencies = []

    for species, img_path_str in TEST_IMAGES:
        img_path = Path(img_path_str)
        print(f"\n------------------------------------------------------------------")
        print(f"Species: {species} | Path: {img_path_str}")
        print(f"------------------------------------------------------------------")

        if not img_path.exists():
            print(f"ERROR: Image not found at {img_path}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"ERROR: Unable to read image frame at {img_path}")
            continue

        t0 = time.perf_counter()
        result = pipeline.process_frame(frame)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        print(f"Frame Size: {result.frame_width} x {result.frame_height} | Latency: {latency_ms:.2f} ms")
        print(f"Detected Fruits: {len(result.fruits)}")

        for idx, fruit in enumerate(result.fruits, 1):
            det = fruit.detection
            sl = fruit.shelf_life
            if sl:
                if sl.remaining_days is not None:
                    sl_str = f"{sl.remaining_days} days (from {sl.typical_min_days}-{sl.typical_max_days})"
                else:
                    sl_str = sl.shelf_life_status
            else:
                sl_str = "N/A"
            sl_basis = sl.basis if sl else "N/A"

            print(f"  Fruit #{idx}: {det.class_name:<10} | Det Conf: {det.confidence:.4f} | BBox: ({det.x1},{det.y1},{det.x2},{det.y2})")
            print(f"           Freshness: {fruit.freshness_class:<11} | Fused Conf: {fruit.fused_confidence:.4f}")
            print(f"           Shelf-Life: {sl_str:<10} | Basis: {sl_basis}")

    pipeline.shutdown()

    print("\n==================================================================")
    print("PHASE 15 — LATENCY PROFILE REPORT")
    print("==================================================================")
    if latencies:
        print(f"Total Samples Processed : {len(latencies)}")
        print(f"Mean Inference Latency  : {np.mean(latencies):.2f} ms")
        print(f"Min Inference Latency   : {np.min(latencies):.2f} ms")
        print(f"Max Inference Latency   : {np.max(latencies):.2f} ms")
    print("==================================================================")


if __name__ == "__main__":
    main()
