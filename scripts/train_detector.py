#!/usr/bin/env python3
"""Train YOLO detector on FreshSense fruit detection dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running directly (python scripts/<name>.py) from any CWD.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.config import Config
from src.utils.environment import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_detector(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    output_dir: Path,
    device: str = "auto",
    workers: int = 4,
    patience: int = 10,
    resume: bool = False,
    seed: int = 42,
) -> Path:
    """Train YOLO model and return path to best weights."""
    # Normalize filesystem paths at the boundary so callers may pass either a
    # str or a Path; every filesystem operation below is reliable.
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics required. Install: pip install ultralytics")
        sys.exit(1)

    if not data_yaml.exists():
        logger.error("data.yaml not found: %s", data_yaml)
        sys.exit(1)

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading base model: %s", model_name)
    model = YOLO(model_name)

    logger.info("Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        patience=patience,
        project=str(output_dir),
        name="detector",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        resume=resume,
        seed=seed,
    )

    run_dir = output_dir / "detector"
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    if best_weights.exists():
        return best_weights
    elif last_weights.exists():
        logger.warning("best.pt not found, using last.pt")
        return last_weights
    else:
        logger.error("No weights found in %s", run_dir)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FreshSense YOLO detector"
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")

    data_yaml = args.data or (Path(config.detection_dataset.detection_data_dir) / "data.yaml")
    model_name = args.model or config.detection_dataset.detector_model
    epochs = args.epochs or config.detection_dataset.detector_epochs
    batch = args.batch or config.detection_dataset.detector_batch
    imgsz = args.imgsz or config.detection_dataset.detector_imgsz
    output_dir = Path(args.output or config.detection_dataset.detector_output_dir)
    device = args.device or config.detection_dataset.detector_device
    workers = args.workers or config.detection_dataset.detector_workers
    patience = args.patience or config.detection_dataset.detector_patience
    seed = args.seed or 42

    logger.info("=" * 70)
    logger.info("FreshSense AI - YOLO Detector Training")
    logger.info("=" * 70)

    best_path = train_detector(
        data_yaml=data_yaml,
        model_name=model_name,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        output_dir=output_dir,
        device=device,
        workers=workers,
        patience=patience,
        resume=args.resume,
        seed=seed,
    )

    logger.info("=" * 70)
    logger.info("Training complete! Best: %s", best_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
