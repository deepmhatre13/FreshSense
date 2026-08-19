#!/usr/bin/env python3
"""Train YOLO detection model with full CLI parameterization and dataset support.

Supports training on any valid YOLO dataset directory (e.g. data/detection/ or
data/detection_v3/). Reuses train_detector logic and provides full CLI controls:
  --data, --model, --epochs, --batch, --imgsz, --device, --workers, --patience,
  --project, --name, --seed, --resume, --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml
from configs.config import Config
from scripts.train_detector import train_detector
from src.utils.environment import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_V3_CONFIG = _REPO_ROOT / "configs" / "detection_v3.yaml"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO detector pipeline (supports V2, V3, and custom dataset paths)."
    )
    parser.add_argument("--data", type=Path, default=None, help="Path to dataset directory or data.yaml file.")
    parser.add_argument("--model", type=str, default=None, help="Model backbone name or weights file (e.g. yolo11n.pt).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Training batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size (pixels).")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu, cuda, 0, auto).")
    parser.add_argument("--workers", type=int, default=None, help="Number of dataloader worker processes.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs).")
    parser.add_argument("--project", type=Path, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default="detector", help="Experiment run sub-directory name.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Display configuration and validation without starting training.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file path.")
    return parser.parse_args(argv)


def resolve_training_params(args: argparse.Namespace) -> dict:
    load_environment()

    # Load defaults from detection_v3.yaml if it exists, else settings.yaml
    cfg_dict = {}
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            cfg_dict = raw.get("detection_dataset", {})
    elif DEFAULT_V3_CONFIG.exists():
        with open(DEFAULT_V3_CONFIG, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            cfg_dict = raw.get("detection_dataset", {})
    else:
        app_config = Config.from_yaml("configs/settings.yaml")
        cfg_dict = {
            "detection_data_dir": app_config.detection_dataset.detection_data_dir,
            "detector_model": app_config.detection_dataset.detector_model,
            "detector_epochs": app_config.detection_dataset.detector_epochs,
            "detector_batch": app_config.detection_dataset.detector_batch,
            "detector_imgsz": app_config.detection_dataset.detector_imgsz,
            "detector_output_dir": app_config.detection_dataset.detector_output_dir,
            "detector_device": app_config.detection_dataset.detector_device,
            "detector_workers": app_config.detection_dataset.detector_workers,
            "detector_patience": app_config.detection_dataset.detector_patience,
            "seed": 42,
        }

    # Resolve data path
    if args.data is not None:
        data_arg = Path(args.data)
        data_yaml = data_arg if data_arg.name == "data.yaml" else data_arg / "data.yaml"
    else:
        data_dir = Path(cfg_dict.get("detection_data_dir", "data/detection_v3"))
        if not data_dir.is_absolute():
            data_dir = _REPO_ROOT / data_dir
        data_yaml = data_dir if data_dir.name == "data.yaml" else data_dir / "data.yaml"

    model_name = args.model or cfg_dict.get("detector_model", "yolo11n.pt")
    epochs = args.epochs if args.epochs is not None else int(cfg_dict.get("detector_epochs", 50))
    batch = args.batch if args.batch is not None else int(cfg_dict.get("detector_batch", 16))
    imgsz = args.imgsz if args.imgsz is not None else int(cfg_dict.get("detector_imgsz", 640))
    
    project_dir = args.project or Path(cfg_dict.get("detector_output_dir", "models/detection/v3"))
    if not project_dir.is_absolute():
        project_dir = _REPO_ROOT / project_dir

    device = args.device or cfg_dict.get("detector_device", "auto")
    workers = args.workers if args.workers is not None else int(cfg_dict.get("detector_workers", 4))
    patience = args.patience if args.patience is not None else int(cfg_dict.get("detector_patience", 10))
    seed = args.seed if args.seed is not None else int(cfg_dict.get("seed", 42))
    name = args.name

    return {
        "data_yaml": data_yaml,
        "model_name": model_name,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "project_dir": project_dir,
        "name": name,
        "device": device,
        "workers": workers,
        "patience": patience,
        "seed": seed,
        "resume": args.resume,
        "dry_run": args.dry_run,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    params = resolve_training_params(args)

    data_yaml: Path = params["data_yaml"]
    dry_run: bool = params["dry_run"]

    logger.info("=" * 70)
    logger.info("SmartFreshAI — YOLO Detection Pipeline Training")
    logger.info("=" * 70)
    logger.info("Dataset       : %s", data_yaml)
    logger.info("Model         : %s", params["model_name"])
    logger.info("Epochs        : %d", params["epochs"])
    logger.info("Batch Size    : %d", params["batch"])
    logger.info("Image Size    : %d", params["imgsz"])
    logger.info("Device        : %s", params["device"])
    logger.info("Workers       : %d", params["workers"])
    logger.info("Patience      : %d", params["patience"])
    logger.info("Output Dir    : %s", params["project_dir"])
    logger.info("Run Name      : %s", params["name"])
    logger.info("Seed          : %d", params["seed"])
    logger.info("Resume        : %s", params["resume"])
    logger.info("Dry Run       : %s", dry_run)
    logger.info("=" * 70)

    if not data_yaml.exists():
        logger.error(
            "Dataset data.yaml not found at: %s.\n"
            "If attempting to train on V3, the V3 dataset must be built first: "
            "python scripts/build_detection_v3.py (after resolving human review gate).",
            data_yaml,
        )
        return 1

    # Read class mapping safely if data.yaml exists
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
            class_names = data_cfg.get("names", [])
            logger.info("Classes (%d)   : %s", len(class_names), class_names)
    except Exception as e:
        logger.warning("Could not parse dataset data.yaml classes: %s", e)

    if dry_run:
        logger.info("[DRY RUN] Configuration validated successfully. No training executed.")
        return 0

    params["project_dir"].mkdir(parents=True, exist_ok=True)
    best_weights = train_detector(
        data_yaml=data_yaml,
        model_name=params["model_name"],
        epochs=params["epochs"],
        batch=params["batch"],
        imgsz=params["imgsz"],
        output_dir=params["project_dir"],
        device=params["device"],
        workers=params["workers"],
        patience=params["patience"],
        resume=params["resume"],
        seed=params["seed"],
    )

    logger.info("=" * 70)
    logger.info("Training finished! Best weights saved to: %s", best_weights)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
