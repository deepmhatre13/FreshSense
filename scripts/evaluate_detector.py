#!/usr/bin/env python3
"""Evaluate trained YOLO detector on test split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from configs.config import Config
from src.utils.environment import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_detector(
    model_path: Path,
    data_yaml: Path,
    split: str = "test",
    output_report: Path | None = None,
) -> dict:
    """Evaluate YOLO model and return metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics required. Install: pip install ultralytics")
        sys.exit(1)

    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    if not data_yaml.exists():
        logger.error("data.yaml not found: %s", data_yaml)
        sys.exit(1)

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    logger.info("Running evaluation on %s split...", split)
    metrics = model.val(data=str(data_yaml), split=split, verbose=False)

    results = {
        "status": "available",
        "model": str(model_path),
        "dataset": str(data_yaml),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {},
    }

    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        class_names = model.names
        for idx, ap in enumerate(metrics.box.maps):
            if idx < len(class_names):
                results["per_class"][class_names[idx]] = {
                    "precision": float(metrics.box.p[idx])
                    if hasattr(metrics.box, "p") and metrics.box.p is not None
                    else 0.0,
                    "recall": float(metrics.box.r[idx])
                    if hasattr(metrics.box, "r") and metrics.box.r is not None
                    else 0.0,
                    "ap50": float(ap),
                }

    logger.info("=" * 70)
    logger.info("Evaluation Results:")
    logger.info("  Precision:      %.4f", results["precision"])
    logger.info("  Recall:         %.4f", results["recall"])
    logger.info("  mAP@50:         %.4f", results["map50"])
    logger.info("  mAP@50-95:      %.4f", results["map50_95"])
    logger.info("=" * 70)

    if results["per_class"]:
        logger.info("Per-class metrics:")
        for cls, metrics_dict in results["per_class"].items():
            logger.info(
                "  %s: P=%.4f R=%.4f AP50=%.4f",
                cls,
                metrics_dict["precision"],
                metrics_dict["recall"],
                metrics_dict["ap50"],
            )

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Report saved to: %s", output_report)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FreshSense YOLO detector"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")

    data_yaml = args.data or (config.detection_dataset.detection_data_dir / "data.yaml")
    output_report = args.output or Path("reports/detection_evaluation.json")

    logger.info("=" * 70)
    logger.info("FreshSense AI - Detector Evaluation")
    logger.info("=" * 70)

    results = evaluate_detector(
        model_path=args.model,
        data_yaml=data_yaml,
        split=args.split,
        output_report=output_report,
    )

    logger.info("=" * 70)
    logger.info("Evaluation complete!")
    logger.info("=" * 70)

    if results["status"] == "available" and results["map50"] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate trained YOLO detector on test split."""


import argparse
import json
import logging
import sys
from pathlib import Path

from configs.config import Config
from src.utils.environment import load_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_detector(
    model_path: Path,
    data_yaml: Path,
    split: str = "test",
    output_report: Path | None = None,
) -> dict:
    """Evaluate YOLO model and return metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics required. Install: pip install ultralytics")
        sys.exit(1)

    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    if not data_yaml.exists():
        logger.error("data.yaml not found: %s", data_yaml)
        sys.exit(1)

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    logger.info("Running evaluation on %s split...", split)
    metrics = model.val(data=str(data_yaml), split=split, verbose=False)

    results = {
        "status": "available",
        "model": str(model_path),
        "dataset": str(data_yaml),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {},
    }

    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        class_names = model.names
        for idx, ap in enumerate(metrics.box.maps):
            if idx < len(class_names):
                results["per_class"][class_names[idx]] = {
                    "precision": float(metrics.box.p[idx])
                    if hasattr(metrics.box, "p") and metrics.box.p is not None
                    else 0.0,
                    "recall": float(metrics.box.r[idx])
                    if hasattr(metrics.box, "r") and metrics.box.r is not None
                    else 0.0,
                    "ap50": float(ap),
                }

    logger.info("=" * 70)
    logger.info("Evaluation Results:")
    logger.info("  Precision:      %.4f", results["precision"])
    logger.info("  Recall:         %.4f", results["recall"])
    logger.info("  mAP@50:         %.4f", results["map50"])
    logger.info("  mAP@50-95:      %.4f", results["map50_95"])
    logger.info("=" * 70)

    if results["per_class"]:
        logger.info("Per-class metrics:")
        for cls, metrics_dict in results["per_class"].items():
            logger.info(
                "  %s: P=%.4f R=%.4f AP50=%.4f",
                cls,
                metrics_dict["precision"],
                metrics_dict["recall"],
                metrics_dict["ap50"],
            )

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Report saved to: %s", output_report)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FreshSense YOLO detector"
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--data", type=Path, default=None, help="Path to data.yaml")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON report",
    )

    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")

    data_yaml = args.data or (config.detection_dataset.detection_data_dir / "data.yaml")
    output_report = args.output or Path("reports/detection_evaluation.json")

    logger.info("=" * 70)
    logger.info("FreshSense AI - Detector Evaluation")
    logger.info("=" * 70)

    results = evaluate_detector(
        model_path=args.model,
        data_yaml=data_yaml,
        split=args.split,
        output_report=output_report,
    )

    logger.info("=" * 70)
    logger.info("Evaluation complete!")
    logger.info("=" * 70)

    if results["status"] == "available" and results["map50"] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
