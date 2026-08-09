"""Baseline evaluation for FreshSense."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from configs.config import Config
from src.preprocessing.augmentation import AugmentationPipeline
from src.preprocessing.dataset import FreshSenseDatasetLoader
from src.training.evaluate import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    config = Config()
    augmentation = AugmentationPipeline(
        image_size=(config.data.image_size, config.data.image_size)
    )
    loader = FreshSenseDatasetLoader(
        dataset_path=config.paths.data_dir / "raw",
        train_transform=augmentation.train_transforms(),
        val_transform=augmentation.validation_transforms(),
        test_transform=augmentation.test_transforms(),
        batch_size=config.data.batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=config.data.drop_last,
        random_state=config.random_seed,
    )
    train_loader, val_loader, test_loader, info = loader.create_dataloaders()
    class_names = info.class_names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = config.paths.best_model_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from src.models.efficientnet import FreshSenseEfficientNet
    model_cfg = checkpoint.get("config_dict", {}).get("model", {})
    model = FreshSenseEfficientNet(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
        dropout=model_cfg.get("dropout", 0.3),
        classifier_hidden=model_cfg.get("classifier_hidden", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=config.paths.metrics_dir / "baseline",
    )
    results = evaluator.evaluate()
    out = {
        "accuracy": results.accuracy,
        "precision": results.precision,
        "recall": results.recall,
        "f1": results.f1,
        "per_class_precision": results.per_class_precision,
        "per_class_recall": results.per_class_recall,
        "per_class_f1": results.per_class_f1,
        "confusion_matrix": results.confusion_matrix.tolist(),
        "num_samples": results.num_samples,
    }
    out_path = config.paths.metrics_dir / "baseline" / "baseline_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved baseline metrics to", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
