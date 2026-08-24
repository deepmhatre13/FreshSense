"""Train the 16-class freshness EfficientNet-B0 model.

Canonical dataset: ``data/freshness/`` (train/valid/test, 16 classes).
Produces a VERSIONED checkpoint at ``models/checkpoints/freshness_efficientnet_b0_16class.pth``
and NEVER overwrites ``models/checkpoints/best_model.pth`` (the immutable
production baseline). Evaluation artifacts are written to ``reports/freshness/``.

Training config follows the repository's established Phase-1 pipeline:
EfficientNet-B0 transfer learning, AdamW, ReduceLROnPlateau, early stopping,
best-checkpoint saving, AMP when CUDA available. Class count is derived from
``data/freshness/class_mapping.json`` (the single classification taxonomy
source of truth).

Usage:
    python scripts/train_freshness_16.py [--epochs N] [--batch-size N] [--lr X]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.efficientnet import FreshSenseEfficientNet
from src.preprocessing.augmentation import AugmentationPipeline
from src.preprocessing.preprocess import ImagePreprocessor
from src.training.evaluate import Evaluator
from src.training.trainer import Trainer

logger = logging.getLogger("train_freshness_16")
SEED = 42
IMGS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_mapping() -> list[str]:
    cm_path = ROOT / "data" / "freshness" / "class_mapping.json"
    with open(cm_path, encoding="utf-8") as f:
        raw = json.load(f)
    ordered = [None] * len(raw)
    for k, v in raw.items():
        ordered[int(k)] = v
    return [c for c in ordered if c]
def build_dataset_rows(split: str, class_names: list[str]):
    base = ROOT / "data" / "freshness" / split
    rows = []
    for i, cls in enumerate(class_names):
        d = base / cls
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMGS:
                rows.append((p, i))
    return rows


def make_loader(rows, class_names, transform, batch_size, num_workers, shuffle):
    from src.preprocessing.dataset import FreshSenseDataset
    paths = [p for p, _ in rows]
    labels = [i for _, i in rows]
    ds = FreshSenseDataset(paths, labels, transform,
                           ImagePreprocessor(image_size=(224, 224)))
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "shuffle": shuffle,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    epochs, batch_size, lr, num_workers = (
        args.epochs, args.batch_size, args.lr, args.workers,
    )

    logging.basicConfig(level=logging.INFO)
    set_seed(SEED)

    class_names = load_class_mapping()
    num_classes = len(class_names)
    logger.info("Loading taxonomy: %d classes", num_classes)
    if num_classes == 0:
        logger.error("class_mapping empty")
        return 1

    aug = AugmentationPipeline(image_size=(224, 224))
    train_rows = build_dataset_rows("train", class_names)
    val_rows = build_dataset_rows("valid", class_names)
    test_rows = build_dataset_rows("test", class_names)
    logger.info("train=%d val=%d test=%d", len(train_rows), len(val_rows), len(test_rows))
    if not train_rows or not val_rows or not test_rows:
        logger.error("Dataset incomplete; cannot train.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FreshSenseEfficientNet(num_classes=num_classes, pretrained=True,
                                   freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    param_groups = model.get_parameter_groups(
        backbone_lr=lr * 0.1, classifier_lr=lr, weight_decay=0.0001,
    )
    optimizer = optim.AdamW(param_groups)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    train_loader = make_loader(train_rows, class_names, aug.get_transforms("train"),
                               batch_size, num_workers, shuffle=True)
    val_loader = make_loader(val_rows, class_names, aug.get_transforms("val"),
                             batch_size, num_workers, shuffle=False)
    test_loader = make_loader(test_rows, class_names, aug.get_transforms("test"),
                              batch_size, num_workers, shuffle=False)

    ckpt_dir = ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "freshness_efficientnet_b0_16class.pth"
    last_path = ckpt_dir / "freshness_efficientnet_b0_16class_last.pth"
    hist_path = ckpt_dir / "training_history_16class.csv"
    metrics_dir = ROOT / "reports" / "freshness"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
        patience=4,
        grad_clip=1.0,
        mixed_precision=torch.cuda.is_available(),
        save_checkpoint_every=5,
        history_csv_path=hist_path,
        resume_from=None,
        class_names=class_names,
        config=None,
    )
    # Route the Trainer's best/last/epoch save points to the versioned path so
    # the immutable production best_model.pth is never touched.
    trainer.best_model_path = best_path
    trainer.last_model_path = last_path

    history = trainer.fit()
    logger.info("Best epoch %d | val_loss %.4f | val_acc %.2f%%",
                history.best_epoch, history.best_val_loss, history.best_val_acc)

    # Evaluate on the held-out test set (untouched until final evaluation).
    evaluator = Evaluator(model=model, test_loader=test_loader, device=device,
                          class_names=class_names, output_dir=metrics_dir)
    results = evaluator.evaluate()
    evaluator.print_results(results)
    evaluator.save_all(results)

    # Persist a compact training_metrics.json summary.
    tmetrics = {
        "model_classifier_type": "1280-256-%d" % num_classes,
        "num_classes": num_classes,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "best_val_acc": history.best_val_acc,
        "test_metrics": results.to_dict(),
    }
    (metrics_dir / "training_metrics.json").write_text(
        json.dumps(tmetrics, indent=2), encoding="utf-8")
    logger.info("Checkpoint: %s", best_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())