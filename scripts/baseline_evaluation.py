#!/usr/bin/env python3
"""Baseline evaluation for the FreshSense six-class freshness model.

Phase 5 - Honest, real-data baseline.

This evaluator measures the model on the **intended freshness classes**
(freshness+fruit combinations, e.g. ``fresh`` + ``apples`` ->
``freshapples``, ``rotten`` + ``banana`` -> ``rottenbanana``) using
real labeled images. It reports:

- total samples
- class distribution
- accuracy
- macro precision / recall / F1
- weighted F1
- balanced accuracy
- per-class precision / recall / F1
- confusion matrix

Data sources (pick one; when unspecified the canonical real-world split
manifest is tried first, then the shipped raw benchmark test split):

1. Canonical real-world split manifest:
   ``--manifest data/real_world/splits/test.csv`` (+ ``--data-root``)
2. Class-folder test set:
   ``--data-root <dir>`` whose immediate sub-folders are the six classes
   (default ``data/raw/dataset/dataset/test`` - the shipped benchmark test
   split).

If no usable checkpoint or dataset is available, every metric is reported as
``"NOT_AVAILABLE"``. No numbers are invented.

CAVEAT printed with every run: the raw benchmark split lacks
``physical_fruit_id`` metadata, so grouped-split leakage cannot be verified
for it. It is real labeled data, but the authoritative benchmark must be
built from the canonical real-world manifest (docs/REAL_WORLD_DATASET.md).

Usage:
    python scripts/baseline_evaluation.py
    python scripts/baseline_evaluation.py --data-root data/raw/dataset/dataset/test
    python scripts/baseline_evaluation.py --max-images 100

Output: ``reports/baseline_evaluation.json`` by default (override with
``--output``). Human summary is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

NOT_AVAILABLE = "NOT_AVAILABLE"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class BaselineResult:
    """Machine-readable baseline evaluation output.

    Every metric defaults to ``NOT_AVAILABLE`` so a missing checkpoint or
    dataset cannot accidentally look like a measured value.
    """

    status: str = NOT_AVAILABLE
    checkpoint: Optional[str] = None
    data_source: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    total_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    accuracy: object = NOT_AVAILABLE
    macro_precision: object = NOT_AVAILABLE
    macro_recall: object = NOT_AVAILABLE
    macro_f1: object = NOT_AVAILABLE
    weighted_f1: object = NOT_AVAILABLE
    balanced_accuracy: object = NOT_AVAILABLE
    per_class_precision: Dict[str, object] = field(default_factory=dict)
    per_class_recall: Dict[str, object] = field(default_factory=dict)
    per_class_f1: Dict[str, object] = field(default_factory=dict)
    confusion_matrix: object = NOT_AVAILABLE

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        return asdict(self)
# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------


def _scan_class_folder(root: Path) -> Tuple[List[Path], List[int], List[str]]:
    """Scan ``root`` for class sub-folders.

    Returns ``(image_paths, labels, class_names)`` where ``labels`` are class
    indices into the sorted ``class_names`` list.
    """
    folders = sorted(
        p for p in root.iterdir()
        if p.is_dir() and any(
            f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            for f in p.iterdir()
        )
    )
    image_paths: List[Path] = []
    labels: List[int] = []
    class_names = [folder.name for folder in folders]
    for idx, folder in enumerate(folders):
        files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        )
        for f in files:
            image_paths.append(f)
            labels.append(idx)
    return image_paths, labels, class_names


def _discover_dataset(
    data_root: Optional[Path],
    manifest: Optional[Path],
    default_root: Path,
) -> Tuple[List[Path], List[str], List[str], str, List[str]]:
    """Resolve the evaluation dataset.

    Returns ``(paths, label_names, class_names, source_description, notes)``.

    Resolution order:
    1. Explicit canonical manifest (``--manifest``).
    2. Explicit class-folder root (``--data-root``).
    3. Canonical split manifest at ``data/real_world/splits/test.csv``.
    4. Class-folder root ``data/raw/dataset/dataset/test``.
    """
    notes: List[str] = []

    def _from_manifest(mpath: Path) -> Tuple[List[Path], List[str], List[str], str]:
        from src.data.real_world_schema import load_canonical_manifest

        records = load_canonical_manifest(mpath)
        base = data_root or mpath.parent
        paths: List[Path] = []
        label_names: List[str] = []
        for rec in records:
            resolved = (base / rec.image_path).resolve()
            if resolved.is_file():
                paths.append(resolved)
                label_names.append(rec.class_name)
            else:
                notes.append(f"missing image file: {rec.image_path}")
        class_names = sorted({n for n in label_names if n})
        return paths, label_names, class_names, f"manifest:{mpath}"

    def _from_folder(folder: Path) -> Tuple[List[Path], List[str], List[str], str]:
        paths, labels, class_names = _scan_class_folder(folder)
        return paths, [class_names[i] for i in labels], class_names, f"class-folder:{folder}"

    candidates = []
    if manifest is not None:
        candidates.append(("manifest", manifest))
    if data_root is not None:
        candidates.append(("folder", data_root))

    for kind, target in candidates:
        if kind == "manifest" and target.exists():
            try:
                paths, labels, class_names, desc = _from_manifest(target)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"could not parse manifest {target}: {exc}")
                continue
            if paths:
                return paths, labels, class_names, desc, notes
            notes.append(f"manifest {target} resolved to zero readable images")
        elif kind == "folder" and target.is_dir():
            try:
                paths, labels, class_names, desc = _from_folder(target)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"could not scan folder {target}: {exc}")
                continue
            if paths:
                return paths, labels, class_names, desc, notes
            notes.append(f"folder {target} contained no images")

    # Defaults: canonical real-world split, then the shipped raw test split.
    if manifest is None:
        default_manifest = Path("data/real_world/splits/test.csv")
        if default_manifest.exists():
            paths, labels, class_names, desc = _from_manifest(default_manifest)
            if paths:
                return paths, labels, class_names, desc, notes
    if data_root is None and default_root.is_dir():
        paths, labels, class_names, desc = _from_folder(default_root)
        if paths:
            return paths, labels, class_names, desc, notes

    return [], [], [], "", notes + [NOT_AVAILABLE]
# ---------------------------------------------------------------------------
# Model + inference
# ---------------------------------------------------------------------------


def _load_model(checkpoint: Path, device: torch.device):
    """Load the EfficientNet classifier and its class names from a checkpoint."""
    from src.models.efficientnet import FreshSenseEfficientNet

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_classes = list(ckpt.get("class_names", []))
    model_cfg = ckpt.get("config_dict", {}).get("model", {})
    model = FreshSenseEfficientNet(
        num_classes=ckpt.get("num_classes", len(ckpt_classes)),
        pretrained=False,
        freeze_backbone=False,
        dropout=model_cfg.get("dropout", 0.3),
        classifier_hidden=model_cfg.get("classifier_hidden", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt_classes


def _infer(
    image_paths: Sequence[Path],
    label_names: Sequence[str],
    model: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    batch_size: int,
    max_images: Optional[int],
) -> Tuple[List[str], List[str], List[Path]]:
    """Run inference and return ``(true_names, pred_names, used_paths)``."""
    from albumentations import Compose, Normalize, Resize
    from albumentations.pytorch import ToTensorV2
    from src.preprocessing.augmentation import IMAGENET_MEAN, IMAGENET_STD

    transform = Compose(
        [
            Resize(height=224, width=224),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    used_paths: List[Path] = []
    true_names: List[str] = []
    pred_names: List[str] = []

    pairs = sorted(zip(image_paths, label_names), key=lambda pair: str(pair[0]))
    if max_images is not None and len(pairs) > max_images:
        stride = len(pairs) / max_images
        pairs = [pairs[int(i * stride)] for i in range(max_images)]

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        batch_paths = [p for p, _ in chunk]
        batch_labels = [lbl for _, lbl in chunk]
        batch_data, kept_indexes = [], []
        for i, path in enumerate(batch_paths):
            import cv2

            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_data.append(transform(image=img)["image"])
            kept_indexes.append(i)
        if not batch_data:
            continue
        batch_tensor = torch.stack(batch_data).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
        for offset, batch_index in enumerate(kept_indexes):
            used_paths.append(batch_paths[batch_index])
            true_names.append(batch_labels[batch_index])
            pred_idx = preds[offset]
            pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else "unknown"
            pred_names.append(pred_name)
    return true_names, pred_names, used_paths


def _compute_metrics(
    true_names: Sequence[str],
    pred_names: Sequence[str],
    class_names: List[str],
    total_available: int,
) -> BaselineResult:
    """Compute all required metrics and fill a :class:`BaselineResult`."""
    result = BaselineResult()
    result.total_samples = len(true_names)
    result.class_distribution = {cls: int(true_names.count(cls)) for cls in class_names}
    if result.total_samples == 0 or not class_names:
        return result

    y_true = np.array([class_names.index(t) for t in true_names])
    y_pred = np.array(
        [class_names.index(p) if p in class_names else -1 for p in pred_names]
    )

    labels = list(range(len(class_names)))
    vals_format = lambda v: round(float(v), 6)  # noqa: E731

    result.accuracy = vals_format(accuracy_score(y_true, y_pred))
    result.macro_precision = vals_format(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result.macro_recall = vals_format(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result.macro_f1 = vals_format(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result.weighted_f1 = vals_format(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    result.balanced_accuracy = vals_format(
        balanced_accuracy_score(y_true, y_pred)
    )

    per_class_p = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    result.per_class_precision = {cls: vals_format(v) for cls, v in zip(class_names, per_class_p)}
    result.per_class_recall = {cls: vals_format(v) for cls, v in zip(class_names, per_class_r)}
    result.per_class_f1 = {cls: vals_format(v) for cls, v in zip(class_names, per_class_f)}

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result.confusion_matrix = cm.tolist()
    result.status = "OK"
    return result
def _print_summary(result: BaselineResult, class_names: List[str]) -> None:
    """Print a concise human-readable summary to stdout."""
    bar = "=" * 68
    print(bar)
    print("BASELINE EVALUATION (Phase 5 - intended freshness classes)")
    print(bar)
    print(f"Status            : {result.status}")
    print(f"Checkpoint        : {result.checkpoint}")
    print(f"Data source       : {result.data_source}")
    print(f"Total samples     : {result.total_samples}")
    if result.status == NOT_AVAILABLE:
        print("\nMetrics: NOT AVAILABLE (no real dataset / checkpoint).")
        for note in result.notes:
            print(f"  - {note}")
        return

    print("Class distribution :")
    for cls, cnt in sorted(result.class_distribution.items()):
        print(f"  {cls:20s} {cnt}")
    print()
    print(f"Accuracy          : {result.accuracy}")
    print(f"Macro precision   : {result.macro_precision}")
    print(f"Macro recall      : {result.macro_recall}")
    print(f"Macro F1          : {result.macro_f1}")
    print(f"Weighted F1       : {result.weighted_f1}")
    print(f"Balanced accuracy : {result.balanced_accuracy}")
    print()
    print("Per-class metrics:")
    print(f"  {'class':20s} {'precision':>10s} {'recall':>10s} {'f1':>10s}")
    for cls in class_names:
        print(
            f"  {cls:20s} {result.per_class_precision[cls]:>10} "
            f"{result.per_class_recall[cls]:>10} {result.per_class_f1[cls]:>10}"
        )
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(f"  classes: {class_names}")
    cm = result.confusion_matrix
    if isinstance(cm, list):
        print(np.asarray(cm, dtype=object))
    else:
        print(cm)
    for note in result.notes:
        print(f"NOTE: {note}")
def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Baseline evaluation on the intended six freshness classes."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/best_model.pth"),
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Class-folder test set (immediate sub-folders are classes).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Canonical real-world split manifest (CSV/JSON).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the number of images evaluated (for smoke runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/baseline_evaluation.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--print-images",
        action="store_true",
        help="Print every evaluated image (debug only).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = BaselineResult()
    result.checkpoint = str(args.checkpoint)

    # --- dataset -----------------------------------------------------------
    default_root = Path("data/raw/dataset/dataset/test")
    paths, label_names, class_names, source, notes = _discover_dataset(
        data_root=args.data_root, manifest=args.manifest, default_root=default_root
    )
    result.data_source = source or None
    result.notes = notes

    if not args.checkpoint.exists():
        result.notes.insert(0, f"checkpoint not found: {args.checkpoint}")
        _print_summary(result, class_names)
        _write_output(result, args.output)
        return 1

    if not paths or not class_names:
        result.notes.insert(
            0,
            "no real evaluation dataset available (no manifest, no class-folder "
            "test set).",
        )
        _print_summary(result, class_names)
        _write_output(result, args.output)
        return 1

    # --- model --------------------------------------------------------------
    model, ckpt_classes = _load_model(args.checkpoint, device)
    if ckpt_classes and set(ckpt_classes) != set(class_names):
        result.notes.append(
            "checkpoint class_names "
            f"({sorted(ckpt_classes)}) differ from the dataset classes "
            f"({sorted(class_names)}); treating checkpoint indices as its own "
            "class order."
        )
    model_class_names = ckpt_classes or class_names
# --- inference ------------------------------------------------------------
    true_names, pred_names, used_paths = _infer(
        image_paths=paths,
        label_names=label_names,
        model=model,
        device=device,
        class_names=model_class_names,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )
    if args.print_images:
        for path, t, p in zip(used_paths, true_names, pred_names):
            print(f"{path}: true={t} pred={p}")

    # --- metrics --------------------------------------------------------------
    measured = _compute_metrics(true_names, pred_names, class_names, len(paths))
    measured.checkpoint = result.checkpoint
    measured.data_source = result.data_source
    measured.notes = result.notes
    if args.max_images is not None:
        measured.notes.append(
            f"partial run capped at {args.max_images} images; not the full set."
        )
    measured.notes.append(
        "The default raw benchmark split lacks physical_fruit_id metadata, so "
        "grouped-split leakage cannot be verified for it; build the canonical "
        "real-world benchmark (docs/REAL_WORLD_DATASET.md) for authoritative "
        "numbers."
    )

    _print_summary(measured, class_names)
    _write_output(measured, args.output)
    return 0


def _write_output(result: BaselineResult, output: Path) -> None:
    """Write the result to ``output`` as machine-readable JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    sys.exit(main())
