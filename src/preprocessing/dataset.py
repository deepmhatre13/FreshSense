"""Dataset loading and splitting for the FreshSense pipeline.

Supports automatic layout detection for multiple dataset structures:

- Layout A (flat): ``data/raw/<class>/*.jpg``
- Layout B (split): ``data/raw/{train,val,test}/<class>/*.jpg``
- Layout C (train/test only): ``data/raw/{train,test}/<class>/*.jpg``
  (validation is created by stratified split from train)
- Layout D (nested): ``data/raw/<any>/<any>/.../{train,test}/<class>/*.jpg``
  (the loader walks down until it finds the first ``train`` or ``test``
  directory, or falls back to class folders)

Key invariants:

1. Each image is loaded **exactly once** for quality checking.
2. Invalid/corrupted images are dropped at scan time.
3. Train/val/test splits are stratified and deterministic.
4. If pre-existing splits are detected (Layout B/C), they are used as-is.
   If only a flat layout or train-only layout exists, the data is split
   according to ``test_size`` and ``val_size``.
5. Class names are always detected from the **leaf** class folders, not
   intermediate container folders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.preprocess import ImagePreprocessor
from src.preprocessing.quality import ImageQualityChecker, QualityReport

logger = logging.getLogger(__name__)

__all__ = ["FreshSenseDataset", "FreshSenseDatasetLoader", "DatasetInfo", "VALID_EXTENSIONS"]

VALID_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

# Known split folder names (case-insensitive).
_SPLIT_NAMES = {"train", "val", "validation", "test"}


@dataclass(frozen=True)
class DatasetInfo:
    """Summary of the scanned dataset, including valid paths and labels."""

    image_paths: List[Path]
    labels: List[int]
    class_names: List[str]
    class_to_idx: Dict[str, int]
    class_distribution: Dict[str, int]
    total_scanned: int
    valid_count: int = field(init=False)
    skipped_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid_count", len(self.image_paths))
        object.__setattr__(
            self,
            "skipped_count",
            self.total_scanned - len(self.image_paths),
        )

    def class_imbalance_report(self) -> str:
        """Return a human-readable class imbalance report.

        Reports per-class counts, percentages, and the min/max ratio.
        """
        if not self.class_distribution:
            return "No class distribution available."

        total = sum(self.class_distribution.values())
        lines = ["Class imbalance report:"]
        for cls, count in sorted(
            self.class_distribution.items(), key=lambda kv: kv[1], reverse=True
        ):
            pct = 100.0 * count / total if total > 0 else 0.0
            lines.append(f"  {cls:20s} {count:6d}  ({pct:5.1f}%)")

        counts = list(self.class_distribution.values())
        if counts:
            min_c, max_c = min(counts), max(counts)
            ratio = max_c / min_c if min_c > 0 else float("inf")
            lines.append(f"  Min/Max ratio: {ratio:.2f}")
        return "\n".join(lines)


class FreshSenseDataset(Dataset):
    """PyTorch Dataset that loads, preprocesses, and transforms images.

    Args:
        image_paths: List of valid image paths.
        labels: Corresponding integer labels.
        transform: Albumentations Compose pipeline (optional).
        preprocessor: ImagePreprocessor instance (optional).

    Raises:
        ValueError: If ``image_paths`` and ``labels`` have different lengths.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform=None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length.")

        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform
        self.preprocessor = preprocessor or ImagePreprocessor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            # Should not happen (filtered at scan time), but be defensive.
            raise RuntimeError(f"Image became unreadable: {image_path}")

        image = self.preprocessor.preprocess(image)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class FreshSenseDatasetLoader:
    """Scans, filters, splits, and loads the FreshSense dataset.

    Automatically detects the dataset layout (flat, split, nested) and
    locates class folders without requiring hardcoded paths.

    Args:
        dataset_path: Root directory containing class subdirectories or
            split subdirectories.
        train_transform: Albumentations pipeline for training.
        val_transform: Albumentations pipeline for validation.
        test_transform: Albumentations pipeline for testing.
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader worker processes.
        pin_memory: If True, use pinned memory for GPU transfer.
        persistent_workers: If True, keep workers alive between epochs.
        prefetch_factor: Number of batches prefetched per worker.
        drop_last: If True, drop the last incomplete batch.
        test_size: Fraction of data for the test split (when no test split
            exists on disk).
        val_size: Fraction of data for the validation split (when no val
            split exists on disk).
        random_state: Seed for reproducible splits and shuffling.
        quality_checker: ImageQualityChecker instance (optional).
        preprocessor: ImagePreprocessor instance (optional).

    Raises:
        FileNotFoundError: If ``dataset_path`` does not exist.
        ValueError: If ``persistent_workers`` is True with ``num_workers == 0``.
        ValueError: If no class directories or no valid images are found.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        train_transform=None,
        val_transform=None,
        test_transform=None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        quality_checker: Optional[ImageQualityChecker] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        if num_workers == 0 and persistent_workers:
            raise ValueError(
                "persistent_workers=True requires num_workers >= 1. "
                "Set num_workers>=1 or persistent_workers=False."
            )

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.quality_checker = quality_checker or ImageQualityChecker()
        self.preprocessor = preprocessor or ImagePreprocessor()

    # ------------------------------------------------------------------
    # Layout detection
    # ------------------------------------------------------------------

    def _resolve_dataset_root(self) -> Path:
        """Locate the actual dataset root, handling nested structures.

        Walks down into subdirectories until it finds a directory that
        contains either a split folder (train/test/val) or class folders
        with images. Falls back to the original path.

        Returns:
            The resolved dataset root path.
        """
        root = self.dataset_path
        # Walk down at most 5 levels to avoid infinite recursion on bad paths.
        for _ in range(5):
            children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
            # If we see split folders or class folders, this is the root.
            has_split = bool(children.keys() & _SPLIT_NAMES)
            if has_split:
                return root
            # If any child directory has images directly (not in subdirs),
            # this is the class level. Using iterdir() instead of rglob("*")
            # prevents container folders like `dataset/dataset/` from being
            # misidentified as class folders when images live deeper.
            class_children = [p for p in root.iterdir() if p.is_dir()]
            if class_children and any(
                any(
                    f.suffix.lower() in VALID_EXTENSIONS
                    for f in p.iterdir()
                    if f.is_file()
                )
                for p in class_children
            ):
                return root
            # Otherwise descend one level: pick the first subdirectory.
            next_dirs = [p for p in root.iterdir() if p.is_dir()]
            if not next_dirs:
                break
            root = next_dirs[0]
        return root

    @staticmethod
    def _is_split_dir(name: str) -> Optional[str]:
        """Return the canonical split name if ``name`` is a split folder."""
        lower = name.lower()
        if lower == "train":
            return "train"
        if lower in ("val", "validation"):
            return "val"
        if lower == "test":
            return "test"
        return None

    def _detect_layout(
        self, root: Path
    ) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Dict[str, Path]]:
        """Detect train/val/test split folders and class folders.

        Returns:
            ``(train_dir, val_dir, test_dir, class_name_to_dir)`` where
            each split path is ``None`` if not found, and
            ``class_name_to_dir`` maps class names to their class folders
            within the detected split (or within ``root`` for flat layouts).
        """
        children = {p.name: p for p in root.iterdir() if p.is_dir()}

        # Check for named split folders (Layout B/C/D).
        train_dir = val_dir = test_dir = None
        for name, path in children.items():
            split = self._is_split_dir(name)
            if split == "train":
                train_dir = path
            elif split == "val":
                val_dir = path
            elif split == "test":
                test_dir = path

        if train_dir is not None or test_dir is not None:
            # Determine class folders from the train split (fall back to test).
            class_name_to_dir = self._discover_class_folders(
                train_dir or test_dir
            )
            return train_dir, val_dir, test_dir, class_name_to_dir

        # Flat layout: every immediate subdirectory is a class (Layout A).
        class_name_to_dir = {p.name: p for p in children.values()}
        return None, None, None, class_name_to_dir

    def _discover_class_folders(self, split_dir: Path) -> Dict[str, Path]:
        """Discover class folders inside a split directory.

        Returns:
            Mapping of class name -> class directory path.
        """
        if not split_dir.exists() or not split_dir.is_dir():
            return {}
        return {p.name: p for p in split_dir.iterdir() if p.is_dir()}

    # ------------------------------------------------------------------
    # Scanning and filtering (each image loaded exactly once)
    # ------------------------------------------------------------------

    def _quality_filter(
        self, image_files: List[Path]
    ) -> Tuple[List[Path], int, Dict[str, int]]:
        """Run quality checks on a list of image files.

        Returns:
            ``(valid_paths, total_scanned, class_counts)``
        """
        valid_paths: List[Path] = []
        class_counts: Dict[str, int] = {}
        total_scanned = 0

        for image_file in image_files:
            total_scanned += 1
            report: QualityReport = self.quality_checker.validate(image_file)
            if not report.is_valid:
                continue
            # Use the parent folder name as the class name.
            class_name = image_file.parent.name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            valid_paths.append(image_file)

        return valid_paths, total_scanned, class_counts

    def scan_dataset(self, root: Optional[Path] = None) -> DatasetInfo:
        """Scan the dataset, filter invalid images, and report statistics.

        Automatically detects the dataset layout and discovers classes from
        leaf class folders, not intermediate container folders.

        Args:
            root: Override the dataset root (used internally when
                ``create_datasets`` resolves nested layouts).

        Returns:
            A DatasetInfo with valid paths, labels, and class statistics.

        Raises:
            ValueError: If no class directories or no valid images are found.
        """
        root = root or self._resolve_dataset_root()
        train_dir, val_dir, test_dir, class_name_to_dir = self._detect_layout(root)

        if not class_name_to_dir:
            raise ValueError(
                f"No class directories found under {root}. "
                "Expected folders named by class (e.g. freshapples, rottenbanana)."
            )

        # Ordered class list + index mapping.
        class_names = sorted(class_name_to_dir.keys())
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        # Collect all image files across class folders.
        image_files: List[Path] = []
        labels: List[int] = []
        class_counts = {cls: 0 for cls in class_names}
        total_scanned = 0

        # For flat layouts or when no split folders exist, scan root directly.
        scan_roots = (
            [root] if train_dir is None and test_dir is None else []
        )
        for split_root in scan_roots:
            for class_name, class_dir in class_name_to_dir.items():
                if not class_dir.exists():
                    continue
                for image_file in class_dir.rglob("*"):
                    if image_file.suffix.lower() not in VALID_EXTENSIONS:
                        continue
                    if not image_file.is_file():
                        continue
                    image_files.append(image_file)
                    labels.append(class_to_idx[class_name])

        # For split layouts, scan train + test (val is derived from train).
        for split_dir in filter(None, [train_dir, test_dir]):
            for class_name, class_dir in class_name_to_dir.items():
                target = split_dir / class_name
                if not target.exists():
                    continue
                for image_file in target.rglob("*"):
                    if image_file.suffix.lower() not in VALID_EXTENSIONS:
                        continue
                    if not image_file.is_file():
                        continue
                    image_files.append(image_file)
                    labels.append(class_to_idx[class_name])

        if not image_files:
            raise ValueError(
                f"No candidate images found under {root} (or its split folders). "
                "Ensure class folders contain .jpg/.png/.bmp/.webp files."
            )

        # Quality filter (each image loaded exactly once).
        valid_paths, total_scanned, class_counts = self._quality_filter(image_files)

        # Build labels for valid images only.
        valid_labels = [
            class_to_idx[p.parent.name] for p in valid_paths
        ]

        if not valid_paths:
            raise ValueError(
                f"No valid images found in {root} after quality filtering."
            )

        logger.info(
            "Scanned %d images: %d valid, %d skipped.",
            total_scanned,
            len(valid_paths),
            total_scanned - len(valid_paths),
        )
        logger.info("Class distribution: %s", class_counts)

        return DatasetInfo(
            image_paths=valid_paths,
            labels=valid_labels,
            class_names=class_names,
            class_to_idx=class_to_idx,
            class_distribution=class_counts,
            total_scanned=total_scanned,
        )

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _split_from_existing(
        self, root: Path, info: DatasetInfo
    ) -> Tuple[List[Path], List[int], List[Path], List[int], List[Path], List[int]]:
        """Build train/val/test from pre-existing split folders on disk.

        If ``val/`` does not exist but ``train/`` and ``test/`` do, a
        stratified validation split is created from the training set.

        Returns:
            ``(train_paths, train_labels, val_paths, val_labels,
               test_paths, test_labels)``
        """
        train_dir, val_dir, test_dir, class_name_to_dir = self._detect_layout(root)

        def _collect(split_dir: Optional[Path]) -> Tuple[List[Path], List[int]]:
            paths: List[Path] = []
            labels: List[int] = []
            if split_dir is None or not split_dir.exists():
                return paths, labels
            for class_name, class_dir in class_name_to_dir.items():
                target = split_dir / class_name
                if not target.exists():
                    continue
                for image_file in target.rglob("*"):
                    if image_file.suffix.lower() not in VALID_EXTENSIONS:
                        continue
                    if not image_file.is_file():
                        continue
                    # Only include images that passed quality filtering.
                    if image_file in info.image_paths:
                        paths.append(image_file)
                        labels.append(info.class_to_idx[class_name])
            return paths, labels

        train_paths, train_labels = _collect(train_dir)
        test_paths, test_labels = _collect(test_dir)

        # Validation: use on-disk val split if present; otherwise derive from train.
        if val_dir is not None and val_dir.exists():
            val_paths, val_labels = _collect(val_dir)
        else:
            # Stratified split from the training set only.
            if not train_paths:
                raise ValueError("No training images found to derive validation split.")
            val_ratio = self.val_size / (1.0 - self.test_size)
            # Stratified split requires at least n_classes samples in the
            # validation fold. Clamp the ratio so tiny datasets still work,
            # and warn the user so they can increase dataset size or reduce
            # val_size.
            n_classes = len(set(train_labels))
            # Try stratified split first; if the dataset is too small for
            # sklearn to honor stratify (it needs at least n_classes samples
            # in the validation fold), fall back to a plain split.
            try:
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    train_paths,
                    train_labels,
                    test_size=val_ratio,
                    stratify=train_labels,
                    random_state=self.random_state,
                )
            except ValueError:
                logger.warning(
                    "Dataset too small for stratified validation split "
                    "(%d training samples, %d classes). Falling back to "
                    "non-stratified split. For better results, increase "
                    "dataset size or reduce val_size.",
                    len(train_paths),
                    n_classes,
                )
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    train_paths,
                    train_labels,
                    test_size=val_ratio,
                    random_state=self.random_state,
                )

        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

    def create_datasets(self) -> Tuple[FreshSenseDataset, FreshSenseDataset, FreshSenseDataset, DatasetInfo]:
        """Create train/val/test datasets from the scanned data.

        Returns:
            ``(train_dataset, val_dataset, test_dataset, dataset_info)``
        """
        root = self._resolve_dataset_root()
        train_dir, val_dir, test_dir, _ = self._detect_layout(root)

        # If pre-existing split folders exist on disk, use them directly.
        if train_dir is not None or test_dir is not None:
            info = self.scan_dataset(root=root)
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = (
                self._split_from_existing(root, info)
            )
        else:
            # Flat layout: scan + stratify from scratch.
            info = self.scan_dataset(root=root)
            train_images, test_images, train_labels, test_labels = train_test_split(
                info.image_paths,
                info.labels,
                test_size=self.test_size,
                stratify=info.labels,
                random_state=self.random_state,
            )

            val_ratio = self.val_size / (1.0 - self.test_size)
            train_images, val_images, train_labels, val_labels = train_test_split(
                train_images,
                train_labels,
                test_size=val_ratio,
                stratify=train_labels,
                random_state=self.random_state,
            )

            train_paths, val_paths, test_paths = train_images, val_images, test_images

        train_dataset = FreshSenseDataset(
            train_paths,
            train_labels,
            self.train_transform,
            self.preprocessor,
        )
        val_dataset = FreshSenseDataset(
            val_paths,
            val_labels,
            self.val_transform,
            self.preprocessor,
        )
        test_dataset = FreshSenseDataset(
            test_paths,
            test_labels,
            self.test_transform,
            self.preprocessor,
        )

        # Recompute class distribution from the final splits.
        final_distribution = {
            cls: 0 for cls in info.class_names
        }
        for label in train_labels + val_labels + test_labels:
            final_distribution[info.class_names[label]] += 1

        info = DatasetInfo(
            image_paths=train_paths + val_paths + test_paths,
            labels=train_labels + val_labels + test_labels,
            class_names=info.class_names,
            class_to_idx=info.class_to_idx,
            class_distribution=final_distribution,
            total_scanned=info.total_scanned,
        )

        return train_dataset, val_dataset, test_dataset, info

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _dataloader_kwargs(self) -> dict:
        """Build the common DataLoader kwargs, handling worker edge cases.

        ``persistent_workers`` and ``prefetch_factor`` are only valid when
        ``num_workers > 0``; they are omitted otherwise to avoid a
        ``ValueError`` from PyTorch.
        """
        kwargs: dict = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
        """Create train/val/test DataLoaders.

        Returns:
            ``(train_loader, val_loader, test_loader, dataset_info)``
        """
        train_dataset, val_dataset, test_dataset, info = self.create_datasets()

        common_kwargs = self._dataloader_kwargs()

        # Deterministic shuffling: seed a generator with random_state.
        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=self.drop_last,
            generator=generator,
            **common_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **common_kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **common_kwargs,
        )

        return train_loader, val_loader, test_loader, info

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> None:
        """Verify the dataset is usable by loading one batch from each split.

        This is a lightweight smoke test that catches shape mismatches,
        transform errors, and worker issues before the full training run.

        Raises:
            RuntimeError: If any split fails to produce a batch.
        """
        train_loader, val_loader, test_loader, info = self.create_dataloaders()

        for name, loader in (
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ):
            try:
                images, labels = next(iter(loader))
                logger.info(
                    "Verified %s split: batch shape %s, labels shape %s, "
                    "dtype %s, range [%.3f, %.3f]",
                    name,
                    tuple(images.shape),
                    tuple(labels.shape),
                    images.dtype,
                    images.min().item(),
                    images.max().item(),
                )
            except Exception as exc:  # noqa: BLE001 - surface any loader error
                raise RuntimeError(f"Failed to load a {name} batch: {exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.preprocessing.augmentation import AugmentationPipeline

    pipeline = AugmentationPipeline()
    loader = FreshSenseDatasetLoader(
        dataset_path="data/raw",
        train_transform=pipeline.train_transforms(),
        val_transform=pipeline.validation_transforms(),
        test_transform=pipeline.test_transforms(),
        batch_size=32,
    )

    train_loader, val_loader, test_loader, info = loader.create_dataloaders()

    print("=" * 60)
    print("FreshSense Dataset Loaded Successfully")
    print("=" * 60)
    print(f"Classes           : {info.class_names}")
    print(f"Class Distribution: {info.class_distribution}")
    print(f"Valid Images      : {info.valid_count}")
    print(f"Skipped Images    : {info.skipped_count}")
    print(f"Training Batches  : {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}")
    print(f"Testing Batches   : {len(test_loader)}")
    print()
    print(info.class_imbalance_report())