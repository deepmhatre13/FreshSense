#!/usr/bin/env python3
"""Download the FreshSense fruit detection dataset from Roboflow.

Downloads a YOLOv8-format export of the configured Roboflow dataset version,
then normalizes the resulting ``data.yaml`` and label files so ultralytics
resolves the split paths and annotations correctly.

Roboflow exports use a ``valid`` folder and ``../train/images`` style paths in
``data.yaml``; ultralytics resolves relative paths against the *yaml file*
location, so ``../train/images`` would point one level too high. This script
rewrites each split to ``<split>/images`` based on the *actual on-disk layout*
and exposes a ``val`` key (ultralytics' preferred name) while tolerating the
``valid`` folder. Polygon/segmentation label rows are also converted to
axis-aligned YOLO box rows.

Security:
  - API key is read from ``--api-key`` or ``ROBOFLOW_API_KEY``; never printed.
  - The manifest written to disk contains non-secret fields only.

Usage:
    python scripts/download_detection_dataset.py --workspace deepam-mhatre \
        --project fruits-test-ajvf8-duncc --version 1 --format yolov8 --overwrite
    python scripts/download_detection_dataset.py --normalize-only \
        --output data/detection --manifest reports/detection_dataset_download.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Module-level import so tests can patch ``scripts.download_detection_dataset.Roboflow``.
# The lazy local import is removed in favour of this single top-level import.
try:
    from roboflow import Roboflow
except ImportError:
    Roboflow = None  # type: ignore[assignment]

# Allow running directly (python scripts/<name>.py) from any CWD.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.config import Config
from src.utils.environment import load_environment, require_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Canonical split key -> accepted folder names on disk. ultralytics uses the
# "val" key, while Roboflow exports sometimes use a "valid" folder/key; both
# are accepted here.
SPLIT_FOLDERS = {
    "train": ("train",),
    "val": ("valid", "val"),
    "test": ("test",),
}


def normalize_data_yaml(data_yaml: Path) -> dict:
    """Rewrite ``data.yaml`` split paths to be relative to the yaml directory.

    Roboflow YOLO exports sometimes use ``../train/images`` style paths.
    ultralytics resolves relative paths against the directory that contains
    ``data.yaml``, so we re-express every split as ``<folder>/images`` based on
    the actual on-disk layout. Validation is canonicalized under ultralytics'
    preferred ``val`` key (tolerating a ``valid`` folder and/or stale ``../``
    paths).
    """
    if not data_yaml.exists():
        logger.warning("data.yaml not found at %s - skipping normalization", data_yaml)
        return {}

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    root = data_yaml.parent

    for canonical, folders in SPLIT_FOLDERS.items():
        for folder in folders:
            if (root / folder / "images").is_dir():
                cfg[canonical] = f"{folder}/images"
                break

    # Drop any stale "valid" alias so ultralytics only sees the canonical "val".
    cfg.pop("valid", None)

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info("Normalized data.yaml split paths relative to: %s", root)
    return cfg


def normalize_labels(root: str | Path) -> dict:
    """Convert polygon/segmentation label rows to YOLO box rows in place.

    ``root`` may be given as a ``str`` or ``Path``; it is normalised to a
    ``Path`` immediately so all filesystem operations are reliable.

    Roboflow exports occasionally mix polygon (variable-length, ``class_id
    x1 y1 x2 y2 ...``) rows into box-label files. ultralytics expects exactly
    5 fields per row (``{class_id} x_center y_center w h``). Each polygon row
    is replaced by its tightest axis-aligned bounding box; already-correct box
    rows are left untouched, making the operation idempotent.

    Returns a small audit dict with ``converted_rows`` and ``files_changed``.
    """
    root = Path(root)
    converted = 0
    files_changed = 0
    for split in ("train", "valid", "val", "test"):
        labels_dir = root / split / "labels"
        if not labels_dir.is_dir():
            continue
        for label_file in sorted(labels_dir.glob("*.txt")):
            lines = label_file.read_text(encoding="utf-8").splitlines()
            out: list[str] = []
            changed = False
            for line in lines:
                tokens = line.split()
                if not tokens:
                    out.append(line)
                    continue
                class_id = tokens[0]
                rest = tokens[1:]
                if len(rest) == 4:
                    # Already a YOLO box row; keep verbatim.
                    out.append(line)
                elif len(rest) % 2 == 0 and len(rest) >= 6:
                    # Polygon: class_id + N (x, y) vertex pairs.
                    xs = [float(v) for v in rest[0::2]]
                    ys = [float(v) for v in rest[1::2]]
                    x_c = (min(xs) + max(xs)) / 2.0
                    y_c = (min(ys) + max(ys)) / 2.0
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    x_c = min(max(x_c, 0.0), 1.0)
                    y_c = min(max(y_c, 0.0), 1.0)
                    w = min(max(w, 0.0), 1.0)
                    h = min(max(h, 0.0), 1.0)
                    out.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                    converted += 1
                    changed = True
                else:
                    # Unknown row format; keep as-is (validator will flag it).
                    out.append(line)
            if changed:
                label_file.write_text("\n".join(out) + "\n", encoding="utf-8")
                files_changed += 1
    logger.info(
        "Label normalization: converted %d polygon rows across %d file(s).",
        converted,
        files_changed,
    )
    return {"converted_rows": converted, "files_changed": files_changed}



def find_data_yaml(output_dir: str | Path) -> Path:
    """Locate ``data.yaml`` after a download (handles version sub-folders).

    ``output_dir`` may be given as a ``str`` or ``Path``; it is normalised to a
    ``Path`` immediately so all filesystem operations are reliable.
    """
    output_dir = Path(output_dir)
    data_yaml = output_dir / "data.yaml"
    if data_yaml.is_file():
        return data_yaml
    candidates = sorted(p for p in output_dir.rglob("data.yaml") if p.is_file())
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"data.yaml not found under {output_dir}")


def _write_manifest(
    manifest_path: Path,
    workspace: str,
    project: str,
    version: int,
    fmt: str,
    output_dir: Path,
    data_yaml: Path,
    label_audit: dict | None = None,
) -> dict:
    cfg: dict = {}
    if Path(data_yaml).is_file():
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    names = cfg.get("names") or []
    manifest = {
        "tool": "FreshSenseAI Roboflow Detection Dataset Downloader",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request": {
            "workspace": workspace,
            "project": project,
            "version": version,
            "format": fmt,
            "output_dir": str(output_dir),
        },
        "data_yaml": str(data_yaml),
        "dataset_root": str(Path(data_yaml).parent),
        "nc": cfg.get("nc"),
        "num_classes": len(names) if isinstance(names, list) else len(names),
        "classes": names,
        "splits": {k: v for k, v in cfg.items() if k in ("train", "val", "valid", "test")},
        "label_audit": label_audit or {},
        "notes": [
            "No secrets are stored in this manifest.",
            "data.yaml split paths are normalized relative to the dataset root.",
            "Polygon label rows converted to axis-aligned bounding boxes.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Download manifest saved to: %s", manifest_path)
    return manifest


def download_dataset(
    workspace: str,
    project: str,
    version: int,
    output_dir: str | Path,
    api_key: str,
    format: str = "yolov8",
    overwrite: bool = False,
) -> tuple[Path, dict]:
    """Download dataset from Roboflow; return ``(data_yaml, label_audit)``.

    ``output_dir`` may be given as a ``str`` or ``Path``; it is normalized to a
    ``Path`` immediately so all filesystem operations below are reliable.
    """
    # Normalize to a Path at the boundary so callers may pass either a str or a
    # Path. Every filesystem operation below (.mkdir, .is_file, .rglob, / joins)
    # then works consistently.
    output_dir = Path(output_dir)

    if Roboflow is None:
        logger.error("roboflow package required. Install: pip install roboflow")
        sys.exit(1)

    logger.info("Connecting to Roboflow workspace: %s", workspace)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    version = project.version(version)

    logger.info("Downloading dataset (format=%s) to %s", format, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = version.download(format, location=str(output_dir), overwrite=overwrite)

    data_yaml = find_data_yaml(output_dir)
    normalize_data_yaml(data_yaml)
    # Normalize labels relative to data.yaml's *actual* directory (the dataset
    # root), not ``output_dir``. Roboflow's version.download() may place the
    # export inside a version-named subfolder (find_data_yaml handles that via
    # rglob), so data_yaml can resolve to ``output_dir/<subfolder>/data.yaml``.
    # Scanning output_dir directly would silently skip the split/label folders
    # (train/labels, valid/labels, test/labels), skipping polygon->box
    # conversion and label validation on every nested download. This mirrors the
    # --normalize-only branch (normalize_labels(Path(data_yaml).parent)) and the
    # manifest's dataset_root, keeping all three consistent.
    label_audit = normalize_labels(data_yaml.parent)

    dataset_root = Path(dataset.location) if hasattr(dataset, "location") else output_dir
    logger.info("Download complete! Dataset root: %s", dataset_root)
    logger.info("data.yaml: %s", data_yaml)
    logger.info("Classes: %s", dataset.classes if hasattr(dataset, "classes") else "N/A")

    return data_yaml, label_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FreshSense detection dataset from Roboflow"
    )
    parser.add_argument("--workspace", type=str, default=None, help="Roboflow workspace name")
    parser.add_argument("--project", type=str, default=None, help="Roboflow project name")
    parser.add_argument("--version", type=int, default=None, help="Dataset version number")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for dataset")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY)",
    )
    parser.add_argument("--format", type=str, default="yolov8", help="Export format (default: yolov8)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if the dataset exists locally")
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Normalize data.yaml and label files in an already-downloaded dataset (no network fetch)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/detection_dataset_download.json"),
        help="Path to save the download manifest JSON",
    )

    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")

    workspace = args.workspace or config.detection_dataset.roboflow_workspace
    project = args.project or config.detection_dataset.roboflow_project
    version = args.version or config.detection_dataset.roboflow_version
    # ``--output`` arrives as a Path (argparse type=Path) while the config value
    # ``detection_data_dir`` may be a plain str after YAML round-tripping. Normalize
    # to a Path here so every downstream filesystem call (.mkdir/.is_file/rglob)
    # receives a proper Path object.
    output_dir = Path(args.output or config.detection_dataset.detection_data_dir)
    api_key = args.api_key or require_env("ROBOFLOW_API_KEY")

    logger.info("=" * 70)
    logger.info("FreshSense AI - Download Detection Dataset")
    logger.info("=" * 70)
    logger.info("Workspace:      %s", workspace)
    logger.info("Project:        %s", project)
    logger.info("Version:        %s", version)
    logger.info("Output:         %s", output_dir)
    logger.info("Format:         %s", args.format)
    logger.info("Overwrite:      %s", args.overwrite)
    logger.info("Normalize-only: %s", args.normalize_only)
    logger.info("=" * 70)

    if args.normalize_only:
        data_yaml = find_data_yaml(output_dir)
        normalize_data_yaml(data_yaml)
        label_audit = normalize_labels(Path(data_yaml).parent)
        _write_manifest(
            args.manifest, workspace, project, version, args.format,
            output_dir, data_yaml, label_audit,
        )
        logger.info("Normalization complete. data.yaml: %s", data_yaml)
        logger.info("Manifest saved to: %s", args.manifest)
        sys.exit(0)

    data_yaml, label_audit = download_dataset(
        workspace=workspace,
        project=project,
        version=version,
        output_dir=output_dir,
        api_key=api_key,
        format=args.format,
        overwrite=args.overwrite,
    )
    _write_manifest(
        args.manifest, workspace, project, version, args.format,
        output_dir, data_yaml, label_audit,
    )
    logger.info("=" * 70)
    logger.info("Download complete! Dataset config: %s", data_yaml)
    logger.info("Manifest saved to: %s", args.manifest)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

