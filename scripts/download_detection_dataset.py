#!/usr/bin/env python3
"""Download the FreshSense fruit detection dataset from Roboflow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from configs.config import Config
from src.utils.environment import load_environment, require_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_dataset(
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
    api_key: str,
    format: str = "yolov8",
) -> None:
    """Download dataset from Roboflow.

    Args:
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.
        output_dir: Local directory to save the dataset.
        api_key: Roboflow API key.
        format: Export format (default: yolov8 for YOLOv8/YOLOv11).
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package required. Install: pip install roboflow")
        sys.exit(1)

    logger.info("Connecting to Roboflow workspace: %s", workspace)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    version = project.version(version)

    logger.info("Downloading dataset (format=%s) to %s", format, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = version.download(format, location=str(output_dir))

    logger.info("Download complete! Dataset at: %s", output_dir)
    logger.info("Classes: %s", dataset.classes if hasattr(dataset, "classes") else "N/A")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FreshSense detection dataset from Roboflow"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Roboflow workspace name",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Roboflow project name",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Dataset version number",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="yolov8",
        help="Export format (default: yolov8)",
    )

    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")

    workspace = args.workspace or config.detection_dataset.roboflow_workspace
    project = args.project or config.detection_dataset.roboflow_project
    version = args.version or config.detection_dataset.roboflow_version
    output_dir = args.output or config.detection_dataset.detection_data_dir
    api_key = args.api_key or require_env("ROBOFLOW_API_KEY")

    logger.info("=" * 70)
    logger.info("FreshSense AI - Download Detection Dataset")
    logger.info("=" * 70)
    logger.info("Workspace:     %s", workspace)
    logger.info("Project:       %s", project)
    logger.info("Version:       %s", version)
    logger.info("Output:        %s", output_dir)
    logger.info("Format:        %s", args.format)
    logger.info("=" * 70)

    download_dataset(
        workspace=workspace,
        project=project,
        version=version,
        output_dir=output_dir,
        api_key=api_key,
        format=args.format,
    )

    logger.info("=" * 70)
    logger.info("Download complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

# placeholder