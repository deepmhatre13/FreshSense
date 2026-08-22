#!/usr/bin/env python3
"""Generate a reproducible Roboflow dataset version for FreshSense AI.

Creates a new generated version of the configured Roboflow object-detection
project. To keep the version reproducible and faithful to the source data,
generation uses NO augmentation and only ``auto-orient`` preprocessing
(normalizes EXIF orientation without resampling pixels). Train/valid/test
splits are the ones already assigned inside the Roboflow project.

The script is idempotent: if the requested version already exists it reports
it without triggering a new generation (override with ``--force``).

Security:
  - API key is read from ``ROBOFLOW_API_KEY``, never printed.
  - The manifest written to disk contains non-secret fields only.

Usage:
    python scripts/generate_dataset_version.py
    python scripts/generate_dataset_version.py --version 1 --manifest reports/roboflow_version.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Reproducible "null" generation settings: no augmentation, only EXIF
# orientation normalization. Pixel data and annotation coordinates are kept.
GENERATION_SETTINGS: Dict[str, Dict[str, Any]] = {
    "preprocessing": {"auto-orient": True},
    "augmentation": {},
}

DEFAULT_TIMEOUT_SECONDS = 900
POLL_INTERVAL_SECONDS = 10


def _list_versions(project: Any) -> List[Dict[str, Any]]:
    """List existing versions as secret-free dicts."""
    try:
        raw = project.get_version_information()
    except Exception as exc:
        logger.warning("Failed to list versions: %s", exc)
        return []
    versions: List[Dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        version_id = item.get("id")
        versions.append(
            {
                "number": (
                    str(version_id).rsplit("/", 1)[-1]
                    if isinstance(version_id, str)
                    else version_id
                ),
                "images": item.get("images"),
                "splits": item.get("splits"),
                "exports": item.get("exports", []),
                "preprocessing": item.get("preprocessing"),
                "augmentation": item.get("augmentation"),
                "created": item.get("created"),
            }
        )
    return versions


def wait_for_version(project: Any, version_number: int, timeout: int) -> Dict[str, Any]:
    """Poll until *version_number* is finished generating (has image counts)."""
    deadline = time.time() + timeout
    last_seen: List[Dict[str, Any]] = []
    while time.time() < deadline:
        versions = _list_versions(project)
        for version in versions:
            if str(version["number"]) == str(version_number) and version.get("images"):
                return version
        if versions:
            last_seen = versions
        seconds_left = max(int(deadline - time.time()), 0)
        if last_seen:
            numbers = sorted(str(v["number"]) for v in last_seen)
            logger.info(
                "Waiting for version %s to finish generating (%s s left, existing: %s)...",
                version_number,
                seconds_left,
                numbers,
            )
        else:
            logger.info(
                "Waiting for version %s to appear (%s s left)...",
                version_number,
                seconds_left,
            )
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Version {version_number} did not finish generating within {timeout}s. "
        "Check the Roboflow web console for generation status."
    )
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible Roboflow dataset version"
    )
    parser.add_argument("--workspace", type=str, default=None, help="Roboflow workspace")
    parser.add_argument("--project", type=str, default=None, help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=None, help="Version number to generate")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Generation wait timeout (s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/roboflow_version.json"),
        help="Path to save the version manifest JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Trigger regeneration even if the version already exists",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Only request generation, do not wait for it to finish (re-run with a poller)",
    )
    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")

    workspace = args.workspace or config.detection_dataset.roboflow_workspace
    project_slug = args.project or config.detection_dataset.roboflow_project
    version_number = args.version or config.detection_dataset.roboflow_version
    api_key = require_env("ROBOFLOW_API_KEY")

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package required. Install: pip install roboflow")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("FreshSense AI - Generate Roboflow Dataset Version")
    logger.info("=" * 70)
    logger.info("Workspace:      %s", workspace)
    logger.info("Project:        %s", project_slug)
    logger.info("Requested v:    %s", version_number)
    logger.info("Generation:     preprocessing=%s augmentation={}",
                GENERATION_SETTINGS["preprocessing"])
    logger.info("=" * 70)

    rf = Roboflow(api_key=api_key)
    try:
        project = rf.workspace(workspace).project(project_slug)
    except Exception as exc:
        logger.error("Failed to reach Roboflow project: %s", exc)
        sys.exit(1)

    existing = _list_versions(project)
    numbers = [str(v["number"]) for v in existing]
    logger.info("Existing versions on project: %s", numbers or "none")

    triggered = False
    if str(version_number) in numbers and not args.force:
        logger.info("Version %s already exists - nothing to generate.", version_number)
    else:
        logger.info("Requesting Roboflow to generate version %s ...", version_number)
        try:
            project.generate_version(settings=GENERATION_SETTINGS)
            triggered = True
        except Exception as exc:
            logger.error("Version generation request failed: %s", exc)
            sys.exit(1)

    if args.no_wait:
        finished = {
            "number": str(version_number),
            "status": "requested" if triggered else "already_existed",
            "images": None,
            "splits": None,
        }
        logger.info(
            "Generation request sent (no-wait). Re-run without --no-wait or "
            "re-run the audit to confirm the version becomes ready."
        )
    else:
        try:
            finished = wait_for_version(project, version_number, timeout=args.timeout)
        except TimeoutError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    manifest = {
        "tool": "FreshSenseAI Roboflow Dataset Version Generator",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "requested": {
            "workspace": workspace,
            "project": project_slug,
            "version": version_number,
        },
        "settings": GENERATION_SETTINGS,
        "triggered_generation": triggered,
        "status": finished.get("status", "ready"),
        "version": finished,
        "notes": [
            "Generation settings are explicit and reproducible (no augmentation, auto-orient only).",
            "This manifest contains no secrets.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    logger.info("=" * 70)
    logger.info(
        "Version %s is ready: %s images | splits=%s",
        finished.get("number"),
        finished.get("images"),
        finished.get("splits"),
    )
    logger.info("Manifest saved to: %s", args.manifest)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()