#!/usr/bin/env python3
"""Audit the Roboflow detection dataset used by FreshSense AI.

Connects to the configured Roboflow workspace/project through the official
Roboflow Python SDK and writes a machine-readable audit report containing
ONLY non-secret metadata:

  - project identity, type, visibility
  - image count, unannotated count, train/valid/test split counts
  - class list and number of classes
  - generated version list (id, images, splits, exports)

Security contract:
  * The API key is read from ``ROBOFLOW_API_KEY`` (via .env) and is NEVER
    printed, logged, or persisted in the report.
  * The script never dumps ``project.__dict__`` (it carries the API key); only
    the public attributes listed above are read.

Usage:
    python scripts/audit_roboflow_dataset.py
    python scripts/audit_roboflow_dataset.py --workspace deepam-mhatre \\
        --project fruits-test-ajvf8-duncc --output reports/roboflow_project_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

# Version payload fields considered safe to persist (never the API key).
_SAFE_VERSION_FIELDS = ("images", "splits", "created", "exports")


def _render_timestamp(value: Any) -> Any:
    """Render a datetime/timestamp value to an ISO string without crashing."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def collect_project_summary(project: Any) -> Dict[str, Any]:
    """Read only public, non-secret project attributes."""
    summary: Dict[str, Any] = {}
    for attr in (
        "id",
        "name",
        "type",
        "public",
        "images",
        "unannotated",
        "multilabel",
        "model_format",
    ):
        try:
            summary[attr] = getattr(project, attr)
        except Exception:
            summary[attr] = None

    for attr in ("created", "updated"):
        try:
            summary[attr] = _render_timestamp(getattr(project, attr))
        except Exception:
            summary[attr] = None

    summary["workspace"] = None
    summary["project_slug"] = None
    try:
        if summary.get("id"):
            summary["workspace"], summary["project_slug"] = str(summary["id"]).rsplit("/", 1)
    except Exception:
        pass

    try:
        splits = getattr(project, "splits", None) or {}
        summary["splits"] = {
            str(key): (int(value) if value is not None else None)
            for key, value in splits.items()
        }
    except Exception:
        summary["splits"] = {}

    try:
        raw_classes = getattr(project, "classes", None)
        if isinstance(raw_classes, dict):
            classes = list(raw_classes.values())
        elif isinstance(raw_classes, list):
            classes = list(raw_classes)
        elif isinstance(raw_classes, str) and raw_classes:
            classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
        else:
            classes = []
    except Exception:
        classes = []
    summary["classes"] = classes
    summary["num_classes"] = len(classes)
    return summary


def collect_versions(project: Any) -> List[Dict[str, Any]]:
    """Return a secret-free list of existing dataset versions."""
    versions: List[Dict[str, Any]] = []
    try:
        raw = project.get_version_information()
    except Exception as exc:
        logger.warning("Could not list version information: %s", exc)
        return versions

    for item in raw or []:
        if not isinstance(item, dict):
            continue
        version_id = item.get("id")
        entry: Dict[str, Any] = {
            "id": version_id,
            "number": (
                str(version_id).rsplit("/", 1)[-1]
                if isinstance(version_id, str)
                else version_id
            ),
        }
        for field in _SAFE_VERSION_FIELDS:
            entry[field] = item.get(field)
        versions.append(entry)
    return versions
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the FreshSense Roboflow detection dataset"
    )
    parser.add_argument("--workspace", type=str, default=None, help="Roboflow workspace")
    parser.add_argument("--project", type=str, default=None, help="Roboflow project slug")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/roboflow_project_audit.json"),
        help="Path to save audit report JSON",
    )
    args = parser.parse_args()

    load_environment()
    config = Config.from_yaml("configs/settings.yaml")

    workspace = args.workspace or config.detection_dataset.roboflow_workspace
    project_slug = args.project or config.detection_dataset.roboflow_project
    api_key = require_env("ROBOFLOW_API_KEY")

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package required. Install: pip install roboflow")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("FreshSense AI - Roboflow Dataset Audit")
    logger.info("=" * 70)
    logger.info("Workspace:      %s", workspace)
    logger.info("Project slug:   %s", project_slug)
    logger.info("=" * 70)

    rf = Roboflow(api_key=api_key)
    try:
        project = rf.workspace(workspace).project(project_slug)
    except Exception as exc:
        logger.error("Failed to reach Roboflow project: %s", exc)
        sys.exit(1)

    summary = collect_project_summary(project)
    versions = collect_versions(project)

    report = {
        "tool": "FreshSenseAI Roboflow Dataset Audit",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "requested": {"workspace": workspace, "project": project_slug},
        "project": summary,
        "versions": versions,
        "num_versions": len(versions),
        "notes": [
            "This report contains no secrets; the Roboflow API key is never printed or persisted.",
            "The API key used for this audit was previously exposed in terminal output and "
            "should be rotated/revoked if it is still active.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    logger.info("Project:        %s (%s)", summary.get("name"), summary.get("id"))
    logger.info("Type:           %s | Public: %s", summary.get("type"), summary.get("public"))
    logger.info("Images:         %s | Unannotated: %s", summary.get("images"), summary.get("unannotated"))
    logger.info("Splits:         %s", summary.get("splits"))
    logger.info("Classes (%d):   %s", summary.get("num_classes", 0), summary.get("classes"))
    logger.info("Versions:       %d", len(versions))
    for version in versions:
        logger.info(
            "  - v%s: images=%s splits=%s",
            version.get("number"),
            version.get("images"),
            version.get("splits"),
        )
    logger.info("Report saved to: %s", args.output)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()