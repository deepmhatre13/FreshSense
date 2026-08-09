"""Diagnostic instrumentation for FreshSense inference."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameDiagnostics:
    frame_id: int
    timestamp: float
    crop_iou: float | None = None
    crop_size: tuple[int, int] | None = None
    crop_position: tuple[float, float, float, float] | None = None
    blur_score: float | None = None
    brightness: float | None = None
    contrast: float | None = None
    detector_confidence: float | None = None
    classifier_confidence: float | None = None
    fused_confidence: float | None = None
    predicted_class: str | None = None
    true_class: str | None = None
    is_correct: bool | None = None
    inference_time_ms: float | None = None
    track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DiagnosticCollector:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[FrameDiagnostics] = []

    def record(self, diag: FrameDiagnostics) -> None:
        self.frames.append(diag)

    def save(self) -> Path:
        import json

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj

        payload = [
            {k: _serialize(v) for k, v in diag.__dict__.items() if v is not None}
            for diag in self.frames
        ]
        path = self.output_dir / "diagnostics.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def summary(self) -> dict[str, Any]:
        if not self.frames:
            return {}
        confidences = np.array(
            [d.classifier_confidence for d in self.frames if d.classifier_confidence is not None]
        )
        ious = np.array([d.crop_iou for d in self.frames if d.crop_iou is not None])
        blur = np.array([d.blur_score for d in self.frames if d.blur_score is not None])
        return {
            "num_frames": len(self.frames),
            "mean_confidence": float(confidences.mean()) if confidences.size else None,
            "std_confidence": float(confidences.std()) if confidences.size else None,
            "mean_crop_iou": float(ious.mean()) if ious.size else None,
            "mean_blur": float(blur.mean()) if blur.size else None,
        }
