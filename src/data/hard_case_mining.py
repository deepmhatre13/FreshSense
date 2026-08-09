"""Hard-case mining for FreshSense Phase 4."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["HardCaseConfig", "HardCase", "HardCaseMiner"]


@dataclass(frozen=True)
class HardCaseConfig:
    """Configuration for hard-case identification."""
    low_confidence_threshold: float = 0.60
    high_confidence_error_threshold: float = 0.80
    quality_blur_threshold: float = 80.0
    quality_brightness_min: int = 30
    quality_brightness_max: int = 240
    instability_window: int = 10
    instability_threshold: float = 0.4
    output_dir: Path = Path("reports")

@dataclass
class HardCase:
    """A identified hard case with diagnostic information."""
    image_path: str
    predicted_class: str
    true_class: Optional[str] = None
    confidence: float = 0.0
    quality_metrics: Optional[Dict[str, Any]] = None
    detector_confidence: Optional[float] = None
    classifier_confidence: Optional[float] = None
    fused_confidence: Optional[float] = None
    tracking_id: Optional[int] = None
    stability_score: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "predicted_class": self.predicted_class,
            "true_class": self.true_class,
            "confidence": self.confidence,
            "detector_confidence": self.detector_confidence,
            "classifier_confidence": self.classifier_confidence,
            "fused_confidence": self.fused_confidence,
            "tracking_id": self.tracking_id,
            "stability_score": self.stability_score,
            "reasons": "; ".join(self.reasons),
            **self.metadata,
        }


class HardCaseMiner:
    """Mines hard cases from inference diagnostics."""

    def __init__(self, config: Optional[HardCaseConfig] = None) -> None:
        self.config = config or HardCaseConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)


    def analyze_diagnostics(self, diagnostics: List[Dict[str, Any]]) -> List[HardCase]:
        """Analyze diagnostic frames to identify hard cases."""
        hard_cases: List[HardCase] = []

        for diag in diagnostics:
            reasons = self._identify_reasons(diag)
            if reasons:
                hard_case = HardCase(
                    image_path=diag.get("image_path", ""),
                    predicted_class=diag.get("predicted_class", ""),
                    true_class=diag.get("true_class"),
                    confidence=diag.get("confidence", 0.0),
                    quality_metrics=diag.get("quality_metrics"),
                    detector_confidence=diag.get("detector_confidence"),
                    classifier_confidence=diag.get("classifier_confidence"),
                    fused_confidence=diag.get("fused_confidence"),
                    tracking_id=diag.get("tracking_id"),
                    stability_score=diag.get("stability_score"),
                    reasons=reasons,
                    metadata={k: v for k, v in diag.items() if k not in [
                        "image_path", "predicted_class", "true_class", "confidence",
                        "quality_metrics", "detector_confidence", "classifier_confidence",
                        "fused_confidence", "tracking_id", "stability_score"
                    ]},
                )
                hard_cases.append(hard_case)

        logger.info("Identified %d hard cases from %d frames", len(hard_cases), len(diagnostics))
        return hard_cases

    def _identify_reasons(self, diag: Dict[str, Any]) -> List[str]:
        """Identify why a frame is a hard case."""
        reasons: List[str] = []
        confidence = diag.get("confidence", 0.0)
        true_class = diag.get("true_class")
        predicted_class = diag.get("predicted_class")

        if confidence < self.config.low_confidence_threshold:
            reasons.append(f"low_confidence_{confidence:.3f}")

        if true_class is not None and predicted_class != true_class:
            if confidence > self.config.high_confidence_error_threshold:
                reasons.append(f"high_confidence_error_{confidence:.3f}")
            else:
                reasons.append("misclassified")

        quality = diag.get("quality_metrics", {})
        if quality:
            if quality.get("blur_score", 100) < self.config.quality_blur_threshold:
                reasons.append(f"blurry_{quality.get('blur_score', 0):.1f}")
            if quality.get("brightness", 128) < self.config.quality_brightness_min:
                reasons.append("too_dark")
            if quality.get("brightness", 128) > self.config.quality_brightness_max:
                reasons.append("too_bright")

        det_conf = diag.get("detector_confidence")
        cls_conf = diag.get("classifier_confidence")
        if det_conf is not None and cls_conf is not None:
            if abs(det_conf - cls_conf) > 0.3:
                reasons.append("detector_classifier_disagreement")

        stability = diag.get("stability_score")
        if stability is not None and stability < 0.5:
            reasons.append("temporally_unstable")

        if 0.4 < confidence < 0.6:
            reasons.append("near_decision_boundary")

        return reasons

    def analyze_directory(self, diagnostics_dir: Path, pattern: str = "diagnostics.json") -> List[HardCase]:
        """Analyze all diagnostic files in a directory."""
        all_diagnostics: List[Dict[str, Any]] = []

        for json_file in diagnostics_dir.rglob(pattern):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    all_diagnostics.extend(data)
                elif isinstance(data, dict) and "frames" in data:
                    all_diagnostics.extend(data["frames"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load %s: %s", json_file, exc)

        return self.analyze_diagnostics(all_diagnostics)

    def save_reports(self, hard_cases: List[HardCase], prefix: str = "hard_cases") -> Dict[str, Path]:
        """Save hard-case reports to disk."""
        saved: Dict[str, Path] = {}

        if not hard_cases:
            logger.warning("No hard cases to save.")
            return saved

        csv_path = self.config.output_dir / f"{prefix}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=hard_cases[0].to_dict().keys())
            writer.writeheader()
            for hc in hard_cases:
                writer.writerow(hc.to_dict())
        saved["csv"] = csv_path

        json_path = self.config.output_dir / f"{prefix}.json"
        json_path.write_text(
            json.dumps([hc.to_dict() for hc in hard_cases], indent=2),
            encoding="utf-8",
        )
        saved["json"] = json_path

        summary_path = self.config.output_dir / f"{prefix}_summary.md"
        self._write_summary(hard_cases, summary_path)
        saved["summary"] = summary_path

        logger.info("Saved hard-case reports to %s", self.config.output_dir)
        return saved

    def _write_summary(self, hard_cases: List[HardCase], path: Path) -> None:
        """Write a Markdown summary of hard cases."""
        total = len(hard_cases)
        reason_counts: Dict[str, int] = {}
        for hc in hard_cases:
            for reason in hc.reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        lines = [
            "# Hard-Case Mining Report",
            "",
            f"**Total hard cases**: {total}",
            "",
            "## Reason Distribution",
            "",
        ]
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{reason}**: {count} ({100.0*count/max(total,1):.1f}%)")

        lines.extend([
            "",
            "## Confidence Distribution",
            "",
            f"- Mean: {np.mean([hc.confidence for hc in hard_cases]):.3f}",
            f"- Std: {np.std([hc.confidence for hc in hard_cases]):.3f}",
            "",
        ])

        if any(hc.true_class is not None for hc in hard_cases):
            errors = [hc for hc in hard_cases if hc.true_class is not None and hc.predicted_class != hc.true_class]
            lines.extend([
                "## Error Analysis",
                "",
                f"- Total with ground truth: {sum(1 for hc in hard_cases if hc.true_class is not None)}",
                f"- Errors: {len(errors)}",
                "",
            ])

        path.write_text("\n".join(lines), encoding="utf-8")
