"""Professional overlay system for FreshSense Phase 2 real-time inference.

This module provides a modern, professional overlay system for displaying
inference results on video frames:

- Rounded bounding boxes with gradient fills
- Color-coded by freshness class (green=fresh, yellow=unknown, red=rotten)
- Transparent overlays for minimal visual obstruction
- FPS, latency, and device information display
- Confidence threshold filtering
- Multi-line text with background padding
- Professional typography and spacing

The Overlay class is designed for high-performance rendering with minimal
latency impact on the inference pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Overlay", "OverlayConfig", "ColorScheme"]


@dataclass(frozen=True)
class ColorScheme:
    """Color definitions for overlay elements.

    Attributes:
        fresh: Color for fresh produce (BGR format).
        stale: Color for stale produce (BGR format).
        rotten: Color for rotten produce (BGR format).
        unknown: Color for unknown/unclassified produce (BGR format).
        background: Background color for text boxes (BGR format).
        text: Text color (BGR format).
        fps_text: FPS display text color (BGR format).
        accent: Accent color for highlights (BGR format).
    """

    fresh: Tuple[int, int, int] = (0, 255, 0)  # Green
    stale: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    rotten: Tuple[int, int, int] = (0, 0, 255)  # Red
    unknown: Tuple[int, int, int] = (128, 128, 128)  # Gray
    background: Tuple[int, int, int] = (0, 0, 0)  # Black
    text: Tuple[int, int, int] = (255, 255, 255)  # White
    fps_text: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    accent: Tuple[int, int, int] = (255, 128, 0)  # Orange


@dataclass(frozen=True)
class OverlayConfig:
    """Configuration for overlay rendering.

    Attributes:
        font_scale: Font scale for text rendering.
        font_thickness: Thickness of text strokes.
        box_thickness: Thickness of bounding box strokes.
        padding: Padding around text in pixels.
        corner_radius: Radius of rounded corners in pixels.
        alpha: Transparency of overlay elements (0.0-1.0).
        show_confidence: If True, display confidence percentage.
        show_latency: If True, display inference latency.
        show_device: If True, display device information.
        show_fps: If True, display FPS counter.
        confidence_threshold: Minimum confidence to display prediction.
    """

    font_scale: float = 0.6
    font_thickness: int = 2
    box_thickness: int = 3
    padding: int = 10
    corner_radius: int = 8
    alpha: float = 0.7
    show_confidence: bool = True
    show_latency: bool = True
    show_device: bool = True
    show_fps: bool = True
    confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.font_scale <= 0:
            raise ValueError("font_scale must be positive.")
        if self.font_thickness <= 0:
            raise ValueError("font_thickness must be positive.")
        if self.box_thickness <= 0:
            raise ValueError("box_thickness must be positive.")
        if self.padding < 0:
            raise ValueError("padding must be non-negative.")
        if self.corner_radius < 0:
            raise ValueError("corner_radius must be non-negative.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0].")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0].")


class Overlay:
    """Professional overlay rendering for inference results.

    This class provides methods to draw:
    - Prediction boxes with class labels and confidence
    - FPS and performance metrics
    - Device information
    - Custom text overlays

    All rendering uses OpenCV with optimized operations for minimal latency.

    Args:
        config: OverlayConfig instance with rendering settings.
        colors: ColorScheme instance with color definitions.
    """

    def __init__(self, config: OverlayConfig, colors: Optional[ColorScheme] = None) -> None:
        self.config = config
        self.colors = colors or ColorScheme()

    def draw_prediction(
        self,
        frame: np.ndarray,
        label: str,
        confidence: float,
        color: Tuple[int, int, int],
        box: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Draw a prediction overlay on the frame.

        Args:
            frame: BGR image as numpy array.
            label: Class label to display.
            confidence: Prediction confidence (0.0-1.0).
            color: BGR color tuple for the box and text.
            box: Optional bounding box (x1, y1, x2, y2). If None, draws
                a full-width overlay at the top of the frame.

        Returns:
            Frame with overlay drawn.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Build label text
        if self.config.show_confidence:
            text = f"{label}: {confidence * 100:.1f}%"
        else:
            text = label

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.font_thickness
        )

        if box is not None:
            x1, y1, x2, y2 = box
            box_h = y2 - y1
            box_w = x2 - x1

            # Draw rounded rectangle background
            bg_y1 = max(0, y1 - text_h - self.config.padding * 2)
            bg_y2 = min(h, y1 + self.config.padding)
            bg_x1 = max(0, x1 - self.config.padding)
            bg_x2 = min(w, x2 + self.config.padding)

            self._draw_rounded_rect(
                overlay, (bg_x1, bg_y1, bg_x2, bg_y2), self.colors.background, self.config.corner_radius
            )

            # Draw bounding box
            self._draw_rounded_rect(
                overlay, (x1, y1, x2, y2), color, self.config.corner_radius, thickness=self.config.box_thickness
            )

            # Draw text
            text_x = x1
            text_y = y1 - self.config.padding
        else:
            # Full-width overlay at top
            bg_y1 = 0
            bg_y2 = text_h + self.config.padding * 3
            bg_x1 = 0
            bg_x2 = w

            self._draw_rounded_rect(
                overlay, (bg_x1, bg_y1, bg_x2, bg_y2), self.colors.background, self.config.corner_radius
            )

            text_x = self.config.padding
            text_y = text_h + self.config.padding

        # Draw text with outline for better visibility
        cv2.putText(
            overlay, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
            self.colors.text, self.config.font_thickness, cv2.LINE_AA
        )

        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, self.config.alpha, frame, 1.0 - self.config.alpha, 0)

        return frame

    def draw_performance_stats(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        device: str,
    ) -> np.ndarray:
        """Draw performance statistics overlay.

        Args:
            frame: BGR image as numpy array.
            fps: Current frames per second.
            latency_ms: Inference latency in milliseconds.
            device: Device name (CPU, CUDA, MPS, etc.).

        Returns:
            Frame with performance stats drawn.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Build stats text
        lines = []
        if self.config.show_fps:
            lines.append(f"FPS: {fps:.1f}")
        if self.config.show_latency:
            lines.append(f"Latency: {latency_ms:.1f}ms")
        if self.config.show_device:
            lines.append(f"Device: {device}")

        # Calculate text positions (bottom-left corner)
        y_offset = h - self.config.padding
        line_height = int(self.config.font_scale * 30)

        # Draw background
        max_text_w = 0
        for line in lines:
            (text_w, _), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.font_thickness
            )
            max_text_w = max(max_text_w, text_w)

        bg_x1 = 0
        bg_y1 = h - (len(lines) * line_height) - self.config.padding * 2
        bg_x2 = max_text_w + self.config.padding * 2
        bg_y2 = h

        self._draw_rounded_rect(
            overlay, (bg_x1, bg_y1, bg_x2, bg_y2), self.colors.background, self.config.corner_radius
        )

        # Draw text
        for i, line in enumerate(lines):
            text_y = y_offset - (len(lines) - i - 1) * line_height
            cv2.putText(
                overlay, line, (self.config.padding, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                self.colors.fps_text, self.config.font_thickness, cv2.LINE_AA
            )

        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, self.config.alpha, frame, 1.0 - self.config.alpha, 0)

        return frame

    def draw_confidence_bar(
        self,
        frame: np.ndarray,
        confidence: float,
        color: Tuple[int, int, int],
        position: str = "top-right",
    ) -> np.ndarray:
        """Draw a horizontal confidence bar.

        Args:
            frame: BGR image as numpy array.
            confidence: Confidence value (0.0-1.0).
            color: BGR color tuple for the bar.
            position: Position of the bar ("top-right", "top-left", "bottom-right", "bottom-left").

        Returns:
            Frame with confidence bar drawn.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        bar_width = w // 4
        bar_height = 10
        margin = 20

        if position == "top-right":
            x1 = w - bar_width - margin
            y1 = margin
        elif position == "top-left":
            x1 = margin
            y1 = margin
        elif position == "bottom-right":
            x1 = w - bar_width - margin
            y1 = h - bar_height - margin
        else:  # bottom-left
            x1 = margin
            y1 = h - bar_height - margin

        x2 = x1 + bar_width
        y2 = y1 + bar_height

        # Draw background bar
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.colors.background, -1)

        # Draw filled portion
        fill_width = int(bar_width * confidence)
        cv2.rectangle(overlay, (x1, y1), (x1 + fill_width, y2), color, -1)

        # Draw border
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.colors.text, 1)

        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, self.config.alpha, frame, 1.0 - self.config.alpha, 0)

        return frame

    def _draw_filled_rounded_rectangle(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        alpha: float = 0.8,
        radius: int = 10,
    ) -> np.ndarray:
        """Draw a filled rounded rectangle with alpha blending.

        Args:
            frame: Image to draw on.
            box: Bounding box (x1, y1, x2, y2).
            color: BGR color tuple.
            alpha: Transparency level (0.0 = fully transparent, 1.0 = opaque).
            radius: Corner radius in pixels.

        Returns:
            Frame with rounded rectangle drawn.
        """
        x1, y1, x2, y2 = box

        # Create overlay for alpha blending
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
        cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
        cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
        cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)

        # Blend with original frame
        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    def _draw_rounded_rect(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        radius: int,
        thickness: int = -1,
    ) -> None:
        """Draw a rounded rectangle directly on the frame.

        Args:
            frame: Image to draw on (modified in place).
            box: Bounding box (x1, y1, x2, y2).
            color: BGR color tuple.
            radius: Corner radius in pixels.
            thickness: Line thickness (-1 for filled).
        """
        x1, y1, x2, y2 = box

        if thickness == -1:
            # Filled rectangle with rounded corners
            # Draw main rectangle
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # Draw four corners
            cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
            cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
            cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
            cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
        else:
            # Outlined rectangle with rounded corners
            # Draw lines
            cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

            # Draw four corners
            cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def get_class_color(self, label: str) -> Tuple[int, int, int]:
        """Get color for a given class label.

        Args:
            label: Class label string (e.g. "freshapples", "rottenbanana",
                "stale", or "fresh_apple").

        Returns:
            BGR color tuple.
        """
        label_lower = (label or "").lower().strip()
        if label_lower.startswith("fresh"):
            return self.colors.fresh
        elif label_lower.startswith("rotten"):
            return self.colors.rotten
        elif label_lower.startswith("stale"):
            return self.colors.stale
        else:
            return self.colors.unknown



    def draw_phase3_overlay(
        self,
        frame: np.ndarray,
        stabilized,
        tracked,
        raw_result,
        quality_report,
        fps_monitor,
        frame_number: int,
        predictor,
    ) -> np.ndarray:
        """Draw Phase 3 enhanced overlay with all information."""
        overlay_frame = frame.copy()

        # Determine prediction display
        if stabilized.is_uncertain:
            pred_label = "Uncertain"
            pred_color = self.colors.stale
            pred_conf = stabilized.confidence
        else:
            pred_label = stabilized.label
            pred_color = self.get_class_color(stabilized.label)
            pred_conf = stabilized.confidence

        # Draw prediction box (top-left)
        box = (20, 20, 320, 280)
        overlay_frame = self.draw_prediction(
            overlay_frame, pred_label, pred_conf, pred_color, box=box
        )

        # Draw quality indicator (top-right)
        quality_box = (overlay_frame.shape[1] - 220, 20, overlay_frame.shape[1] - 20, 140)
        overlay_frame = self._draw_quality_indicator(overlay_frame, quality_report, quality_box)

        # Draw performance dashboard (bottom-left)
        perf_box = (20, overlay_frame.shape[0] - 180, 280, overlay_frame.shape[0] - 20)
        overlay_frame = self._draw_performance_dashboard(
            overlay_frame, fps_monitor, raw_result, frame_number, predictor, perf_box
        )

        # Draw tracking stability (bottom-right)
        track_box = (
            overlay_frame.shape[1] - 220,
            overlay_frame.shape[0] - 100,
            overlay_frame.shape[1] - 20,
            overlay_frame.shape[0] - 20,
        )
        overlay_frame = self._draw_tracking_stability(overlay_frame, stabilized, tracked, track_box)

        # Draw warnings if any
        if quality_report.warnings:
            warning_box = (20, overlay_frame.shape[0] - 220, overlay_frame.shape[1] - 20, overlay_frame.shape[0] - 180)
            overlay_frame = self._draw_warnings(overlay_frame, quality_report.warnings, warning_box)

        return overlay_frame

    def draw_quality_warning(
        self, frame: np.ndarray, warning: str, quality_report
    ) -> np.ndarray:
        """Draw quality warning overlay."""
        overlay_frame = frame.copy()
        h, w = overlay_frame.shape[:2]

        box = (20, 20, w - 20, 100)
        overlay_frame = self._draw_filled_rounded_rectangle(
            overlay_frame, box, self.colors.rotten, alpha=0.8
        )

        text = warning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - text_w) // 2
        y = 20 + (80 - text_h) // 2 + text_h // 2

        cv2.putText(
            overlay_frame, text, (x, y), font, font_scale,
            self.colors.text, thickness, cv2.LINE_AA
        )

        return overlay_frame

    def _draw_quality_indicator(self, frame: np.ndarray, report, box: tuple) -> np.ndarray:
        """Draw quality indicator box."""
        overlay_frame = self._draw_filled_rounded_rectangle(frame, box, self.colors.background, alpha=0.7)

        cv2.putText(
            overlay_frame, "Image Quality", (box[0] + 10, box[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.text, 1, cv2.LINE_AA
        )

        stars = int(report.quality_score * 5)
        star_text = "*" * stars + "-" * (5 - stars)
        cv2.putText(
            overlay_frame, star_text, (box[0] + 10, box[1] + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.accent, 1, cv2.LINE_AA
        )

        if report.quality_score >= 0.8:
            quality_label = "Good"
            quality_color = self.colors.fresh
        elif report.quality_score >= 0.5:
            quality_label = "Average"
            quality_color = self.colors.stale
        else:
            quality_label = "Poor"
            quality_color = self.colors.rotten

        cv2.putText(
            overlay_frame, quality_label, (box[0] + 10, box[1] + 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1, cv2.LINE_AA
        )

        return overlay_frame

    def _draw_performance_dashboard(
        self, frame: np.ndarray, fps_monitor, raw_result, frame_number: int, predictor, box: tuple
    ) -> np.ndarray:
        """Draw performance dashboard box."""
        overlay_frame = self._draw_filled_rounded_rectangle(frame, box, self.colors.background, alpha=0.7)

        stats = fps_monitor.get_stats()

        lines = [
            f"Frame: {frame_number}",
            f"FPS: {stats.current_fps:.1f}",
            f"Avg FPS: {stats.average_fps:.1f}",
            f"Inference: {raw_result.latency_ms:.1f}ms",
            f"Model: {predictor.model_version}",
        ]

        y_offset = box[1] + 25
        for line in lines:
            cv2.putText(
                overlay_frame, line, (box[0] + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors.text, 1, cv2.LINE_AA
            )
            y_offset += 22

        return overlay_frame

    def _draw_tracking_stability(
        self, frame: np.ndarray, stabilized, tracked, box: tuple
    ) -> np.ndarray:
        """Draw tracking stability indicator."""
        overlay_frame = self._draw_filled_rounded_rectangle(frame, box, self.colors.background, alpha=0.7)

        cv2.putText(
            overlay_frame, "Tracking", (box[0] + 10, box[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.text, 1, cv2.LINE_AA
        )

        stability_pct = stabilized.confidence * 100
        stability_text = f"Stability: {stability_pct:.0f}%"
        cv2.putText(
            overlay_frame, stability_text, (box[0] + 10, box[1] + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors.fps_text, 1, cv2.LINE_AA
        )

        lock_text = "Locked" if stabilized.is_locked else "Unlocked"
        lock_color = self.colors.fresh if not stabilized.is_locked else self.colors.stale
        cv2.putText(
            overlay_frame, lock_text, (box[0] + 10, box[1] + 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, lock_color, 1, cv2.LINE_AA
        )

        return overlay_frame

    def _draw_warnings(self, frame: np.ndarray, warnings: list, box: tuple) -> np.ndarray:
        """Draw warnings box."""
        overlay_frame = self._draw_filled_rounded_rectangle(frame, box, self.colors.background, alpha=0.7)

        y_offset = box[1] + 25
        for warning in warnings[:2]:
            cv2.putText(
                overlay_frame, warning, (box[0] + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors.rotten, 1, cv2.LINE_AA
            )
            y_offset += 22

        return overlay_frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = OverlayConfig()
    colors = ColorScheme()
    overlay = Overlay(config, colors)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (128, 128, 128)

    print("Drawing test overlays...")
    frame = overlay.draw_prediction(frame, "fresh", 0.95, colors.fresh, box=(50, 50, 300, 400))
    frame = overlay.draw_performance_stats(frame, 28.5, 35.2, "CUDA")
    frame = overlay.draw_confidence_bar(frame, 0.95, colors.fresh, position="top-right")

    print("Overlay test complete. Display frame (close window to exit).")
    cv2.imshow("Overlay Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
