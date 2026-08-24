/**
 * Live detection + stability gate + auto-capture state machine.
 *
 * Pure (no React, no DOM). Given the latest backend preview detections it
 * decides - from configured thresholds only - whether to keep waiting,
 * capture now, or surface a terminal state (no fruit / multiple fruits).
 *
 * Stability rule:
 *   A capture is triggered only after REQUIRED_STABLE_FRAMES consecutive
 *   frames return the SAME fruit (by label) above MIN_DETECTION_CONFIDENCE,
 *   with no positional drift beyond MAX_BBOX_DRIFT_RATIO. A missing frame,
 *   a different fruit, or a large jump restarts the window (so a shaky or
 *   multi-fruit scene never auto-captures).
 *
 * All thresholds come from src/config/scannerConfig.js.
 */

import { scannerConfig } from "../config/scannerConfig.js";

const {
  MIN_DETECTION_CONFIDENCE,
  REQUIRED_STABLE_FRAMES,
  MAX_BBOX_DRIFT_RATIO,
} = scannerConfig;

function normalizeLabel(label) {
  return String(label == null ? "" : label).trim().toLowerCase();
}

function boxCenter(b) {
  return { x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 };
}
function boxSize(b) {
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  return Math.sqrt(w * w + h * h);
}
/** Positional drift of moving bbox centre relative to ef bbox size. */
function bboxDrift(moving, ref) {
  if (boxSize(ref) <= 0) return 0;
  const cm = boxCenter(moving);
  const cr = boxCenter(ref);
  return Math.hypot(cm.x - cr.x, cm.y - cr.y) / boxSize(ref);
}

export const Verdict = Object.freeze({
  DETECTING: "detecting",
  CAPTURE: "capture",
  NO_FRUIT: "no_fruit",
  MULTIPLE_FRUITS: "multiple_fruits",
});

/**
 * Pure decision over the latest preview detections.
 *
 * @param {Array} latest         raw detections [{label, confidence, bbox}].
 * @param {Array} stableBuffer  previously-accepted consecutive candidates.
 * @param {object|null} lastLocked the most recent candidate reference (for drift).
 * @returns {{verdict, candidate, buffer}}
 */
export function evaluatePreview(latest, stableBuffer, lastLocked) {
  const list = Array.isArray(latest) ? latest : [];
  const valid = list.filter(
    (d) => (Number(d.confidence) || 0) >= MIN_DETECTION_CONFIDENCE
  );

  if (valid.length === 0) {
    return { verdict: Verdict.NO_FRUIT, candidate: null, buffer: [] };
  }
  if (valid.length > 1) {
    return { verdict: Verdict.MULTIPLE_FRUITS, candidate: null, buffer: [] };
  }

  const single = valid[0];
  const candidate = {
    label: single.label,
    key: normalizeLabel(single.label),
    confidence: Number(single.confidence) || 0,
    bbox: single.bbox,
  };

  let buffer;
  if (lastLocked && lastLocked.key === candidate.key) {
    if (bboxDrift(candidate.bbox, lastLocked.bbox) > MAX_BBOX_DRIFT_RATIO) {
      // Fruit moved substantially -> restart the stability window.
      buffer = [candidate];
      return { verdict: Verdict.DETECTING, candidate, buffer };
    }
    // Same fruit, stable position -> accumulate.
    buffer = [...(stableBuffer || []), candidate];
  } else {
    // Nothing locked yet, or a different fruit appeared -> start fresh.
    buffer = [candidate];
  }

  if (buffer.length < REQUIRED_STABLE_FRAMES) {
    return { verdict: Verdict.DETECTING, candidate, buffer };
  }
  // Enough consecutive, same-fruit, stable frames -> lock and capture.
  const locked = buffer[buffer.length - 1];
  return { verdict: Verdict.CAPTURE, candidate: locked, buffer };
}

/** Stateful wrapper consumed by the live scanner component. */
export class DetectionStabilityGate {
  constructor() {
    this.stableBuffer = [];
    this.lockedCandidate = null;
  }

  reset() {
    this.stableBuffer = [];
    this.lockedCandidate = null;
  }

  /** Feed the latest raw preview detections through the gate. */
  push(detections) {
    const result = evaluatePreview(
      detections,
      this.stableBuffer,
      this.lockedCandidate
    );

    if (result.verdict === Verdict.DETECTING) {
      // Accumulate the growing window and advance the drift reference.
      this.stableBuffer = result.buffer;
      this.lockedCandidate = result.candidate;
    } else if (result.verdict === Verdict.CAPTURE) {
      this.lockedCandidate = result.candidate;
      this.stableBuffer = []; // consumed; reset for the next scan
    } else {
      // NO_FRUIT / MULTIPLE_FRUITS -> clear stale context so a shaky scene
      // does not carry an old candidate into capture.
      this.stableBuffer = [];
      this.lockedCandidate = null;
    }

    return result;
  }
}
