import { describe, it, expect, beforeEach } from "vitest";
import { DetectionStabilityGate, Verdict, evaluatePreview } from "../services/detectionService.js";
import { scannerConfig } from "../config/scannerConfig.js";

function det(label, conf, bbox) {
  return { label, confidence: conf, bbox: bbox || { x1: 0, y1: 0, x2: 10, y2: 10 } };
}

const APPLE = det("Apple", 0.9, { x1: 5, y1: 5, x2: 15, y2: 15 });

describe("evaluatePreview (pure)", () => {
  it("returns NO_FRUIT when no detections pass the threshold", () => {
    const r = evaluatePreview([det("Apple", 0.3)], [], null);
    expect(r.verdict).toBe(Verdict.NO_FRUIT);
  });

  it("returns MULTIPLE_FRUITS when more than one valid fruit is present", () => {
    const r = evaluatePreview(
      [det("Apple", 0.9), det("banana", 0.85)],
      [],
      null
    );
    expect(r.verdict).toBe(Verdict.MULTIPLE_FRUITS);
  });

  it("returns DETECTING for stable-but-not-yet-enough frames", () => {
    const r = evaluatePreview([APPLE], [], null);
    expect(r.verdict).toBe(Verdict.DETECTING);
    expect(r.buffer.length).toBe(1);
  });
});

describe("DetectionStabilityGate", () => {
  let gate;
  beforeEach(() => {
    gate = new DetectionStabilityGate();
  });

  it("does not fire CAPTURE before REQUIRED_STABLE_FRAMES consecutive same-fruit frames", () => {
    for (let i = 0; i < scannerConfig.REQUIRED_STABLE_FRAMES - 1; i++) {
      const r = gate.push([APPLE]);
      expect(r.verdict).toBe(Verdict.DETECTING);
    }
    const r = gate.push([APPLE]);
    expect(r.verdict).toBe(Verdict.CAPTURE);
  });

  it("resets the window when a different fruit appears (no premature capture)", () => {
    for (let i = 0; i < scannerConfig.REQUIRED_STABLE_FRAMES - 1; i++) {
      expect(gate.push([APPLE]).verdict).toBe(Verdict.DETECTING);
    }
    // A banana shows up -> window restarts, must not capture the apple.
    const r = gate.push([det("banana", 0.9)]);
    expect(r.verdict).toBe(Verdict.DETECTING);
    expect(r.candidate.key).toBe("banana");
  });

  it("does not capture when the fruit jumps suddenly (positional drift)", () => {
    const stable = det("Apple", 0.95, { x1: 5, y1: 5, x2: 15, y2: 15 });
    for (let i = 0; i < scannerConfig.REQUIRED_STABLE_FRAMES - 1; i++) {
      expect(gate.push([stable]).verdict).toBe(Verdict.DETECTING);
    }
    // Now the apple appears at a wildly different position.
    const jumped = det("Apple", 0.95, { x1: 80, y1: 80, x2: 90, y2: 90 });
    const r = gate.push([jumped]);
    expect(r.verdict).toBe(Verdict.DETECTING);
  });

  it("returns NO_FRUIT when the fruit leaves and none remain above threshold", () => {
    expect(gate.push([det("Apple", 0.2)]).verdict).toBe(Verdict.NO_FRUIT);
  });

  it("reset() clears accumulated state", () => {
    for (let i = 0; i < scannerConfig.REQUIRED_STABLE_FRAMES - 1; i++) {
      gate.push([APPLE]);
    }
    gate.reset();
    expect(gate.push([APPLE]).buffer.length).toBe(1);
  });
});
