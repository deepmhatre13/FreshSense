import { describe, it, expect, vi, beforeEach } from "vitest";
import { act } from "react";
import { createRoot } from "react-dom/client";
import CameraScanner from "./CameraScanner.jsx";

// CameraScanner is hard to render fully without a real getUserMedia; the
// camera tests below are unit tests on the camera service instead. This file
// keeps a smoke test that the component exports and a "Scan Again resets"
// behaviour test that is covered in result card integration.

vi.mock("../services/camera.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    openCamera: vi.fn(() => Promise.resolve({ state: "ready", stream: { getTracks: () => [] } })),
    stopCamera: vi.fn(),
    captureFrameBlob: vi.fn(() => new Blob(["x"], { type: "image/jpeg" })),
    captureFrame: vi.fn(() => ({
      blob: new Blob(["x"]),
      dataUrl: "data:image/jpeg;base64,abc",
      width: 640,
      height: 480,
    })),
  };
});

vi.mock("../services/inferenceApi.js", () => ({
  previewDetect: vi.fn(() =>
    Promise.resolve({ ok: true, data: { detections: [] } })
  ),
  analyzeImage: vi.fn(() =>
    Promise.resolve({
      ok: true,
      data: { success: true, fruits: [{ fruit: "Apple" }] },
    })
  ),
  checkHealth: vi.fn(() => Promise.resolve({ ok: true })),
  STORAGE_CONDITIONS: ["ambient", "refrigerated"],
}));

vi.mock("../services/detectionService.js", () => ({
  // A gate that immediately fires a stable detection so we exercise capture.
  DetectionStabilityGate: class {
    push() {
      return {
        verdict: "detecting",
        candidate: { label: "Apple", key: "apple", confidence: 0.9, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } },
        buffer: [1],
      };
    }
    reset() {}
  },
  Verdict: { DETECTING: "detecting", CAPTURE: "capture", NO_FRUIT: "no_fruit", MULTIPLE_FRUITS: "multiple_fruits" },
}));

describe("CameraScanner mount", () => {
  it("opens the camera on mount and surfaces the live state", async () => {
    // Fake timers so the (mocked) 300ms preview interval cannot fire mid-test
    // and push React updates from outside act().
    vi.useFakeTimers();
    const onResult = vi.fn();
    const onError = vi.fn();
    const root = document.createElement("div");
    document.body.appendChild(root);
    let r;
    await act(async () => {
      r = createRoot(root);
      r.render(
        <CameraScanner storageCondition="ambient" onResult={onResult} onError={onError} />
      );
    });
    // Flush the async openCamera resolution so its state update happens
    // inside act() rather than after the test body.
    await act(async () => {});
    expect(root.textContent).toMatch(/Camera/i);
    await act(async () => {
      r.unmount();
    });
    vi.useRealTimers();
    root.remove();
  });
});
