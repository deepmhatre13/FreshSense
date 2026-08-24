// Scanner configuration.
//
// All magic numbers that drive the live detection / capture flow live here so
// they can be tuned without scattering unexplained constants through components.
// Defaults are aligned with the production detector configuration in
// configs/settings.yaml (detection.confidence = 0.45, yolo imgsz = 640).

const scannerConfig = Object.freeze({
  // Minimum raw YOLO detection confidence required to consider a detection
  // valid during the live preview loop. The backend already filters with its
  // own threshold (0.45); this is an additional frontend gate.
  MIN_DETECTION_CONFIDENCE: 0.5,

  // Number of consecutive stable frames required before the scanner considers
  // the detection locked and triggers a capture.
  REQUIRED_STABLE_FRAMES: 4,

  // Milliseconds between live detection requests to the preview endpoint.
  // 300ms = roughly 3-4 requests/sec worst case. The backend is never
  // hammered at full camera framerate.
  DETECTION_INTERVAL_MS: 300,

  // Maximum capture resolution used for the frozen analysis image.
  // Larger than this is downscaled before upload to keep inference fast.
  MAX_CAPTURE_WIDTH: 640,
  MAX_CAPTURE_HEIGHT: 640,

  // Preview frame resolution sent to the detection endpoint. Small enough to
  // keep preview requests cheap, good enough for YOLO.
  PREVIEW_WIDTH: 320,
  PREVIEW_HEIGHT: 320,

  // Max ms to wait for the full analysis inference response before surfacing a
  // friendly timeout error.
  INFERENCE_TIMEOUT_MS: 30000,

  // Bounding-box positional drift (fraction of averaged box size) allowed
  // while counting a frame as "still" for stability.
  MAX_BBOX_DRIFT_RATIO: 0.2,
});

export { scannerConfig };
