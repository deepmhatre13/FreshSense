import { useEffect, useRef, useState } from "react";
import {
  openCamera,
  stopCamera,
  captureFrameBlob,
  captureFrame,
} from "../services/camera.js";
import {
  previewDetect,
  analyzeImage,
} from "../services/inferenceApi.js";
import {
  DetectionStabilityGate,
  Verdict,
} from "../services/detectionService.js";
import { scannerConfig } from "../config/scannerConfig.js";
import DetectionOverlay from "./DetectionOverlay.jsx";
import ErrorView from "./ErrorView.jsx";

const CAM = Object.freeze({
  REQUESTING: "requesting",
  READY: "ready",
  ERROR: "error",
});
const SCAN = Object.freeze({
  LIVE: "live",
  DETECTING: "detecting",
  STABLE: "stable",
  CAPTURING: "capturing",
  ANALYZING: "analyzing",
});

export default function CameraScanner({ storageCondition, onResult, onError }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const previewPendingRef = useRef(false);
  const gateRef = useRef(new DetectionStabilityGate());
  const storageRef = useRef(storageCondition || "ambient");

  const [camState, setCamState] = useState(CAM.REQUESTING);
  const [camErrorKey, setCamErrorKey] = useState(null);
  const [scanState, setScanState] = useState(SCAN.LIVE);
  const [detection, setDetection] = useState(null);
  const [statusMessage, setStatusMessage] = useState("Place a fruit in the frame");
  const [frozenDataUrl, setFrozenDataUrl] = useState(null);

  // Keep the latest storage condition in a ref so a capture fires with the
  // current user choice even before React re-renders.
  useEffect(() => {
    storageRef.current = storageCondition || "ambient";
  }, [storageCondition]);

  // --- Open / stop the rear camera on mount ---
  useEffect(() => {
    let mounted = true;
    const video = videoRef.current;
    openCamera(video).then((res) => {
      if (!mounted) return;
      if (res.state === "ready") {
        streamRef.current = res.stream;
        setCamState(CAM.READY);
        setCamErrorKey(null);
      } else {
        streamRef.current = null;
        setCamState(CAM.ERROR);
        setCamErrorKey(res.state);
      }
    });

    return () => {
      mounted = false;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      stopCamera(video, streamRef.current);
      streamRef.current = null;
    };
  }, []);

  // --- Re-request camera from the inline camera-error screen ---
  const retryCamera = () => {
    setCamState(CAM.REQUESTING);
    setCamErrorKey(null);
    const video = videoRef.current;
    openCamera(video).then((res) => {
      if (res.state === "ready") {
        streamRef.current = res.stream;
        setCamState(CAM.READY);
        setCamErrorKey(null);
      } else {
        streamRef.current = null;
        setCamState(CAM.ERROR);
        setCamErrorKey(res.state);
      }
    });
  };

  // --- Live detection loop (throttled YOLO-only preview) ---
  useEffect(() => {
    if (camState !== CAM.READY) return;
    if (scanState === SCAN.CAPTURING || scanState === SCAN.ANALYZING) return;
    const video = videoRef.current;
    if (!video) return;

    let active = true;
    const { DETECTION_INTERVAL_MS, PREVIEW_WIDTH } = scannerConfig;

    const id = setInterval(() => {
      if (!active) return;
      if (previewPendingRef.current) return;
      if (video.readyState < 2) return;
      const blob = captureFrameBlob(video, PREVIEW_WIDTH, PREVIEW_WIDTH, 0.8);
      if (!blob) return;
      previewPendingRef.current = true;
      previewDetect(blob).then((res) => {
        previewPendingRef.current = false;
        if (!active) return;
        if (!res.ok) {
          setStatusMessage(res.error || "Connection problem");
          return;
        }
        handlePreviewResult(res.data);
      });
    }, DETECTION_INTERVAL_MS);

    intervalRef.current = id;
    return () => {
      active = false;
      clearInterval(id);
      intervalRef.current = null;
    };
    }, [camState, scanState]);

  // --- Process one preview detection batch through the stability gate ---
  function handlePreviewResult(data) {
    const detections = data && data.detections ? data.detections : [];
    const gate = gateRef.current;
    const gateResult = gate.push(detections);

    // Show the best raw detection in the overlay regardless of verdict.
    let top = null;
    if (detections.length) {
      const sorted = detections
        .slice()
        .sort((a, b) => b.confidence - a.confidence);
      const best = sorted[0];
      top = {
        label: best.label,
        confidence: Number(best.confidence) || 0,
        bbox: best.bbox,
        low: (Number(best.confidence) || 0) < scannerConfig.MIN_DETECTION_CONFIDENCE,
      };
    }
    setDetection(top);

    switch (gateResult.verdict) {
      case Verdict.NO_FRUIT:
        setScanState(SCAN.LIVE);
        setStatusMessage("No fruit detected. Move the fruit into the frame.");
        break;
      case Verdict.MULTIPLE_FRUITS:
        setScanState(SCAN.LIVE);
        setStatusMessage("Please place one fruit in the frame.");
        break;
      case Verdict.CAPTURE:
        captureAndAnalyze();
        break;
      case Verdict.DETECTING:
      default:
        setScanState(SCAN.DETECTING);
        setStatusMessage(
          (gateResult.candidate && gateResult.candidate.label) || "Hold steady..."
        );
        break;
    }
  }

  // --- Capture the frozen frame + run full analysis exactly once ---
  async function captureAndAnalyze() {
    const video = videoRef.current;
    const cap = captureFrame(
      video,
      scannerConfig.MAX_CAPTURE_WIDTH,
      scannerConfig.MAX_CAPTURE_HEIGHT,
      0.92
    );
    if (!cap) {
      onError && onError("Could not analyze this image.");
      return;
    }

    // Freeze the frozen frame into the UI and stop the live loop visually.
    setFrozenDataUrl(cap.dataUrl);
    setScanState(SCAN.ANALYZING);
    setStatusMessage("Analyzing...");

    const storage = storageRef.current || "ambient";
    const res = await analyzeImage(cap.blob, storage, scannerConfig.INFERENCE_TIMEOUT_MS);
    if (res.ok) {
      onResult && onResult({ data: res.data, capturedBlob: cap.blob });
    } else {
      setFrozenDataUrl(null);
      onError && onError(res.error);
    }
  }

    // --- Inline camera-error retry (does NOT unmount the scanner) ---
  function camErrorFriendly(key) {
    switch (key) {
      case "permission_denied":
        return "Camera access is required to scan a fruit.";
      case "no_camera":
        return "No camera was found on this device.";
      case "error":
        return "Could not start the camera. Please try again.";
      default:
        return "SmartFreshAI server is unavailable.";
    }
  }
  if (camState === CAM.ERROR) {
    const friendly = camErrorFriendly(camErrorKey);
    return (
      <ErrorView message={friendly} code={camErrorKey} onRetry={retryCamera} />
    );
  }

  const isFrozen = Boolean(frozenDataUrl);

  return (
    <div className="scanner">
      <div className="camera-view">
        <div className="video-wrap">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className={isFrozen ? "hidden" : ""}
            aria-label="Live camera preview"
          />
          {!isFrozen && (
            <DetectionOverlay
              scanState={scanState}
              detection={detection}
              videoRef={videoRef}
            />
          )}
        </div>

        {isFrozen && frozenDataUrl && (
          <img
            src={frozenDataUrl}
            alt="Captured frame"
            className="captured-image"
            style={{ position: "absolute", inset: 0, objectFit: "contain" }}
          />
        )}
      </div>

      {!isFrozen && (
        <div className="instruction">
          <div className="status-chip stable" style={{ justifyContent: "center" }}>
            {scanState === SCAN.DETECTING && detection
              ? `${detection.label} • ${Math.round(detection.confidence * 100)}%`
              : scanState === SCAN.LIVE
              ? statusMessage
              : statusMessage}
          </div>
          <p style={{ marginTop: 8 }}>{statusMessage}</p>
        </div>
      )}
    </div>
  );
}
