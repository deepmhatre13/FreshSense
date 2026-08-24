import { useEffect, useState, useCallback } from "react";
import CameraScanner from "./components/CameraScanner.jsx";
import ResultCard from "./components/ResultCard.jsx";
import ErrorView from "./components/ErrorView.jsx";
import StorageSelector from "./components/StorageSelector.jsx";
import { checkHealth } from "./services/inferenceApi.js";
import "./styles/styles.css";

/** Top-level screens. */
const SCREEN = Object.freeze({
  BOOTSTRAP: "bootstrap",
  LIVE: "live",
  RESULT: "result",
  ERROR: "error",
});

export default function App() {
  const [screen, setScreen] = useState(SCREEN.BOOTSTRAP);
  const [healthErr, setHealthErr] = useState(null);
  const [result, setResult] = useState(null);
  const [storageCondition, setStorageCondition] = useState("ambient");
  const [scanKey, setScanKey] = useState(0);
  // Incremented to re-run the backend readiness probe.
  const [probeKey, setProbeKey] = useState(0);

  // Probe the backend readiness. Re-runs whenever probeKey changes (mount +
  // explicit retries from the error screen). We do NOT hard-fail the whole
  // app if the backend is momentarily unavailable — the user can retry.
  useEffect(() => {
    let cancelled = false;
    checkHealth().then((res) => {
      if (cancelled) return;
      if (!res.ok) {
        setHealthErr(res.error);
        setScreen(SCREEN.ERROR);
      } else {
        setHealthErr(null);
        setScreen(SCREEN.LIVE);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [probeKey]);

  const handleResult = useCallback((payload) => {
    setResult(payload);
    setScreen(SCREEN.RESULT);
  }, []);

  const handleError = useCallback((error) => {
    setHealthErr(error || "Something went wrong.");
    setScreen(SCREEN.ERROR);
  }, []);

  const scanAgain = useCallback(() => {
    setResult(null);
    setHealthErr(null);
    // Force a remount of CameraScanner so tracks/intervals always reset.
    setScanKey((k) => k + 1);
    setScreen(SCREEN.LIVE);
  }, []);

    // Re-probe backend readiness when retrying from an error screen.
  const retryHealth = useCallback(() => {
    setHealthErr(null);
    setScreen(SCREEN.BOOTSTRAP);
    setProbeKey((k) => k + 1); // re-runs the health-check effect
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>SmartFreshAI</h1>
        <StorageSelector
          value={storageCondition}
          onChange={setStorageCondition}
          compact
        />
      </header>

      {screen === SCREEN.BOOTSTRAP && (
        <div className="error-view">
          <div className="status-chip stable" style={{ position: "relative" }}>
            Initializing…
          </div>
          <p style={{ color: "var(--text-dim)" }}>
            Checking the SmartFreshAI server.
          </p>
        </div>
      )}

      {screen === SCREEN.LIVE && (
        <CameraScanner
          key={scanKey}
          storageCondition={storageCondition}
          onResult={handleResult}
          onError={handleError}
        />
      )}

      {screen === SCREEN.RESULT && result && (
        <ResultCard
          result={result.data}
          imageBlob={result.capturedBlob}
          onScanAgain={scanAgain}
        />
      )}

      {screen === SCREEN.ERROR && (
        <ErrorView
          message={healthErr || "Something went wrong."}
          code={healthErr}
          onRetry={healthErr === "SmartFreshAI server is unavailable."
            ? retryHealth
            : scanAgain}
        />
      )}
    </div>
  );
}
