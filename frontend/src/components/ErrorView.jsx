/**
 * Friendly, user-facing error screen.
 *
 * Never surfaces raw Python tracebacks or Axios/fetch internals. The friendly
 * message is mapped from a stable error code/string by the components that
 * produce the error; here we only render what we are given.
 */
const FRIENDLY = {
  permission_denied: "Camera access is required to scan a fruit.",
  no_camera: "No camera was found on this device.",
  unavailable: "SmartFreshAI server is unavailable.",
  timeout: "Analysis took too long. Please try again.",
  corrupted: "Could not analyze this image.",
  no_fruit: "No fruit detected. Move the fruit into the frame.",
  multiple: "Please place one fruit in the frame.",
  unsupported_freshness: "Freshness analysis is not available for this fruit yet.",
  uncertain_freshness: "Freshness could not be determined reliably.",
  null_shelf_life: "Remaining shelf life cannot be reliably estimated.",
  backend_error: "SmartFreshAI server is unavailable.",
};

function friendlyKeyFromMessage(message) {
  const m = String(message || "").toLowerCase();
  if (m.includes("permission")) return "permission_denied";
  if (m.includes("no camera")) return "no_camera";
  if (m.includes("took too long") || m.includes("timeout")) return "timeout";
  if (m.includes("server is unavailable") || m.includes("unavailable")) return "unavailable";
  if (m.includes("could not analyze")) return "corrupted";
  if (m.includes("no fruit")) return "no_fruit";
  if (m.includes("one fruit")) return "multiple";
  if (m.includes("not available for this fruit")) return "unsupported_freshness";
  if (m.includes("not determined reliably")) return "uncertain_freshness";
  if (m.includes("cannot be reliably estimated")) return "null_shelf_life";
  return null;
}

export default function ErrorView({ message, code, onRetry }) {
  const key = code || friendlyKeyFromMessage(message);
  const text = FRIENDLY[key] || message || "Something went wrong.";
  const isServer = key === "unavailable" || key === "backend_error";

  return (
    <div className="error-view">
      <div className="fresh-dot" style={{ background: "var(--danger)" }} />
      <h2 style={{ color: "var(--danger)" }}>{isServer ? "Server unavailable" : "Unable to scan"}</h2>
      <p>{text}</p>
      {onRetry && (
        <button type="button" className="btn restart-btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
