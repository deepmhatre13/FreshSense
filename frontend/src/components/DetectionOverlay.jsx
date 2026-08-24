/**
 * Project a detection bounding box (frame pixel coords) onto the CSS pixels of
 * the rendered <video> element, assuming object-fit: contain inside its
 * positioned container.
 *
 * @param {HTMLVideoElement} videoEl
 * @param {{x1,y1,x2,y2}} bbox  raw pixel bbox (frame coords)
 * @returns {{left,top,width,height}} CSS pixel rect within the container
 */
export function projectBbox(videoEl, bbox) {
  const vw = videoEl.videoWidth || 1;
  const vh = videoEl.videoHeight || 1;
  const rect = videoEl.getBoundingClientRect();
  const ratio = Math.min(rect.width / vw, rect.height / vh);
  const dW = vw * ratio;
  const dH = vh * ratio;
  const offX = (rect.width - dW) / 2 + rect.left;
  const offY = (rect.height - dH) / 2 + rect.top;
  const bx1 = bbox.x1,
    by1 = bbox.y1,
    bx2 = bbox.x2,
    by2 = bbox.y2;
  const left = offX + (bx1 / vw) * dW;
  const top = offY + (by1 / vh) * dH;
  const width = ((bx2 - bx1) / vw) * dW;
  const height = ((by2 - by1) / vh) * dH;
  return { left, top, width, height };
}

/** Short, human-friendly status text for each scanner sub-state. */
export function statusText(state, detection) {
  switch (state) {
    case "live":
      return detection
        ? "Hold steady..."
        : "Point the camera at a fruit";
    case "detecting":
      return "Hold steady...";
    case "stable":
      return "Fruit locked — capturing...";
    case "capturing":
      return "Capturing...";
    case "analyzing":
      return "Analyzing…";
    default:
      return "Point the camera at a fruit";
  }
}

export default function DetectionOverlay({ scanState, detection, videoRef }) {
  if (!detection) {
    // No detection: still show a status chip for "live" guidance.
    return (
      <div className="detection-overlay">
        <div className="status-chip live">{statusText(scanState, detection)}</div>
      </div>
    );
  }

          // The parent passes the live <video> React ref so this component can map
  // frame coordinates to the CSS pixels of the rendered video element.
  const vEl = videoRef && videoRef.current;

  let style = null;
  if (vEl && detection.bbox) {
    style = projectBbox(vEl, detection.bbox);
  }

  return (
    <div className="detection-overlay">
      <div className={`status-chip ${scanState}`}>
        <span
          className="dot"
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background:
              scanState === "stable" || scanState === "capturing" || scanState === "analyzing"
                ? "#22c55e"
                : "#f59e0b",
          }}
        />
        <span>{statusText(scanState, detection)}</span>
      </div>

      {style && (
        <>
          <div className="label-bubble" style={{ left: style.left + 4, top: style.top - 22 }}>
            {detection.label} • {Math.round(detection.confidence * 100)}%
          </div>
          <div
            className="bounding-box"
            style={{
              left: style.left,
              top: style.top,
              width: Math.max(2, style.width),
              height: Math.max(2, style.height),
            }}
          />
        </>
      )}
    </div>
  );
}
