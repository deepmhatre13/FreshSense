/**
 * Browser camera access and frame capture helpers.
 *
 * Responsibilities:
 *  - request the PHONE BACK (environment-facing) camera
 *  - surface permission / track errors cleanly to the caller
 *  - capture a still frame from the live video into a JPEG blob (optionally
 *    downscaled for the lightweight detection preview)
 *  - release MediaStream tracks on teardown (avoids leaks / camera staying on)
 */

/**
 * Open the rear camera and attach it to a video element.
 *
 * @param {HTMLVideoElement} videoEl
 * @param {object} [opts]
 * @param {boolean} [opts.preferRear=true] - use facingMode "environment".
 * @param {number} [opts.width]
 * @param {number} [opts.height]
 * @returns {Promise<{stream: MediaStream, state: string}>}
 *   state is one of: "ready" | "permission_denied" | "no_camera" | "error"
 */
export async function openCamera(videoEl, opts = {}) {
  const preferRear = opts.preferRear !== false;
  const width = opts.width || 640;
  const height = opts.height || 480;

  const constraints = {
    video: {
      facingMode: preferRear ? { ideal: "environment" } : "user",
      width: { ideal: width },
      height: { ideal: height },
    },
    audio: false,
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = stream;
    // Some browsers require playing() before drawing frames; resolve on play.
    if (videoEl.readyState >= 2 /* HAVE_CURRENT_DATA */) {
      // already have data; nothing to await
    } else {
      await new Promise((resolve) => {
        const onPlaying = () => {
          videoEl.removeEventListener("playing", onPlaying);
          resolve();
        };
        videoEl.addEventListener("playing", onPlaying);
        videoEl.play().catch(() => resolve()); // fall through regardless
      });
    }
    return { stream, state: "ready" };
  } catch (err) {
    if (err && err.name === "NotAllowedError") {
      return { stream: null, state: "permission_denied" };
    }
    if (err && err.name === "NotFoundError") {
      return { stream: null, state: "no_camera" };
    }
    return { stream: null, state: "error" };
  }
}

/** Stop all video/audio tracks of a stream and detach it. */
export function stopCamera(videoEl, stream) {
  try {
    if (stream && stream.getTracks) {
      stream.getTracks().forEach((t) => t.stop());
    }
  } catch {
    /* ignore */
  }
  if (videoEl) {
    try {
      videoEl.srcObject = null;
    } catch {
      /* ignore */
    }
  }
}

/**
 * Draw the current video frame into an off-screen canvas, resize to fit
 * within maxW x maxH (preserving aspect ratio) and return a JPEG blob.
 * The browser keeps the preview stream local; only the small JPEG is uploaded.
 *
 * @param {HTMLVideoElement} videoEl
 * @param {number} [maxW]  default 320 for the preview loop
 * @param {number} [maxH]
 * @param {number} [quality] JPEG quality 0-1
 * @returns {Blob|null} null if the video has no frames yet
 */
export function captureFrameBlob(videoEl, maxW = 320, maxH = 320, quality = 0.82) {
  if (!videoEl || videoEl.readyState < 2) {
    return null;
  }
  // Use intrinsic video dimensions; fall back to element display size.
  const vw = videoEl.videoWidth || videoEl.offsetWidth || maxW;
  const vh = videoEl.videoHeight || videoEl.offsetHeight || maxH;

  let dw = vw;
  let dh = vh;
  const ratio = Math.min(maxW / vw, maxH / vh, 1);
  if (vw > maxW || vh > maxH) {
    dw = Math.round(vw * ratio);
    dh = Math.round(vh * ratio);
  }

  const canvas = document.createElement("canvas");
  canvas.width = dw;
  canvas.height = dh;
    const ctx = canvas.getContext("2d");
  if (!ctx) {
    canvas.remove();
    return null;
  }
  ctx.drawImage(videoEl, 0, 0, dw, dh);

  // toDataURL is synchronous and the simplest reliable cross-browser path.
  const dataUrl = canvas.toDataURL("image/jpeg", quality);
  canvas.remove();
  return dataURLToBlob(dataUrl);
}

/** Decode a data:URL to a Blob. */

/** Decode a data:URL to a Blob. */
export function dataURLToBlob(dataurl) {
  const parts = dataurl.split(",");
  if (parts.length < 2) return null;
  const mimeMatch = parts[0].match(/:(.*?);/);
  const mime = mimeMatch ? mimeMatch[1] : "image/jpeg";
  const bstr = atob(parts[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) u8arr[n] = bstr.charCodeAt(n);
  return new Blob([u8arr], { type: mime });
}

/**
 * Draw the *current* video frame at full (capture) resolution onto a target
 * canvas (used to freeze the captured frame in the UI while inference runs).
 *
 * @param {HTMLVideoElement} videoEl
 * @param {HTMLCanvasElement} targetCanvas
 * @param {number} [maxW]
 * @param {number} [maxH]
 */
export function drawCurrentFrame(videoEl, targetCanvas, maxW = 640, maxH = 640) {
  if (!videoEl || !targetCanvas) return false;
  const vw = videoEl.videoWidth || targetCanvas.width;
  const vh = videoEl.videoHeight || targetCanvas.height;
  let dw = vw;
  let dh = vh;
  const ratio = Math.min(maxW / vw, maxH / vh, 1);
  if (vw > maxW || vh > maxH) {
    dw = Math.round(vw * ratio);
    dh = Math.round(vh * ratio);
  }
  targetCanvas.width = dw;
  targetCanvas.height = dh;
     const ctx = targetCanvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, dw, dh);
  return true;
}

/**
 * Capture the current video frame returning BOTH a Blob (for upload to the
 * full analysis endpoint) and a dataURL (for display of the frozen frame).
 *
 * @param {HTMLVideoElement} videoEl
 * @param {number} [maxW]  default 640
 * @param {number} [maxH]
 * @param {number} [quality] JPEG quality 0-1
 * @returns {{blob: Blob|null, dataUrl: string|null, width: number, height: number}|null}
 */
export function captureFrame(videoEl, maxW = 640, maxH = 640, quality = 0.9) {
  if (!videoEl || videoEl.readyState < 2) {
    return null;
  }
  const vw = videoEl.videoWidth || videoEl.offsetWidth || maxW;
  const vh = videoEl.videoHeight || videoEl.offsetHeight || maxH;

  let dw = vw;
  let dh = vh;
  const ratio = Math.min(maxW / vw, maxH / vh, 1);
  if (vw > maxW || vh > maxH) {
    dw = Math.round(vw * ratio);
    dh = Math.round(vh * ratio);
  }

  const canvas = document.createElement("canvas");
  canvas.width = dw;
  canvas.height = dh;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    canvas.remove();
    return null;
  }
  ctx.drawImage(videoEl, 0, 0, dw, dh);
  const dataUrl = canvas.toDataURL("image/jpeg", quality);
  canvas.remove();
  return { blob: dataURLToBlob(dataUrl), dataUrl, width: dw, height: dh };
}
