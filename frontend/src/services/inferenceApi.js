/**
 * SmartFreshAI inference API client.
 *
 * Wraps the backend entry points the frontend actually uses:
 *
 *   1. GET  /health
 *      Readiness probe. Confirms the backend (YOLO + EfficientNet) is up
 *      before camera access is attempted.
 *
 *   2. POST /api/v1/detection/preview
 *      Lightweight YOLO-ONLY detection. Called a few times per second during
 *      the live camera preview so the scanner can apply a stability gate before
 *      capturing. It deliberately does NOT run freshness / shelf-life.
 *
 *   3. POST /api/v1/inference/image
 *      Full production pipeline (detect + crop + classify + stabilize +
 *      shelf life). Invoked exactly ONCE per scan, on the captured frame.
 *
 * Backend URL comes from VITE_API_BASE_URL (see frontend/.env.example).
 * No API URL string is hard-coded elsewhere in the app.
 */

const BASE_URL =
  (import.meta && import.meta.env && import.meta.env.VITE_API_BASE_URL) ||
  "http://127.0.0.1:8000";

export const API_BASE_URL = BASE_URL.replace(/\/+$/, "");

export const ENDPOINTS = Object.freeze({
  HEALTH: "/health",
  DETECTION_PREVIEW: "/api/v1/detection/preview",
  INFERENCE_IMAGE: "/api/v1/inference/image",
});

export const STORAGE_CONDITIONS = Object.freeze(["ambient", "refrigerated"]);

/** Envelope so callers never see raw network / stack traces. */
function ok(data) {
  return { ok: true, data };
}

function fail(message, code = "REQUEST_ERROR") {
  return { ok: false, error: message, code };
}

/** Resolve a friendly user-facing message for a non-2xx response. */
async function handleResponse(response) {
  if (response.ok) {
    let data;
    try {
      data = await response.json();
    } catch {
      return fail("Could not parse the server response.");
    }
    return ok(data);
  }

  let detail = "";
  try {
    const body = await response.json();
    detail = body && (body.detail || body.message || "");
  } catch {
    try {
      detail = await response.text();
    } catch {
      detail = "";
    }
  }

  const text = String(detail || "").toLowerCase();
  if (response.status >= 500) {
    // Never surface upstream/internal details (Phase 12/20).
    return fail("SmartFreshAI server is unavailable.", "UNAVAILABLE");
  }
  if (response.status === 400) {
    return fail(detail || "Invalid request.", "BAD_REQUEST");
  }
  if (response.status === 408 || text.includes("timeout")) {
    return fail("Analysis took too long. Please try again.", "TIMEOUT");
  }
  return fail(detail || "The server did not return a valid response.", "SERVER_ERROR");
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.HEALTH}`, {
      method: "GET",
    });
    return handleResponse(res);
  } catch {
    return fail("SmartFreshAI server is unavailable.", "UNAVAILABLE");
  }
}

/**
 * Lightweight YOLO-only detection preview.
 * @param {Blob} frameBlob - JPEG frame captured from the live video canvas.
 * @returns {{ok:boolean, data?: {detections: Array}, error?: string}}
 */
export async function previewDetect(frameBlob) {
  const form = new FormData();
  form.append("image", frameBlob, "frame.jpg");
  try {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.DETECTION_PREVIEW}`, {
      method: "POST",
      body: form,
    });
    return handleResponse(res);
  } catch {
    return fail("SmartFreshAI server is unavailable.", "UNAVAILABLE");
  }
}

/**
 * Submit exactly ONE captured (frozen) frame for full analysis.
 * @param {Blob} imageBlob
 * @param {string} storageCondition - "ambient" | "refrigerated"
 * @param {number} [timeoutMs]
 */
export async function analyzeImage(imageBlob, storageCondition, timeoutMs = 30000) {
  if (storageCondition !== "ambient" && storageCondition !== "refrigerated") {
    return fail("Invalid storage condition.", "BAD_REQUEST");
  }

  const form = new FormData();
  form.append("image", imageBlob, "capture.jpg");
  form.append("storage_condition", storageCondition);

  let controller = null;
  let timeoutId = null;
  let timedOut = false;

  if (typeof AbortSignal !== "undefined" && AbortSignal.timeout) {
    controller = new AbortController();
    timeoutId = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
  }

  try {
    const res = await fetch(`${API_BASE_URL}${ENDPOINTS.INFERENCE_IMAGE}`, {
      method: "POST",
      body: form,
      signal: controller ? controller.signal : undefined,
    });
    if (timedOut) {
      return fail("Analysis took too long. Please try again.", "TIMEOUT");
    }
    return handleResponse(res);
  } catch (e) {
    if (timedOut || (e && e.name === "AbortError")) {
      return fail("Analysis took too long. Please try again.", "TIMEOUT");
    }
    return fail("SmartFreshAI server is unavailable.", "UNAVAILABLE");
    } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

/**
 * Normalize the backend shelf_life object for display.
 *
 * The backend is the single source of truth: this NEVER invents a number.
 * When remaining_days is absent/null the status degrades to "unsupported"
 * with an honest explanation, so the UI never shows a fabricated "0 days".
 *
 * Accepts either the full inference response, a fruit object, or the raw
 * shelf_life object itself.
 */
export function parseShelfLife(payload) {
  const sl =
    (payload && payload.shelf_life) ||
    payload ||
    {};

  const remaining = sl.remaining_days == null ? null : Number(sl.remaining_days);
  const status =
    sl.shelf_life_status ||
    sl.status ||
    (remaining != null ? "estimated" : "unsupported");

  return {
    fruit: sl.fruit != null ? sl.fruit : null,
    freshness_class: sl.freshness_class != null ? sl.freshness_class : null,
    freshness_confidence:
      sl.freshness_confidence != null ? Number(sl.freshness_confidence) : null,
    shelf_life_status: status,
    remaining_days: remaining,
    typical_min_days: sl.typical_min_days != null ? sl.typical_min_days : null,
    typical_max_days: sl.typical_max_days != null ? sl.typical_max_days : null,
    unit: sl.unit || "days",
    basis: sl.basis != null ? sl.basis : null,
    storage_condition: sl.storage_condition || "ambient",
        explanation:
      remaining == null && !sl.explanation
        ? "Remaining shelf life cannot be reliably estimated."
        : sl.explanation || "",
  };
}
