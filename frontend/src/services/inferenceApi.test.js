import { describe, it, expect, vi, beforeEach } from "vitest";
import { analyzeImage, previewDetect, parseShelfLife, STORAGE_CONDITIONS } from "../services/inferenceApi.js";

function makeFetchResponse(body) {
  return {
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
  };
}

describe("analyzeImage (multipart + storage)", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
    global.FileReader = global.FileReader || undefined;
  });

  it("uploads a multipart/form-data body with the image blob", async () => {
    const blob = new Blob(["fake-bytes"], { type: "image/jpeg" });
    global.fetch.mockResolvedValueOnce(makeFetchResponse({ success: true }));

    await analyzeImage(blob, "ambient");

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url, opts] = global.fetch.mock.calls[0];
    expect(url).toMatch(/\/inference\/image$/);
    expect(opts.method).toBe("POST");
    // The browser must generate the multipart boundary itself: the client
    // deliberately does NOT hard-code a Content-Type for FormData bodies.
    expect(opts.body).toBeInstanceOf(FormData);
    expect(
      opts.headers && (opts.headers["Content-Type"] || opts.headers.get)
        ? opts.headers["Content-Type"]
        : undefined
    ).toBeUndefined();
        expect(opts.body.get("storage_condition")).toBe("ambient");
    // jsdom wraps appended Blobs into File objects; assert the payload
    // survived intact rather than relying on reference identity.
    const uploaded = opts.body.get("image");
    expect(uploaded).toBeInstanceOf(Blob);
    expect(uploaded.size).toBe(blob.size);
  });

  it("sends the selected storage_condition", async () => {
    const blob = new Blob(["x"], { type: "image/jpeg" });
    global.fetch.mockResolvedValue(makeFetchResponse({ success: true }));
    await analyzeImage(blob, "refrigerated");
    const [, opts] = global.fetch.mock.calls[0];
    expect(opts.body.get("storage_condition")).toBe("refrigerated");
  });

  it("returns {ok:false} on HTTP 5xx with a friendly server message", async () => {
    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: () => Promise.resolve({ detail: "upstream down" }),
    });
    const res = await analyzeImage(new Blob(["x"]), "ambient");
    expect(res.ok).toBe(false);
    expect(res.error).toBe("SmartFreshAI server is unavailable.");
  });
});

describe("previewDetect (throttled lightweight detection)", () => {
  it("returns parsed {ok:true} detections on a 200", async () => {
    global.fetch = vi.fn().mockResolvedValueOnce(
      makeFetchResponse({
        success: true,
        detections: [{ label: "Apple", confidence: 0.9, bbox: { x1: 0, y1: 0, x2: 10, y2: 10 } }],
      })
    );
    const res = await previewDetect(new Blob(["x"]));
    expect(res.ok).toBe(true);
    expect(res.data.detections).toHaveLength(1);
  });

  it("returns {ok:false} on a network error", async () => {
    global.fetch = vi.fn().mockRejectedValueOnce(new Error("disconnected"));
    const res = await previewDetect(new Blob(["x"]));
    expect(res.ok).toBe(false);
    expect(res.error).toMatch(/server is unavailable/i);
  });
});

describe("parseShelfLife (honest null handling)", () => {
  it("does not fabricate a remaining_days when the backend omits it", () => {
    const raw = {
      shelf_life: {
        remaining_days: null,
        typical_min_days: 14,
        typical_max_days: 30,
        storage_condition: "ambient",
      },
    };
    const parsed = parseShelfLife(raw);
    expect(parsed.remaining_days).toBeNull();
    expect(parsed.shelf_life_status).toBe("unsupported");
    expect(parsed.explanation).toMatch(/cannot be reliably estimated/i);
  });

  it("preserves a real estimate", () => {
    const raw = {
      shelf_life: {
        remaining_days: 26,
        typical_min_days: 14,
        typical_max_days: 30,
        storage_condition: "ambient",
        status: "estimated",
      },
    };
    const parsed = parseShelfLife(raw);
    expect(parsed.remaining_days).toBe(26);
    expect(parsed.shelf_life_status).toBe("estimated");
  });
});

describe("STORAGE_CONDITIONS contract", () => {
  it("offers ambient and refrigerated", () => {
    expect(STORAGE_CONDITIONS).toContain("ambient");
    expect(STORAGE_CONDITIONS).toContain("refrigerated");
  });
});
