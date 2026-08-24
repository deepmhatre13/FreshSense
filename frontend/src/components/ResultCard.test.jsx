import { describe, it, expect, vi } from "vitest";
import { act } from "react";
import { createRoot } from "react-dom/client";
import ResultCard from "./ResultCard.jsx";

function renderResult(result, imageBlob = null) {
  const root = document.createElement("div");
  document.body.appendChild(root);
  const r = createRoot(root);
  const onScanAgain = vi.fn();
  act(() => {
    r.render(<ResultCard result={result} imageBlob={imageBlob} onScanAgain={onScanAgain} />);
  });
  return { root, onScanAgain };
}

const baseFruit = (over) => ({
  fruit: "Apple",
  label: "Apple",
  confidence: 0.4,
  detection_confidence: 0.95,
  freshness: "fresh",
  freshness_confidence: 0.88,
  shelf_life: {
    fruit: "Apple",
    freshness_class: "fresh",
    shelf_life_status: "estimated",
    remaining_days: 26,
    typical_min_days: 14,
    typical_max_days: 30,
    unit: "days",
    storage_condition: "ambient",
    explanation: "Based on typical storage...",
  },
  ...over,
});

describe("ResultCard rendering", () => {
  it("renders fresh fruit state correctly", () => {
    const { root } = renderResult({ fruits: [baseFruit({ fruit: "Apple" })] });
    const text = root.textContent;
    expect(text).toMatch(/Apple/);
    expect(text).toMatch(/Fresh/);
    expect(text).toMatch(/88%/);
    expect(text).toMatch(/~26 days/);
  });

  it("renders rotten fruit state correctly", () => {
    const { root } = renderResult({
      fruits: [
        baseFruit({
          fruit: "Apple",
          freshness: "rotten",
          freshness_confidence: 0.9,
          shelf_life: {
            shelf_life_status: "expired",
            remaining_days: 0,
            typical_min_days: 14,
            typical_max_days: 30,
            storage_condition: "ambient",
            explanation: "Expired",
          },
        }),
      ],
    });
    const text = root.textContent;
    expect(text).toMatch(/Rotten/);
    expect(text).toMatch(/Expired/);
    expect(text).toMatch(/consume now or discard/i);
  });

  it("renders uncertain freshness correctly", () => {
    const { root } = renderResult({
      fruits: [
        baseFruit({
          fruit: "Apple",
          freshness: "uncertain",
          freshness_confidence: null,
          shelf_life: {
            shelf_life_status: "uncertain",
            remaining_days: null,
            storage_condition: "ambient",
          },
        }),
      ],
    });
    const text = root.textContent;
    expect(text).toMatch(/Freshness uncertain/i);
    expect(text).not.toMatch(/Rotten/);
  });

  it("renders unsupported freshness without pretending it is fresh", () => {
    const { root } = renderResult({
      fruits: [
        baseFruit({
          fruit: "Strawberry",
          freshness: "unsupported",
          freshness_confidence: null,
          shelf_life: {
            shelf_life_status: "unsupported",
            remaining_days: null,
            storage_condition: "ambient",
          },
        }),
      ],
    });
    const text = root.textContent;
    expect(text).toMatch(/Freshness unavailable/i);
    expect(text).not.toMatch(/Fresh\b/);
  });

    it("renders data_not_available without pretending it is fresh", () => {
    const { root } = renderResult({
      fruits: [
        baseFruit({
          fruit: "Mango",
          freshness: "data_not_available",
          freshness_confidence: null,
          confidence: null,
          shelf_life: {
            shelf_life_status: "data_not_available",
            remaining_days: null,
            storage_condition: "ambient",
          },
        }),
      ],
    });
    const text = root.textContent;
    expect(text).toMatch(/Data not available/i);
    expect(text).not.toMatch(/\bFresh\b/);
    expect(text).toMatch(/Not available/); // shelf-life block
  });

  it("does not display 0 when remaining_days is null", () => {
    const { root } = renderResult({
      fruits: [
        baseFruit({
          fruit: "kiwi",
          shelf_life: {
            shelf_life_status: "estimated",
            remaining_days: null,
            typical_min_days: null,
            typical_max_days: null,
            storage_condition: "ambient",
          },
        }),
      ],
    });
    const text = root.textContent;
    expect(text).not.toMatch(/~0 days/);
    expect(text).toMatch(/Not estimated/i);
  });

  it("loads nutrition from the database by fruit identity", () => {
    const { root } = renderResult({ fruits: [baseFruit({ fruit: "banana" })] });
    const text = root.textContent;
    expect(text).toMatch(/Nutrition \/ 100g/i);
    expect(text).toMatch(/Calories/i);
  });

  it("handles missing nutrition data safely", () => {
    const { root } = renderResult({
      fruits: [baseFruit({ fruit: "dragonfruit" })],
    });
    const text = root.textContent;
    // dragonfruit is not in nutrition.json -> safe fallback, not fake numbers.
    expect(text).toMatch(/Nutrition data unavailable/i);
  });

  it("calls onScanAgain when the Scan Again button is clicked", async () => {
    const { root, onScanAgain } = renderResult({ fruits: [baseFruit({ fruit: "Apple" })] });
    const btn = root.querySelector("button");
    await act(async () => {
      btn.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(onScanAgain).toHaveBeenCalledTimes(1);
  });

  it("renders a captured image when an imageBlob is provided", () => {
    const blob = new Blob(["frame"], { type: "image/jpeg" });
    const { root } = renderResult({ fruits: [baseFruit({ fruit: "Apple" })] }, blob);
    const img = root.querySelector("img");
    expect(img).toBeTruthy();
    expect(img.getAttribute("alt")).toMatch(/Captured/i);
  });
});
