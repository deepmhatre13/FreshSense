/**
 * Vitest global setup (jsdom).
 *
 * 1. jsdom does not implement URL.createObjectURL / revokeObjectURL, which the
 *    result screen legitimately uses to display the captured Blob. Real
 *    browsers always provide these; this polyfill exists purely for tests.
 * 2. The component tests drive React through `act(...)`, which requires the
 *    global IS_REACT_ACT_ENVIRONMENT flag (React 18+) to avoid spurious
 *    warnings.
 */
if (typeof URL !== "undefined" && typeof URL.createObjectURL !== "function") {
  let counter = 0;
  URL.createObjectURL = function (blob) {
    void blob;
    counter += 1;
    return `blob:mock-${counter}`;
  };
  URL.revokeObjectURL = function () {};
}

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

