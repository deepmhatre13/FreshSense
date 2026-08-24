import { STORAGE_CONDITIONS } from "../services/inferenceApi.js";

const LABELS = {
  ambient: "Ambient",
  refrigerated: "Refrigerated",
};

/**
 * Simple ambient / refrigerated toggle.
 *
 * Storage condition is user-provided CONTEXT, not a measured value — the
 * system has no temperature sensors. This is surfaced via the helper text.
 * The value persists for the whole scan flow and is sent with the single
 * analysis request.
 *
 * @param {string} value  "ambient" | "refrigerated"
 * @param {function} onChange
 * @param {boolean} [compact]  smaller style for the header bar
 */
export default function StorageSelector({ value, onChange, compact = false }) {
  return (
    <div className={compact ? "storage-selector" : "storage-selector block"}>
      {STORAGE_CONDITIONS.map((c) => (
        <button
          type="button"
          key={c}
          className={value === c ? "active" : ""}
          onClick={() => onChange(c)}
          aria-pressed={value === c}
        >
          {LABELS[c]}
        </button>
      ))}
    </div>
  );
}
