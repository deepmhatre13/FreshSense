import { useEffect } from "react";
import { getNutrition } from "../services/nutritionService.js";
import { FRUIT_EMOJIS } from "../data/fruitEmojis.js";

/** Pretty-print a detected fruit label. */
function displayLabel(fruit) {
  if (!fruit) return "Fruit";
  return String(fruit)
    .trim()
    .replace(/(?:^|\s)\w/g, (c) => c.toUpperCase());
}

function pct(conf) {
  if (conf == null || Number.isNaN(conf)) return "—";
  return `${Math.round(Number(conf) * 100)}%`;
}

/**
 * Freshness display model derived purely from the backend freshness class.
 * We never fabricate a freshness value the backend did not provide.
 * (Freshness = ML via EfficientNet-B0; surfaced honestly.)
 */
function freshnessDisplay(fruitResult) {
  const cls = (fruitResult.freshness || "unknown").toLowerCase();
  const conf = fruitResult.freshness_confidence;
  const detConf = fruitResult.confidence;
  switch (cls) {
    case "fresh":
      return {
        label: "Fresh",
        confidence: conf != null ? pct(conf) : pct(detConf),
        dotClass: "fresh",
      };
    case "stale":
      return { label: "Stale", confidence: pct(conf), dotClass: "warning" };
    case "rotten":
      return { label: "Rotten", confidence: pct(conf), dotClass: "rotten" };
    case "uncertain":
      return { label: "Freshness uncertain", confidence: "—", dotClass: "warning" };
    case "data_not_available":
      return {
        label: "Data not available",
        confidence: "—",
        dotClass: "warning",
      };
    case "unsupported":
      return {
        label: "Freshness unavailable",
        confidence: "—",
        dotClass: "warning",
      };
    case "unknown":
      return { label: "Freshness unknown", confidence: "—", dotClass: "warning" };
    default:
      return { label: "Freshness unknown", confidence: "—", dotClass: "warning" };
  }
}

function NutritionRow({ label, value, unit }) {
  const text = value == null ? "Nutrition data unavailable" : `${value} ${unit}`;
  return (
    <>
      <span className="label">{label}</span>
      <span className={`value ${value == null ? "dim" : ""}`}>{text}</span>
    </>
  );
}

/**
 * ResultCard renders the frozen captured image plus the full backend result
 * (freshness + shelf life + nutrition). No value here is hardcoded by fruit.
 */
export default function ResultCard({ result, imageBlob, onScanAgain }) {
  // `result` is the InferenceResponse: { success, message, fruits:[...] }
  const fruits = (result && result.fruits) || [];
  const fruit = fruits[0];

  const freshness = fruit ? freshnessDisplay(fruit) : null;
  const shelf = fruit && fruit.shelf_life ? fruit.shelf_life : null;
  const nutrition = fruit ? getNutrition(fruit.fruit) : getNutrition(null);
  const emoji = FRUIT_EMOJIS[(fruit && fruit.fruit) || ""] || "🍎";

  const imageUrl =
    imageBlob instanceof Blob
      ? URL.createObjectURL(imageBlob)
      : imageBlob || "";

  // Resolve the per-status shelf-life display (never fabricate a number the
  // backend returned as None).
  let remainingBlock = null;
  if (shelf) {
    if (shelf.shelf_life_status === "expired") {
      remainingBlock = (
        <div className="shelf-life-value" style={{ color: "var(--danger)" }}>
          Expired — consume now or discard
        </div>
      );
    } else if (shelf.shelf_life_status === "estimated") {
      if (shelf.remaining_days != null) {
        remainingBlock = (
          <div className="shelf-life-value">~{shelf.remaining_days} days</div>
        );
      } else {
        remainingBlock = <div className="shelf-life-value">Not estimated</div>;
      }
    } else {
      remainingBlock = <div className="shelf-life-value">Not available</div>;
    }
  }

  const storageLabel = shelf && shelf.storage_condition
    ? shelf.storage_condition.charAt(0).toUpperCase() + shelf.storage_condition.slice(1)
    : "Ambient";

  return (
    <div className="result-card">
      {imageUrl && (
        <img src={imageUrl} alt="Captured fruit" className="captured-image" />
      )}

      <section className="card-section">
        <h2>Detected fruit</h2>
        <p className="fruit-title">
          <span aria-label={fruit ? fruit.fruit : "fruit"}>
            {emoji} {fruit ? displayLabel(fruit.fruit) : "—"}
          </span>
        </p>
        {fruit && fruit.detection_confidence != null && (
          <p className="shelf-life-sub">
            Detection confidence {pct(fruit.detection_confidence)}
          </p>
        )}
      </section>

      <section className="card-section">
        <h2>Freshness</h2>
        {freshness ? (
          <>
            <div className="freshness-row">
              <span className={`fresh-dot ${freshness.dotClass}`} />
              <span
                className={
                  freshness.dotClass === "fresh"
                    ? "fresh"
                    : freshness.dotClass === "rotten"
                    ? "rotten"
                    : "warning-text"
                }
              >
                {freshness.label}
              </span>
              <span className="shelf-life-sub">{freshness.confidence} confidence</span>
            </div>
            <p className="shelf-life-sub">
              Freshness is graded by an ML model (EfficientNet-B0), not by
              image-derived rules.
            </p>
          </>
        ) : (
          <p className="shelf-life-sub">Freshness data unavailable</p>
        )}
      </section>

      <section className="card-section">
        <h2>Shelf life</h2>
        {shelf ? (
          <>
            {remainingBlock}
            <p className="shelf-life-sub">
              {shelf.typical_min_days != null && shelf.typical_max_days != null
                ? `Typical range: ${shelf.typical_min_days}–${shelf.typical_max_days} days`
                : "Typical range: not available"}
            </p>
            <p className="shelf-life-sub">Storage: {storageLabel}</p>
            {shelf.explanation && (
              <p className="shelf-life-sub">{shelf.explanation}</p>
            )}
          </>
        ) : (
          <p className="shelf-life-sub">
            Remaining shelf life cannot be reliably estimated.
          </p>
        )}
      </section>

      <section className="card-section">
        <h2>Nutrition / 100g</h2>
        {nutrition.available ? (
          <div className="nutrition-grid">
            <NutritionRow label="Calories" value={nutrition.calories_kcal} unit="kcal" />
            <NutritionRow label="Carbohydrates" value={nutrition.carbohydrates_g} unit="g" />
            <NutritionRow label="Fiber" value={nutrition.fiber_g} unit="g" />
            <NutritionRow label="Protein" value={nutrition.protein_g} unit="g" />
            <NutritionRow label="Fat" value={nutrition.fat_g} unit="g" />
            <NutritionRow label="Vitamin C" value={nutrition.vitamin_c_mg} unit="mg" />
          </div>
        ) : (
          <p className="shelf-life-sub">Nutrition data unavailable</p>
        )}
        {nutrition.available && nutrition.source_ref && (
          <p className="shelf-life-sub">
            Source: {nutrition.source_ref} (typical values per 100g)
          </p>
        )}
      </section>

      <div style={{ display: "flex", justifyContent: "center", paddingBottom: 8 }}>
        <button type="button" className="btn" onClick={onScanAgain}>
          Scan Again
        </button>
      </div>

      {imageUrl && <URLRevoker url={imageUrl} />}
    </div>
  );
}

/** Revoke the object URL when the captured image leaves the DOM. */
function URLRevoker({ url }) {
  useEffect(() => {
    return () => URL.revokeObjectURL(url);
  }, [url]);
  return null;
}

