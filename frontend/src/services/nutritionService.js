/**
 * Nutrition data access.
 *
 * Nutrition values come purely from the static database
 * (src/data/nutrition.json) keyed by detected fruit identity. They are NOT
 * inferred from image pixels. If a fruit is missing from the database we
 * surface "unavailable" rather than inventing numbers.
 */
import nutritionData from "../data/nutrition.json";

const byFruit = nutritionData.fruits || {};

/** Normalize a detected fruit label to the database key (lowercase). */
function keyFor(label) {
  if (!label) return "";
  return String(label).trim().toLowerCase();
}

export function getNutrition(fruit) {
  const key = keyFor(fruit);
  const entry = byFruit[key];
  if (!entry) {
    // Explicit "missing" — the UI must NOT fabricate nutrition values here.
    return { available: false, fruit: key, source: nutritionData.source };
  }
  return {
    available: true,
    fruit: key,
    scientific_name: entry.scientific_name,
    calories_kcal: entry.calories_kcal,
    carbohydrates_g: entry.carbohydrates_g,
    fiber_g: entry.fiber_g,
    protein_g: entry.protein_g,
    fat_g: entry.fat_g,
    vitamin_c_mg: entry.vitamin_c_mg,
    form: entry.form,
    source: nutritionData.source,
    source_ref: entry.source_ref,
  };
}

/** For diagnostics/testing: list every fruit covered by the nutrition DB. */
export function listNutritionFruits() {
  return Object.keys(byFruit);
}
