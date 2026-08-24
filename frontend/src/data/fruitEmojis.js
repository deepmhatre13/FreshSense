/**
 * Emoji glyphs for the 10 detected fruit classes.
 *
 * Keyed by the lower-cased fruit identity (the canonical form the rest of the
 * app normalizes YOLO labels to). The lookup is case-insensitive so both
 * "Apple" and "apple" map to the same glyph.
 */
export const FRUIT_EMOJIS = {
  apple: "🍎",
  grapes: "🍇",
  grape: "🍇",
  kiwi: "🥝",
  mango: "🥭",
  orange: "🍊",
  "sweet orange": "🍊",
  strawberry: "🍓",
  banana: "🍌",
  cherries: "🍒",
  cherry: "🍒",
  chickoo: "🟤",
  sapodilla: "🟤",
  guava: "🫐",
  // Fallback handled in the caller, not here.
};

export function emojiFor(fruit) {
  if (!fruit) return "🍎";
  return FRUIT_EMOJIS[String(fruit).trim().toLowerCase()] || "🍎";
}
