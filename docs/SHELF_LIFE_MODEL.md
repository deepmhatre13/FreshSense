# Shelf-Life (`docs/SHELF_LIFE_MODEL.md`)

## What this is

The shelf-life output is a **deterministic baseline**, not an ML time-to-event
model. There is no shelf-life ground-truth dataset in the repository, so no
validated ML shelf-life model exists and none is faked.

> **This is a deterministic estimate based on fruit metadata and freshness
> state/confidence, not a validated time-to-event model.**

> **Shelf-life is not estimated when freshness data is unavailable or
> uncertain.**

## Shelf-life depends on freshness availability

| Freshness class | Shelf-life behaviour |
|---|---|
| `fresh`   | estimated: `remaining_days` set (confidence-scaled heuristic) |
| `rotten`  | `remaining_days = 0`, status `expired` |
| `stale`   | `remaining_days = 0`, status `expired` |
| `uncertain` | `remaining_days = null`, status `uncertain` |
| `data_not_available` | `remaining_days = null`, status `data_not_available` |

For any fruit without a validated freshness model, shelf-life is NOT estimated —
even if `typical_shelf_life_days` exists in `fruit_database.json`. The metadata
may still be shown as informational context (e.g. "Typical range: 5–14 days"),
but the application states:

> "Shelf-life estimate unavailable because freshness data is not available."

## No synthetic / fabricated shelf-life ground truth

XGBoost or any learned shelf-life model is **not** implemented, because no real
shelf-life ground-truth labels exist. The deterministic baseline is clearly
labelled as such in code (`ShelfLifeEstimator`, `BASIS_HEURISTIC`) and is never
called an ML prediction.

## Basis identifiers

- `fruit_typical_range + freshness_state + freshness_confidence` — heuristic for
  a supported fruit with a valid freshness prediction.
- `freshness_model_unsupported_or_uncertain` — freshness unavailable or
  uncertain.
- `metadata_unavailable` — fruit has no metadata entry.
- `metadata_invalid` — fruit metadata range is invalid.

## Contract

- `remaining_days` is an integer only for `estimated` / `expired` states.
- `remaining_days = 0` occurs ONLY for `expired` (`rotten` / `stale`).
- `remaining_days = null` for `uncertain` and `data_not_available`.
- No fake remaining days are ever produced.

## Known limitations

- No temperature/humidity sensors and no measured storage-duration history.
- Storage condition is a caller-supplied assumption, not a measured value.
- Estimates are heuristic, confidence-scaled values, not probabilities.