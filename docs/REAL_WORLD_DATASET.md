# Real-World Dataset — Canonical Schema (Phase 5)

This document defines the canonical format for all real-world benchmark data
used by FreshSense AI. It is the single source of truth for writing a
manifest, for grouped splitting, and for leakage-safe evaluation.

**Why a manifest?** Folder names alone cannot express physical fruit identity,
capture session identity, or the imaging conditions that matter for a
trustworthy benchmark. Every image in the real-world benchmark must therefore
have one row in a manifest that ties it to those facts.

---

## 1. Terminology — three IDs you must not confuse

| ID | Meaning | Cardinality | Example |
|---|---|---|---|
| **IMAGE ID** (`image_id`) | Uniquely identifies one *image file*. | 1 per row | `IMG_000123` |
| **PHYSICAL FRUIT ID** (`physical_fruit_id`) | Uniquely identifies one *physical piece of fruit* (specimen) across all its captures. | 1 per physical object; **many rows** | `FRUIT_A7` |
| **CAPTURE SESSION ID** (`capture_session_id`) | Uniquely identifies one *collection run* (one camera on at one time). | 1 per run; **many rows** | `SESS_2026-08-01_001` |

These are **not interchangeable**:

- Two images of the **same fruit** on day 1 and day 5 have different
  `image_id`, the same `physical_fruit_id`, and different `capture_session_id`.
- Two images of **different fruits** in the same session have different
  `image_id`, different `physical_fruit_id`, and the same `capture_session_id`.
- `physical_fruit_id` is what governs **splitting**. A specimen (and every
  image of it) must stay entirely inside one split (train / validation / test).

---

## 2. Canonical manifest fields

A manifest is **CSV** (`manifest.csv`, default) or **JSON**
(`manifest.json`, an array of objects). One row per image. Paths inside the
manifest are **relative to the dataset root** (the directory that contains the
manifest).

### 2.1 Required fields

| Field | Type | Description |
|---|---|---|
| `image_id` | string | Unique per-image identifier. |
| `image_path` | string | Relative path to the image file. |
| `fruit_type` | string | Fruit type. Canonical values: `apples`, `banana`, `oranges` (extensible). |
| `freshness_label` | string | Freshness tier. Canonical values: `fresh`, `stale`, `rotten` (extensible). |
| `physical_fruit_id` | string | Unique specimen identifier. **Determines splits.** |
| `capture_session_id` | string | Identifier of the capture run the image was taken in. |
| `capture_timestamp` | ISO-8601 datetime (e.g. `2026-08-01T14:03:22Z`) | When the image was captured. |
| `camera_id` | string | Identifier of the camera used (e.g. `webcam_builtin_1`). |

### 2.2 Optional but strongly recommended fields

| Field | Type | Description |
|---|---|---|
| `lighting_condition` | string | `natural`, `indoor_artificial`, `mixed`, `low_light` (extensible). |
| `background_type` | string | `plain`, `cluttered`, `hand`, `surface`, `other` (extensible). |
| `viewing_angle` | string | `front`, `side`, `top`, `angled`, `overhead` (extensible). |
| `occlusion_level` | float in `[0, 1]` | Fraction of the fruit occluded (0 = fully visible). |
| `distance_category` | string | `close`, `medium`, `far` (extensible). |
| `storage_condition` | string | e.g. `room_temp`, `fridge`, `counter`, `bag` (extensible). |
| `days_since_purchase` | non-negative number | Days between purchase/harvest and capture. |
| `annotator` | string | Who assigned the ground-truth freshness label. |
| `annotation_confidence` | float in `[0, 1]` | Annotator's confidence in the label. |

No optional field may block ingest: missing optional values are recorded as
`missing_optional` in validation and never prevent a row from being used —
they only reduce the analyses that can be performed for that row.

### 2.3 Derived values

- **Model class:** `freshness_label + fruit_type` (concatenated), e.g.
  `fresh + apples → freshapples`. The current 6-class checkpoint recognizes
  `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`,
  `rottenbanana`, `rottenoranges`.

### 2.4 Allowed values (canonical enums)

Validation treats these as canonical but **extensible**: unknown values are
flagged as warnings (`unknown`), not rejected, so new fruits/freshness tiers
do not require a code change.

| Field | Canonical allowed values |
|---|---|
| `fruit_type` | `apples`, `banana`, `oranges` |
| `freshness_label` | `fresh`, `stale`, `rotten` |
| `lighting_condition` | `natural`, `indoor_artificial`, `mixed`, `low_light` |
| `background_type` | `plain`, `cluttered`, `hand`, `surface`, `other` |
| `viewing_angle` | `front`, `side`, `top`, `angled`, `overhead` |
| `distance_category` | `close`, `medium`, `far` |
| `storage_condition` | `room_temp`, `fridge`, `counter`, `bag`, `other` |

---

## 3. On-disk layout

```
data/real_world/
├── manifest.csv            # canonical manifest (required for splitting)
├── images/                 # or accepted/ for collector output
│   └── <image_id>.<ext>
├── metadata/               # legacy collector JSON (kept for backwards compat)
├── splits/                 # generated by scripts/create_dataset_split.py
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── README.md
```
---

## 4. Example manifest (2 physical fruits, 4 images)

```csv
image_id,image_path,fruit_type,freshness_label,physical_fruit_id,capture_session_id,capture_timestamp,camera_id,lighting_condition,background_type,viewing_angle,occlusion_level,distance_category,storage_condition,days_since_purchase,annotator,annotation_confidence
IMG_0001,images/IMG_0001.jpg,apples,fresh,FRUIT_A7,SESS_2026-08-01_001,2026-08-01T09:12:00Z,webcam_1,natural,plain,front,0.0,medium,room_temp,0,alice,1.0
IMG_0002,images/IMG_0002.jpg,apples,fresh,FRUIT_A7,SESS_2026-08-01_001,2026-08-01T09:12:03Z,webcam_1,natural,plain,side,0.0,medium,room_temp,0,alice,1.0
IMG_0003,images/IMG_0003.jpg,apples,stale,FRUIT_A7,SESS_2026-08-05_002,2026-08-05T10:00:00Z,webcam_1,indoor_artificial,hand,front,0.1,close,room_temp,4,alice,0.9
IMG_0004,images/IMG_0004.jpg,banana,fresh,FRUIT_B2,SESS_2026-08-05_002,2026-08-05T10:02:00Z,webcam_1,indoor_artificial,hand,front,0.0,close,room_temp,1,bob,1.0
```

Notes about this example:

- `FRUIT_A7` appears on two days (two sessions) with different
  `freshness_label` (`fresh` → `stale`). This is **expected** for aging
  studies: freshness may change over time for the same specimen.
- `fruit_type` for one specimen never changes; a row whose
  `fruit_type` contradicts the majority `fruit_type` of its
  `physical_fruit_id` is flagged as an impossible metadata combination.
- Splitting groups all four rows by `physical_fruit_id`, so `IMG_0001`,
  `IMG_0002`, `IMG_0003` always travel together, and `IMG_0004` independently.

---

## 5. Collection guidance (what is reasonable to collect)

| Metadata | How to record it |
|---|---|
| `physical_fruit_id` | Label each physical specimen once (e.g. a sticker `A`, `B`, ..., or `FRUIT_<n>`); reuse it across every capture of that piece. |
| `capture_session_id` | Auto-generate once per collection run (the collector already does this as `session_id`). |
| `capture_timestamp` | Auto-recorded by the capture tool. |
| `camera_id` | Fixed string per camera in the collection config. |
| `lighting_condition`, `background_type`, `viewing_angle`, `distance_category` | One prompt per capture batch in the collection UI; if a strict protocol is followed, apply a batch default and let the operator override per frame. |
| `occlusion_level` | Estimate in steps (0.0, 0.25, 0.5, 0.75); not required to be precise. |
| `storage_condition`, `days_since_purchase` | Record once per session for multi-day aging studies. |
| `annotator`, `annotation_confidence` | Operator identifier + a simple 0.5/0.75/1.0 confidence pick per label. |

Do **not** block collection on optional fields. Whatever is unknown should be
left blank in the manifest; the validator will report it as missing-optional
and the row will still be usable.

---

## 6. Validation rules the manifest must satisfy

1. All required fields present for every row (missing → `missing_labels` /
   `missing_metadata` findings).
2. `image_path` exists on disk relative to the manifest's directory.
3. No two rows share the same `image_id`.
4. No two distinct paths are byte-identical files (MD5) → exact duplicates.
5. Images that are perceptually near-duplicates are reported (pHash distance).
6. `fruit_type` / `freshness_label` memberships in the canonical enums
   (warn-only for extensions).
7. Numeric bounds: `0 <= occlusion_level <= 1`,
   `0 <= annotation_confidence <= 1`, `days_since_purchase >= 0`.
8. `capture_timestamp` parses as ISO-8601.
9. `physical_fruit_id` has a consistent `fruit_type` across all its rows
   (freshness may change; fruit type may not).
10. In an **existing split manifest**, no `physical_fruit_id` appears in more
    than one split; if the same `capture_session_id` appears in more than one
    split that is reported as session leakage (advisory).

---

## 7. The one rule that makes the benchmark trustworthy

> **Images belonging to the same physical fruit MUST NEVER appear across
> train/validation/test.**

The splitter (`scripts/create_dataset_split.py`) enforces this by grouping
rows on `physical_fruit_id`. Validation (`scripts/validate_real_world_dataset.py`)
re-checks it on any already-produced split files.