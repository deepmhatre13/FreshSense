# Freshness Dataset (`docs/FRESHNESS_DATASET.md`)

## Datasets discovered in the repository

| Source | Fresh | Rotten | Notes |
|---|---|---|---|
| `data/Original Image/` (Mendeley) | 200/class | 200/class | 8 fruits × 200 fresh + 200 rotten = 3200 images, explicit labels, license CC BY 4.0 |
| `data/Quality Dataset/` | 181 | 178 | Tiny residual counts; NOT treated as sufficient for training |
| `data/raw/dataset/dataset/` (Kaggle Fresh & Rotten) | 2088/1962/1854 (apple/banana/orange fresh) | 2943/2754/1750 (rotten) | Supplementary, larger counts for apple/banana/orange |

The canonical freshness dataset used for training is derived **only** from
`data/Original Image/` (8 fruits × 200 fresh + 200 rotten = 3200 images).

## Fruits available (valid fresh AND rotten labels)

Apple, Banana, Grape, Guava, Jujube, Orange, Pomegranate, Strawberry
(16 explicit fresh/rotten classes).

## Fruits unavailable

Kiwi, Mango, Cherry, Chickoo — no valid fresh training data in the repository.
The tiny Quality-Dataset residuals are **not** treated as sufficient training
data.

## Canonical dataset structure

```
data/freshness/
  train/   Apple_fresh/ Apple_rotten/ banana_fresh/ ...   (2240 images)
  valid/   same 16 classes                                (480 images)
  test/    same 16 classes                                (480 images)
  class_mapping.json
  metadata.json
  dataset_manifest.json
```

Per-class split: **140 / 30 / 30** (train / valid / test), i.e. a 70/15/15
class-stratified deterministic split.

Original source data (`data/Original Image/`, the raw Kaggle dataset) is never
overwritten.

## Data quality

Before training the builder:

1. Removes zero-byte / corrupt images.
2. SHA256-deduplicates exact duplicates (audit: 0 non-unique source files).
3. Applies perceptual-hash near-duplicate detection.
4. Prevents the same physical/source image from appearing across
   train/valid/test (zero cross-split collision via per-image identity).
5. Preserves source provenance and license (CC BY 4.0).
6. Records rejected samples and per-class counts in `dataset_manifest.json`.

File-level random splitting is NOT used because it creates leakage; the split
is per-physical-image stratified.

## Labels

Explicit directory labels (`FreshApple`, `RottenApple`, etc.). Labels are valid
for freshness training. No labels are fabricated.

## Leakage

The split key is the physical source image (SHA256 + perceptual hash), so the
same physical/derived image never crosses train/valid/test. Leakage checks pass.