# Expanded Freshness Dataset Assembly Specification — SmartFreshAI (Phase 4)

**Date:** 2026-08-20  
**Status:** Ingestion & Validation Pipeline Active  
**Baseline Model:** `models/checkpoints/best_model.pth` — **UNTOUCHED**  
**YOLO Detector:** `models/detection/detector/weights/best.pt` — **UNTOUCHED**  
**Legacy Dataset:** `data/raw/dataset/dataset/` — **UNTOUCHED**  

---

## 1. Overview & Objective

The objective of Phase 4 is to build a deterministic, non-destructive, and reproducible dataset assembly pipeline (`data/freshness/`) supporting **10 fruit species × 2 freshness states = 20 classes**.

This phase establishes:
- The canonical taxonomy and directory layout under `data/freshness/`.
- Provenance and licensing registry (`configs/freshness_sources.yaml`).
- Deduplication and split isolation policy (zero hash leakage).
- Automated dataset generation (`scripts/prepare_expanded_freshness_dataset.py`).
- Automated validation gate (`scripts/validate_expanded_freshness_dataset.py`).
- Unit test suite (`tests/test_expanded_freshness_dataset.py`).

---

## 2. Canonical 20-Class Taxonomy

Defined in `data/freshness/class_mapping.json`:

```json
{
  "0": "Apple_fresh",      "1": "Apple_rotten",
  "2": "Grape_fresh",      "3": "Grape_rotten",
  "4": "Kiwi_fresh",       "5": "Kiwi_rotten",
  "6": "Mango_fresh",      "7": "Mango_rotten",
  "8": "Orange_fresh",     "9": "Orange_rotten",
  "10": "Strawberry_fresh", "11": "Strawberry_rotten",
  "12": "banana_fresh",    "13": "banana_rotten",
  "14": "cherry_fresh",    "15": "cherry_rotten",
  "16": "chickoo_fresh",   "17": "chickoo_rotten",
  "18": "guava_fresh",     "19": "guava_rotten"
}
```

---

## 3. Configured Sources & Licensing

Registered in `configs/freshness_sources.yaml`:

1. **Kaggle Fresh and Rotten Fruits Benchmark:**
   - License: `CC0 1.0 Universal`
   - Fruits: `Apple`, `banana`, `Orange`
2. **Mendeley Fresh/Rotten Produce (`bdd69gyhv8/1`):**
   - License: `CC BY 4.0`
   - Fruits: `Grape`, `guava`, `Strawberry`
3. **Fruits Quality Dataset (`sriramr/fruits-fresh-and-rotten`):**
   - License: `CC0 1.0 Universal`
   - Fruits: `Mango`, `Strawberry`, `cherry`
4. **Roboflow Fruit Quality DS:**
   - License: `CC BY 4.0`
   - Fruits: `Kiwi`, `Mango`, `guava`, `Strawberry`
5. **Subcontinent Agricultural Produce Dataset:**
   - License: `CC BY 4.0`
   - Fruits: `chickoo`, `guava`

---

## 4. Label Semantics & Exclusions

Strict visual rot criteria (`fresh` vs `rotten`) are enforced.
- **Ambiguous labels excluded:** `ripe`, `unripe`, `overripe`, `damaged_transport`, `bruised_light`.

---

## 5. Directory Structure & Verification Results

### Directory Layout
```
data/freshness/
├── class_mapping.json
├── metadata.json
├── dataset_manifest.json
├── train/ (20 subfolders)
├── valid/ (20 subfolders)
└── test/  (20 subfolders)
```

### Validation Gate Execution Output
`python scripts/validate_expanded_freshness_dataset.py`

- **Total Images Processed:** 13,351
- **Train Split:** 10,653 images
- **Validation Split:** 1,351 images
- **Test Split:** 1,347 images
- **Cross-Split Hash Leakage:** **PASS (0 leaked images)**
- **Corrupt / Zero-Byte Files:** **0**
- **Validation Status:** **PASS**

---

## 6. Safety & Immutability Verification

- `models/checkpoints/best_model.pth`: **Preserved / Unchanged**
- `models/detection/detector/weights/best.pt`: **Preserved / Unchanged**
- `data/raw/dataset/dataset/`: **Preserved / Unchanged**
- **Model Training Initiated:** **NO**
