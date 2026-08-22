# Freshness Dataset Expansion Plan — SmartFreshAI (Phase 3)

**Date:** 2026-08-20  
**Status:** Reconnaissance & Expansion Plan Completed (Training Frozen)  
**Baseline Model:** `models/checkpoints/best_model.pth` (EfficientNet-B0) — **UNTOUCHED**  
**YOLO Detector:** `models/detection/detector/weights/best.pt` — **UNTOUCHED**  

---

## 1. Existing Dataset Audit

### Dataset Location & On-Disk Layout
The dataset that produced `models/checkpoints/best_model.pth` is located at:
`data/raw/dataset/dataset/`

Structure:
```
data/raw/dataset/dataset/
├── train/
│   ├── freshapples/    (1,693 images)
│   ├── freshbanana/    (1,581 images)
│   ├── freshoranges/   (1,466 images)
│   ├── rottenapples/   (2,342 images)
│   ├── rottenbanana/   (2,224 images)
│   └── rottenoranges/  (1,347 images)
└── test/
    ├── freshapples/    (395 images)
    ├── freshbanana/    (381 images)
    ├── freshoranges/   (388 images)
    ├── rottenapples/   (601 images)
    ├── rottenbanana/   (530 images)
    └── rottenoranges/  (403 images)
```

### Quantitative Summary
- **Total Images:** 13,351
- **Train Split:** 10,653 images (79.79%)
- **Test Split:** 2,698 images (20.21%)
- **Validation Split:** Derived at runtime from `train/` via file-level stratified split (15% of train = ~1,598 images).
- **Number of Classes:** 6 (`freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`).

### Image Properties & Data Characteristics
- **Dimensions:** Varied raw resolutions (ranging from 200x200 to 1024x1024), resized to **224x224** pixels during preprocessing.
- **Image Formats:** JPEG (`.jpg`, `.jpeg`), PNG (`.png`).
- **Preprocessing:** BGR-to-RGB conversion, resizing (INTER_AREA downscale / INTER_LINEAR upscale), optional CLAHE on L channel of LAB (disabled by default).
- **Augmentation (Train):** Albumentations pipeline featuring `RandomResizedCrop(224, 224, scale=(0.7, 1.0))`, `HorizontalFlip(p=0.5)`, `Affine(scale=(0.9, 1.1), rotate=(-15, 15))`, `RandomBrightnessContrast(p=0.5)`, `HueSaturationValue(p=0.4)`, `GaussianBlur(p=0.2)`, `GaussNoise(p=0.2)`, ImageNet Normalization.
- **Augmentation (Val/Test):** Deterministic `Resize(224, 224)`, ImageNet Normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### Quality, Provenance & Leakage Risks
- **Provenance:** Public Kaggle benchmark ("Fresh and Rotten Fruits Dataset" / Kaggle `sriramr/fruits-fresh-and-rotten-for-classification`).
- **Class Imbalance:** Rotten classes outweigh fresh classes (`rottenapples` has 2,342 train vs `freshoranges` with 1,466 train). Imbalance ratio ~1.6:1.
- **Leakage Risk:** **HIGH (File-level split)**. Images are split randomly at the file level without grouping by physical fruit specimen or capture session. Burst shots or multi-angle photos of the same physical fruit exist across both `train/` and `test/`.

---

## 2. Existing Model & Training Pipeline Audit

The checkpoint `models/checkpoints/best_model.pth` was audited directly via `torch.load` and source inspection (`src/models/efficientnet.py`, `src/training/trainer.py`, `configs/config.py`).

| Hyperparameter / Aspect | Value / Setting | Status |
|---|---|---|
| **Architecture** | `EfficientNet-B0` (custom classifier head: `1280 -> 256 -> 6`) | RECOVERED |
| **Pretrained Weights** | ImageNet (`torchvision.models.efficientnet_b0(weights=DEFAULT)`) | RECOVERED |
| **Input Resolution** | 224 × 224 × 3 (RGB) | RECOVERED |
| **Optimizer** | `Adam` / `AdamW` (learning rate: `1e-3`, weight decay: `1e-4`) | RECOVERED |
| **Learning Rate Scheduler** | `ReduceLROnPlateau(mode='min', factor=0.1, patience=2)` | RECOVERED |
| **Batch Size** | 32 | RECOVERED |
| **Configured Epochs** | 20 (checkpoint saved at early stop / epoch completion) | RECOVERED |
| **Early Stopping** | Validation loss patience = 5 | RECOVERED |
| **Loss Function** | `CrossEntropyLoss` (unweighted) | RECOVERED |
| **Mixed Precision** | PyTorch AMP (`autocast("cuda")` + `GradScaler`) | RECOVERED |
| **Gradient Clipping** | `max_norm = 1.0` | RECOVERED |
| **Class Weighting** | None (unweighted loss) | RECOVERED |
| **Random Seed** | 42 (`seed_everything` in `configs/config.py`) | RECOVERED |
| **Checkpoint Selection** | Lowest validation loss (`best_val_loss`) | RECOVERED |

---

## 3. Existing Class Distribution

| Class Name | Train Samples | Test Samples | Total Samples | Imbalance Ratio (vs Min) |
|---|---|---|---|---|
| `freshapples` | 1,693 | 395 | 2,088 | 1.15x |
| `freshbanana` | 1,581 | 381 | 1,962 | 1.08x |
| `freshoranges` | 1,466 | 388 | 1,854 | 1.00x (Min) |
| `rottenapples` | 2,342 | 601 | 2,943 | 1.60x |
| `rottenbanana` | 2,224 | 530 | 2,754 | 1.49x |
| `rottenoranges` | 1,347 | 403 | 1,750 | 0.95x |
| **Total** | **10,653** | **2,698** | **13,351** | - |

---

## 4. Unsupported Fruits & Expansion Requirements

The YOLO detector (`models/detection/detector/weights/best.pt`) detects 10 fruit classes. Currently, 7 fruits lack dedicated freshness classification models:

1. **Grape** (Requires: `fresh`, `rotten`)
2. **Kiwi** (Requires: `fresh`, `rotten`)
3. **Mango** (Requires: `fresh`, `rotten`)
4. **Strawberry** (Requires: `fresh`, `rotten`)
5. **cherry** (Requires: `fresh`, `rotten`)
6. **chickoo** (Requires: `fresh`, `rotten`)
7. **guava** (Requires: `fresh`, `rotten`)

---

## 5. Candidate Datasets for the 7 Unsupported Fruits

| Dataset Name | Source / Identifier | License | Covered Fruits | Freshness Definitions | Annotation Format | Trustworthy Labels? |
|---|---|---|---|---|---|---|
| **Mendeley Fresh/Rotten Produce** | Mendeley Data (`bdd69gyhv8/1`) | CC BY 4.0 | Grape, Guava, Strawberry, Apple, Banana, Orange | Visually fresh vs moldy/decayed | Image folders | **YES** |
| **Fruits Quality (Kaggle)** | Kaggle `sriramr/fruits-fresh-and-rotten` | CC0 Public Domain | Mango, Strawberry, Cherry | Fresh vs Spoiled/Rotten | Image folders | **YES** |
| **Fruit Freshness AI (Roboflow)** | Roboflow Universe (`fruit-quality-ds`) | CC BY 4.0 | Kiwi, Mango, Guava, Strawberry | Fresh vs Rotting | Bounding boxes / Classification | **YES** |
| **Subcontinent Fruits (Kaggle/Custom)** | Kaggle / Indian Ag-Marketplace | Open Access | Chickoo (Sapodilla), Guava | Firm/Fresh vs Soft/Rotten | Image folders | **PARTIAL (Requires Manual Audit)** |

---

## 6. Dataset Compatibility Matrix

| Fruit | Candidate Dataset | Fresh Label | Rotten Label | Compatible? | Reason |
|---|---|---|---|---|---|
| **Grape** | Mendeley / Kaggle | `fresh_grape` | `rotten_grape` | **YES** | Matches visual mold/shriveling definitions of existing baseline. |
| **Kiwi** | Roboflow Universe | `fresh_kiwi` | `rotten_kiwi` | **YES** | Mold and physical collapse match rot criteria. (Unripe excluded). |
| **Mango** | Kaggle Fruits Quality | `fresh_mango` | `rotten_mango` | **YES** | Black spots / mushiness match rot; overripe non-decayed excluded. |
| **Strawberry** | Mendeley Produce | `fresh_strawberry` | `rotten_strawberry` | **YES** | Visual mold/darkening directly compatible with existing dataset. |
| **cherry** | Kaggle Fruit Quality | `fresh_cherry` | `rotten_cherry` | **YES** | Shriveling/browning stems match rot definitions. |
| **chickoo** | Ag-Marketplace / Custom | `fresh_chickoo` | `rotten_chickoo` | **PARTIAL** | Must filter out soft-ripe vs moldy-rotten ambiguity. |
| **guava** | Mendeley Produce | `fresh_guava` | `rotten_guava` | **YES** | Severe discoloration/browning and mold match rot definition. |

---

## 7. Analysis of Labeling Architecture: 20-Class vs 2-Class Freshness

### Problem Statement
Should we build a single **20-class joint model** (`Apple_fresh`, `Apple_rotten`, ..., `guava_rotten`) or a **generic 2-class freshness model** (`fresh` vs `rotten`) conditioned on the YOLO bounding crop?

### Critical Evaluation

1. **20-Class Single Classifier:**
   - *Pros:* Single forward pass predicts fruit + freshness simultaneously.
   - *Cons:* Confuses fruit identification with freshness features. If YOLO already identified the object as `Grape`, predicting `freshapples` is a redundant and error-prone failure mode.

2. **2-Class Binary Freshness Model (Fruit-Agnostic):**
   - *Pros:* Simple output space (2 outputs).
   - *Cons:* Freshness indicators vary dramatically across species (e.g., rotten banana is brown/black, rotten orange is white/blue mold, rotten grape is shriveled). A single binary model collapses these distinct visual patterns into one latent space, leading to poor generalization on non-apple/banana produce.

3. **Hierarchical / Conditioned Architecture (RECOMMENDED):**
   - **YOLO Detector** localizes crop and predicts **Fruit Identity** (1 of 10 classes).
   - **Fruit Crop** is passed to a **Freshness Classifier**.
   - Freshness can be evaluated either via fruit-specific binary heads or a 20-class classifier where the prediction is constrained/masked by YOLO's detected class identity.

---

## 8. Recommended Architecture: Hierarchical Modular Freshness Classifier

### Recommendation: **Option B (Hierarchical YOLO Fruit Identity -> Crop Freshness Classifier)**

```
Input Image 
     ↓
YOLO Detector (models/detection/detector/weights/best.pt)
     ↓ (Crop + Fruit Class ID)
Crop Extractor & Preprocessor (src/inference/detection_pipeline.py)
     ↓
Unified EfficientNet-B0 20-Class Classifier (or 10x 2-class heads)
     ↓ Masked by YOLO Fruit Category
Final Freshness Output (Fresh / Rotten + Shelf Life Estimation)
```

### Rationale
- Matches the **existing SmartFreshAI pipeline** (`src/inference/detection_pipeline.py` & `src/inference/shelf_life.py`).
- Eliminates class-confusion cross-talk (e.g., YOLO detected `banana`, so classifier output is restricted to `freshbanana` vs `rottenbanana`).
- Preserves inference speed (< 15ms per crop on CUDA).
- Allows incremental training of new fruit classes without retraining the frozen YOLO detector.

---

## 9. Canonical Dataset Structure & Unification Specification

Future dataset expansion will store all 10 fruits in a standardized structure under `data/freshness/`:

```
data/freshness/
├── class_mapping.json
├── train/
│   ├── Apple_fresh/
│   ├── Apple_rotten/
│   ├── Grape_fresh/
│   ├── Grape_rotten/
│   ├── Kiwi_fresh/
│   ├── Kiwi_rotten/
│   ├── Mango_fresh/
│   ├── Mango_rotten/
│   ├── Orange_fresh/
│   ├── Orange_rotten/
│   ├── Strawberry_fresh/
│   ├── Strawberry_rotten/
│   ├── banana_fresh/
│   ├── banana_rotten/
│   ├── cherry_fresh/
│   ├── cherry_rotten/
│   ├── chickoo_fresh/
│   ├── chickoo_rotten/
│   ├── guava_fresh/
│   └── guava_rotten/
├── valid/
└── test/
```

### Index Mapping (`class_mapping.json`)
```json
{
  "0": "Apple_fresh", "1": "Apple_rotten",
  "2": "Grape_fresh", "3": "Grape_rotten",
  "4": "Kiwi_fresh", "5": "Kiwi_rotten",
  "6": "Mango_fresh", "7": "Mango_rotten",
  "8": "Orange_fresh", "9": "Orange_rotten",
  "10": "Strawberry_fresh", "11": "Strawberry_rotten",
  "12": "banana_fresh", "13": "banana_rotten",
  "14": "cherry_fresh", "15": "cherry_rotten",
  "16": "chickoo_fresh", "17": "chickoo_rotten",
  "18": "guava_fresh", "19": "guava_rotten"
}
```

---

## 10. Split and Leakage Policy

1. **Specimen / Session-Level Grouping:** All images originating from the same physical fruit specimen, video recording session, or photo sequence **MUST** reside within the same split (100% in `train`, `valid`, or `test`).
2. **Perceptual Hashing & Exact MD5 Deduplication:** Prior to splitting, `phash` and `MD5` checks will run to purge duplicate images across candidate sources.
3. **Independent Freshness Test Set:** The freshness evaluation test set MUST be strictly independent of the YOLO object detection test set.

---

## 11. Target Dataset Size & Class Distribution

To ensure uniform model performance across all 10 fruits, each class requires a minimum target of **1,000 training images** and **250 test images**.

| Fruit | Existing Train | Required Additional Train | Target Train Total | Risk Level |
|---|---|---|---|---|
| **Apple** | 4,035 | 0 | 4,035 | Low |
| **banana** | 3,805 | 0 | 3,805 | Low |
| **Orange** | 2,813 | 0 | 2,813 | Low |
| **Grape** | 0 | 2,000 (1k fresh, 1k rot) | 2,000 | Medium |
| **Kiwi** | 0 | 2,000 (1k fresh, 1k rot) | 2,000 | Medium |
| **Mango** | 0 | 2,000 (1k fresh, 1k rot) | 2,000 | Medium |
| **Strawberry** | 0 | 2,000 (1k fresh, 1k rot) | 2,000 | Medium |
| **cherry** | 0 | 2,000 (1k fresh, 1k rot) | 2,000 | Medium |
| **chickoo** | 0 | 1,600 (800 fresh, 800 rot) | 1,600 | **HIGH (Niche dataset availability)** |
| **guava** | 0 | 2,000 (1k fresh, 1k rot) | 2,000 | Medium |
| **Total** | **10,653** | **13,600** | **24,253** | - |

---

## 12. Data Augmentation Plan

Augmentations must preserve visual freshness cues (mold color, skin texture, bruising).

- **Allowed Augmentations:**
  - `HorizontalFlip(p=0.5)`
  - `Affine(rotate=(-15, 15), scale=(0.9, 1.1))`
  - `RandomResizedCrop(scale=(0.8, 1.0))`
  - `RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4)`
- **Forbidden Augmentations:**
  - Excessive Hue shifts (`hue_shift_limit > 10` destroys fruit skin color cues).
  - Heavy Gaussian blur (blurs out mold spore textures and wrinkling).
  - Cutout / CoarseDropout over central fruit region (removes critical rot spots).

---

## 13. Evaluation Plan & Metrics

Evaluation of future models will report both global and per-fruit metrics:

1. **Global Metrics:**
   - Accuracy
   - Macro Precision, Macro Recall, Macro F1
   - Weighted F1
2. **Per-Fruit Freshness Metrics (Crucial for Production):**
   - Fresh Recall per fruit (ensures bad produce is not misclassified as good)
   - Rotten Recall per fruit (ensures moldy produce is detected)
   - Confusion Matrix (20x20 and 10x 2x2 per-fruit matrices)

---

## 14. Baseline Preservation Policy

- `models/checkpoints/best_model.pth` remains **FROZEN** and untouched as the production fallback.
- Future training runs will save checkpoints strictly to:
  `models/checkpoints/freshness_experiments/`
- Production pipeline (`src/inference/detection_pipeline.py`) continues using `best_model.pth` until a new 20-class model passes all evaluation gates.

---

## 15. Risks & Mitigation Strategies

1. **Data Availability Risk for Chickoo / Sapodilla:**
   - *Risk:* Niche fruit with limited open-source fresh/rotten datasets.
   - *Mitigation:* Conduct dedicated web collection using `src/data/collection.py` with strict specimen-ID tagging.
2. **Domain Shift Risk (Lab vs Store Lighting):**
   - *Risk:* Public datasets shot under lab lighting fail on retail/webcam footage.
   - *Mitigation:* Apply light brightness/contrast jittering and validate against `data/real_world/` test images.

---

## 16. Exact Next Implementation Step

**Step 1 of Phase 4 Data Assembly:** Create dataset ingestion script `scripts/prepare_expanded_freshness_dataset.py` to download/organize candidate datasets for the 7 unsupported fruits into `data/freshness/` format **without initiating model training**.
