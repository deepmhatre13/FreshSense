# Phase 5 Dataset Audit — FreshSense AI

**Date:** 2026-08-09
**Scope:** Dataset infrastructure only. Findings are factual observations of the
repository at commit `13b7f2a` ("feat: prepare FreshSense AI phase 1-4 release").
No code was modified to produce this document.

---

## 1. What dataset format the current training pipeline expects

The training pipeline (via `src/preprocessing/dataset.py` →
`FreshSenseDatasetLoader`) accepts a folder tree of images:

- **Layout A (flat):** `data/raw/<class>/*.jpg`
- **Layout B (split):** `data/raw/{train,val,test}/<class>/*.jpg`
- **Layout C (train/test only):** `data/raw/{train,test}/<class>/*.jpg`
  (validation is derived from a file-level stratified `train_test_split`)
- **Layout D (nested):** `data/raw/<any>/.../{train,test}/<class>/*.jpg`
  (the loader descends up to 5 levels looking for a `train`/`test` split or
  class folders containing images directly)

Class labels are the **leaf folder names**. Class names are detected from the
leaf class folders, not intermediate container folders.

### Actual on-disk layout

The repository ships a nested variant of Layout C:

```
data/raw/dataset/dataset/
├── train/
│   ├── freshapples/    (1693 images)
│   ├── freshbanana/    (1581)
│   ├── freshoranges/   (1466)
│   ├── rottenapples/   (2342)
│   ├── rottenbanana/   (2224)
│   └── rottenoranges/  (1347)
└── test/
    ├── freshapples/    (395)
    ├── freshbanana/    (381)
    ├── freshoranges/   (388)
    ├── rottenapples/   (601)
    ├── rottenbanana/   (530)
    └── rottenoranges/  (403)
```

`_resolve_dataset_root("data/raw")` walks to `data/raw/dataset/dataset/`,
detects the `{train,test}` split folders and uses them as pre-existing splits.

- Train images: **10,653**
---

## 2. What classes currently exist

Six classes, named as `fruit-type + freshness` concatenations:

| Folder          | Interpretation        |
|-----------------|-----------------------|
| `freshapples`   | fresh apples          |
| `freshbanana`   | fresh bananas         |
| `freshoranges`  | fresh oranges         |
| `rottenapples`  | rotten apples         |
| `rottenbanana`  | rotten bananas        |
| `rottenoranges` | rotten oranges        |

`configs/settings.yaml` / `configs/config.py` do **not** hard-code a class
list; classes are discovered from the folder tree at scan time.

---

## 3. What the labels represent

**Fruit + freshness combination.** Each class folder name encodes both the
fruit type (`apples` / `banana` / `oranges`) and the freshness tier
(`fresh` / `rotten`). There is no independent `fruit_type` and
`freshness_label` split in the current on-disk dataset.

By contrast, the Phase 4 collection tool (`src/data/collection.py`) records a
single free-text `label` field (documented as `fresh` / `stale` / `rotten` /
---

## 4. Whether the current evaluation code actually evaluates the intended classes

**No — the committed evaluation artifacts do not evaluate the six intended classes.**

- `models/metrics/evaluation_metrics.json` (the committed "100%" result)
  reports `num_samples: 1436`, `class_names == ["dataset"]`, and a 1×1
  confusion matrix with accuracy/precision/recall/F1 all `1.0`.
  A single-class evaluation is **not** evidence of 6-class freshness
  performance. The 100% figure is an artifact of a loader/class-resolution
  mistake at the time it was produced (the class discovered was a container
  folder literally named `dataset`).
- `models/metrics/confusion_matrix.png`, `misclassified.png`, `roc_curves.png`,
  `pr_curves.png`, `confidence_distribution.png` were generated from that same
  single-class run and are equally non-informative.
- The trained checkpoint `models/checkpoints/best_model.pth` **is** a real
  6-class model:
  - `num_classes = 6`
  - `class_names = ['freshapples', 'freshbanana', 'freshoranges',
    'rottenapples', 'rottenbanana', 'rottenoranges']`
  - trained on `/content` (Google Colab), `val_acc = 99.705`
  - its training history (`models/checkpoints/training_history.csv`) records
    100.0/100.0 train/val accuracy for every epoch — i.e. the *training*
    harness itself shows the same single-class pathology (loss 0.0, acc
    100.0 from epoch 1), so the checkpoint history is **not** trustworthy as
    6-class evidence either.

**Conclusion:** the code is capable of evaluating the six intended classes
when pointed at the correct split layout, but every committed artifact was
produced from a broken single-class run. A fresh baseline evaluation against
real six-class data is required (Step 5 of Phase 5).

---

## 5. Whether train/validation/test leakage can occur

**Yes — leakage is possible and currently unverifiable.**

1. `FreshSenseDatasetLoader._collect` includes every image file found in the
   on-disk train/test folders. It has **no concept of physical fruit identity**,
   so if the same physical fruit appears in both `train/` and `test/`, nothing
   in the pipeline detects or prevents it.
2. When no on-disk `val/` folder exists, validation is derived from `train/`
   via a **file-level stratified** `train_test_split`. Files from the same
   physical fruit (or from the same capture session) can freely land on both
   sides of the train/val boundary.
3. The legacy real-world collection metadata (`metadata/sample_*.json`) groups
   frames by `session_id` only. The tooling explicitly notes "Physical fruit
---

## 7. Related infrastructure inventory (facts)

| Component | Path | Notes |
|---|---|---|
| Dataset loader | `src/preprocessing/dataset.py` | Layouts A–D; file-level stratified val |
| Classifier model | `src/models/efficientnet.py` | EfficientNet-B0, 6-class head in checkpoint |
| Evaluator | `src/training/evaluate.py` | accuracy, weighted P/R/F1, per-class metrics |
| Metrics helpers | `src/utils/metrics.py` | top-k acc, per-class acc, ROC/PR AUC |
| Dataset validation toolkit | `src/data/dataset_validation.py` | scan, MD5, pHash, duplicates, session split |
| Collection | `src/data/collection.py` | webcam + metadata JSON (`session_id`, no fruit id) |
| Validation script | `scripts/validate_real_world_dataset.py` | text report; human-readable only |
| Leakage script | `scripts/check_dataset_leakage.py` | hash-based duplicate/cross-split check |
| Analysis script | `scripts/analyze_real_world_data.py` | metadata stats + Markdown report |
| Baseline evaluation script | `scripts/baseline_evaluation.py` | calls `Evaluator` on raw test split |
| Config | `configs/config.py`, `configs/settings.yaml` | no class list; 70/15/15 ratios; seed 42 |
| Tests | `tests/` (19 files) | 175 passing as of this audit |
| CI | `.github/workflows/python.yml` | syntax + import checks **only**; does not run pytest |
| Report policy | `.gitignore` | ignores `/data/`, `models/metrics/`, `benchmark_results/`; `reports/` is **not** ignored |
| Real-world data | `data/real_world/` | contains only a `README.md`; no images/metadata yet |

### Evaluator capabilities vs Phase 5 requirements

`src/training/evaluate.py` already computes and can report:
- total samples, accuracy, weighted precision, weighted recall, weighted F1,
  per-class precision/recall/F1, per-class accuracy, confusion matrix,
  ROC/PR AUC.

It does **not** currently report as distinct fields:
- **macro precision / macro recall / macro F1**
- **balanced accuracy**
- **class distribution** (it knows `class_names` but not per-class test counts)

Those will be added by the Phase 5 baseline evaluator (Step 5).

---

## 8. Summary of concrete gaps (factual)

1. No canonical dataset schema or manifest exists (no `physical_fruit_id`,
   `fruit_type`, `freshness_label`, camera/session fields).
2. No physical-fruit-grouped splitter exists; validation split is file-level.
3. Leakage detection covers file hashes only, not physical-fruit identity.
4. Committed 100% evaluation artifacts are single-class and must not be cited
   as six-class evidence.
5. No machine-readable `reports/dataset_validation.json` output exists.
6. The baseline evaluator does not emit macro metrics / balanced accuracy /
   class distribution, and there is no "NOT AVAILABLE" mode.
7. No integration between the legacy collection `session_id` metadata and a
   canonical `physical_fruit_id` split.
8. CI does not execute the test suite.
   identity is NOT recorded" (`scripts/validate_real_world_dataset.py` line
   ~285). Two sessions may contain the same specimen, so `session_id`
   grouping is a weaker proxy than `physical_fruit_id`.
4. `scripts/check_dataset_leakage.py` detects exact/near-duplicate images by
   file hash but cannot detect semantic leakage of the *same physical object*
   captured under different angles/lighting.

**No grouped-by-physical-fruit splitting exists anywhere in the repository.**

---

## 6. Whether the existing real-world collection infrastructure can store the metadata needed for grouped splitting

**Partially — with a required change in the metadata schema.**

- `CollectedSample` (in `src/data/collection.py`) already stores:
  `sample_id`, `session_id`, `timestamp`, `image_path`, `label`, quality
  metrics, resolution, `source_information`, `tracking_id`.
- It does **not** store `physical_fruit_id`, `fruit_type`, or the camera /
  lighting / background / angle / occlusion / distance / storage /
  annotator fields required by the Phase 5 canonical schema.
- Two design gaps must be closed to support grouped splitting:
  1. **Physical fruit identity** must be recorded per sample (the collection
     UI currently has no "which fruit is this?" prompt).
  2. The current `label` conflates fruit type and freshness. Phase 5 splits
     these into `fruit_type` + `freshness_label`, with the 6-class model class
     derived as `freshness_label + fruit_type` (e.g. `freshapples`).

The metadata sidecar directory structure (`accepted/`, `rejected/`,
`metadata/`) and the JSON metadata-per-sample pattern are suitable to extend;
they do not need to be redesigned.
`uncertain` / `unlabeled`) per sample — fruit type is not recorded separately
there either.
- Test images: **2,698**
- Total: **13,351**, across **6 class folders**.