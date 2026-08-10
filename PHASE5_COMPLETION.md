# Phase 5 Completion — Real-World Dataset & Evaluation Foundation

**Date:** 2026-08-09
**Status:** Evaluation foundation complete. No model improvement performed
(stop condition honored: baseline groundwork only).

---

## 1. What Changed

### Files created

| File | Purpose |
|---|---|
| `PHASE5_DATASET_AUDIT.md` | Factual audit of the pre-existing dataset infrastructure (Step 1). |
| `docs/REAL_WORLD_DATASET.md` | Canonical dataset schema (IMAGE ID vs PHYSICAL FRUIT ID vs CAPTURE SESSION ID), manifest format, collection guidance, validation rules (Step 2). |
| `docs/EVALUATION_WORKFLOW.md` | Single documented, reproducible command recipe (Step 6). |
| `src/data/real_world_schema.py` | Schema definition, manifest loading (CSV/JSON), manifest validation, physical-fruit + session leakage detection, deterministic class-balanced grouped splitter (Step 3/4 core). |
| `scripts/create_dataset_split.py` | CLI splitter. Fails loudly on leakage; writes `train.csv`/`val.csv`/`test.csv` + `split_report.json` (Step 3). |
| `tests/test_real_world_schema.py` | 31 tests: manifest load, metadata validation, invalid labels, duplicates, impossible combinations, class imbalance, leakage detection, grouped splitting, determinism. |
| `tests/test_baseline_evaluation.py` | 12 tests: evaluator output fields, macro/weighted/balanced metrics, confusion matrix, `NOT_AVAILABLE` mode, dataset discovery. |
| `tests/test_phase5_scripts.py` | 4 integration tests: splitter CLI success/abort, validation script writes `dataset_validation.json`/`.md`. |

### Files modified

| File | Change |
|---|---|
| `scripts/validate_real_world_dataset.py` | Now runs the Phase 5 canonical-manifest section (labels, metadata, duplicates, impossible combinations, class imbalance, split-file leakage checks) and writes **`reports/dataset_validation.json`** and **`reports/dataset_validation.md`**. |
| `scripts/baseline_evaluation.py` | Rewritten to evaluate the **intended six freshness classes** and report total samples, class distribution, accuracy, macro precision/recall/F1, weighted F1, balanced accuracy, per-class metrics and confusion matrix. Reports `NOT_AVAILABLE` (never fake numbers) when no checkpoint/dataset is usable. |

### Generated reports (not committed)

- `reports/baseline_evaluation.json` — real evaluation on 2,698 labeled test images.
- `reports/dataset_validation.json` / `reports/dataset_validation.md` — current real-world data state (no benchmark data collected yet → `NOT READY`).

---

## 2. Tests Run

Command: `python -m pytest -q` on the full suite.

- **222 passed**
- **0 failed**
- **0 skipped**

(Previously 175 tests; **47 new Phase 5 tests** were added. No existing tests were weakened.)

---

## 3. Results

### Real baseline evaluation (genuine numbers, not fabricated)

Evaluated `models/checkpoints/best_model.pth` on the shipped labeled 6-class
test split (`data/raw/dataset/dataset/test`, all 2,698 images).

| Metric | Value |
|---|---|
| Total samples | 2698 |
| Classes | freshapples, freshbanana, freshoranges, rottenapples, rottenbanana, rottenoranges |
| Accuracy | 0.98369 |
| Macro precision | 0.98481 |
| Macro recall | 0.98292 |
| Macro F1 | 0.98376 |
| Weighted F1 | 0.98365 |
| Balanced accuracy | 0.98292 |
| Confusion matrix | 6×6 (rows=true, cols=predicted) |

Per-class F1: freshapples 0.986, freshbanana 0.997, freshoranges 0.982,
rottenapples 0.977, rottenbanana 0.994, rottenoranges 0.966.

Caveat recorded in the report JSON: the raw benchmark split has **no
`physical_fruit_id` metadata**, so grouped-split leakage cannot be verified
for it. It is real labeled data, but it is **not** the authoritative
canonical benchmark yet.

### The old "100%" claim is retired

`models/metrics/evaluation_metrics.json` reported 100% on a **single class
named `dataset`** (1×1 confusion matrix, 1436 samples). It is not evidence of
six-class performance and is explicitly labeled as invalid in
`PHASE5_DATASET_AUDIT.md`. The Phase 5 evaluator measures the real six
classes instead.

### Dataset / split / leakage status

- Current real-world data: **0 benchmark samples** (`data/real_world/`
  contains no images/manifest) → validation verdict **NOT READY FOR TRAINING**
  (`reports/dataset_validation.json`).
- Physical-fruit grouped splittings produced on fixtures and checked:
  every `physical_fruit_id` stays in exactly one split
  (`no_fruit_leakage = true`, `full_partition = true`).
- Leakage detector: reports empty on separated fixtures and flags any fruit /
  session that crosses splits (unit-verified).

---

## 4. Remaining Blockers
---

## 5. Exact Next Step

**Do not proceed to model improvement yet.** The next step is to build the
real benchmark data:

1. Collect labeled images of physical fruit specimens exactly as
   `docs/REAL_WORLD_DATASET.md` prescribes (record `physical_fruit_id`,
   `capture_session_id`, `image_id`, timestamps, camera, and the optional
   environment fields).
2. Write `data/real_world/manifest.csv` in the canonical schema.
3. Run, in order (documented in `docs/EVALUATION_WORKFLOW.md`):

   ```powershell
   .\venv\Scripts\python.exe scripts\validate_real_world_dataset.py
   .\venv\Scripts\python.exe scripts\create_dataset_split.py
   .\venv\Scripts\python.exe scripts\baseline_evaluation.py
   ```

   - `create_dataset_split.py` fails loudly if any `physical_fruit_id`
     crosses a split.
   - `validate_real_world_dataset.py` re-checks `physical_fruit_leakage: []`.
4. Evaluate the checkpoint (or a retrained model) against
   `--manifest data\real_world\splits\test.csv` and
   `--data-root data\real_world`, and record the metrics.

Only after the canonical benchmark data exists should the phase proceed to
model training/improvement (Phase 6), using the physically-grouped splits
produced by `scripts/create_dataset_split.py`.

---

## 6. Stop-Condition Checklist

| Condition | Met |
|---|---|
| 1. Dataset schema exists | ✅ `docs/REAL_WORLD_DATASET.md` |
| 2. Physical-fruit grouping exists | ✅ `src/data/real_world_schema.py` + `scripts/create_dataset_split.py` |
| 3. Leakage detection works | ✅ physical-fruit + session leakage checks, unit-tested |
| 4. Baseline evaluator measures correct classes | ✅ 6-class real evaluation (`reports/baseline_evaluation.json`) |
| 5. Tests pass | ✅ 222 passed, 0 failed, 0 skipped |
| 6. No fake metrics | ✅ `NOT_AVAILABLE` mode; single-class "100%" retired; all reported numbers computed on real labeled data |
| 7. Docs explain how to create the real benchmark | ✅ `docs/REAL_WORLD_DATASET.md` + `docs/EVALUATION_WORKFLOW.md` |

1. **No collected real-world benchmark yet.** `data/real_world/manifest.csv`
   does not exist. Every tool is ready and tested, but the authoritative
   numbers require real captured data with `physical_fruit_id` recorded.
2. **Raw benchmark lacks specimen identity.** The 2,698-image result above is
   from pre-existing splits with no `physical_fruit_id`, so its grouped-split
   leakage status is **unverified**. It is an indicative, not authoritative,
   baseline.
3. **Checkpoint training history is unreliable.** `training_history.csv`
   shows 100% from epoch 1 (single-class artifact), so the checkpoint is
   treated as a black-box baseline, not proven training evidence.