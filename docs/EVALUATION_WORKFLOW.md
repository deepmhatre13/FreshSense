# Evaluation Workflow (Phase 5) — Reproducible Commands

Single source of truth for running the Phase 5 real-world benchmark pipeline.

All commands run from the repository root with the project virtualenv:

```powershell
cd d:\SmartFreshAI
.\venv\Scripts\python.exe --version
```

The canonical commands are:

```powershell
# 1. Validate the dataset (images + canonical manifest + any existing splits)
.\venv\Scripts\python.exe scripts\validate_real_world_dataset.py

# 2. Create the physical-fruit-grouped splits (train / val / test)
.\venv\Scripts\python.exe scripts\create_dataset_split.py

# 3. Evaluate the checkpoint on the real benchmark test split
.\venv\Scripts\python.exe scripts\baseline_evaluation.py

# 4. Extract metrics from the machine-readable reports
.\venv\Scripts\python.exe scripts\analyze_real_world_data.py
```

## What each command produces

| Command | Input | Output |
|---|---|---|
| `scripts/validate_real_world_dataset.py` | `data/real_world/` (images + `manifest.csv` + optional `splits/`) | `reports/dataset_validation.json` (machine-readable), `reports/dataset_validation.md` (human-readable), `reports/REAL_WORLD_DATASET_REPORT.md` (legacy) |
| `scripts/create_dataset_split.py` | `data/real_world/manifest.csv` | `data/real_world/splits/{train,val,test}.csv` + `split_report.json` |
| `scripts/baseline_evaluation.py` | checkpoint + canonical `splits/test.csv` (or a class-folder test set) | `reports/baseline_evaluation.json` + stdout summary |
| `scripts/analyze_real_world_data.py` | `data/real_world/metadata/` (legacy collector JSON) | `reports/real_world_dataset_report.md` |

## Required inputs (no real data yet)

Currently `data/real_world/` contains **no benchmark images or manifest**, so:

1. `scripts/validate_real_world_dataset.py` reports
   `Verdict: NOT READY FOR TRAINING` and `manifest_present: false`.
2. `scripts/create_dataset_split.py` aborts with
   `ERROR: manifest not found: data/real_world/manifest.csv`.
3. `scripts/baseline_evaluation.py` reports metrics `NOT_AVAILABLE` when no
   checkpoint or dataset is usable.

## Evaluating against the shipped raw benchmark while building the canonical set

The repository ships a real labeled 6-class benchmark test split at
`data/raw/dataset/dataset/test`. It is real data but **lacks
`physical_fruit_id` metadata**, so grouped-split leakage cannot be verified.
An interim evaluation can be produced with:

```powershell
.\venv\Scripts\python.exe scripts\baseline_evaluation.py `
  --data-root data\raw\dataset\dataset\test `
  --checkpoint models\checkpoints\best_model.pth
```

Limit inference time on CPU with `--max-images N` (each image is roughly
0.2–0.3 s on CPU). The result JSON is always explicit that this partial/full
run is on the raw benchmark, not the canonical real-world set.

## Building the real benchmark (reproducible recipe)

1. Collect labeled images per `docs/REAL_WORLD_DATASET.md` (specimen,
   session, image IDs and metadata).
2. Write `data/real_world/manifest.csv` in the canonical schema.
3. Run validation (command 1) until `VERDICT: READY FOR TRAINING`.
4. Run the splitter (command 2); it fails loudly if any `physical_fruit_id`
   spans splits.
5. Re-run validation (command 1) to confirm `physical_fruit_leakage: []`.
6. Run the baseline evaluator (command 3) pointed at
   `--manifest data\real_world\splits\test.csv` and
   `--data-root data\real_world`.
7. Record the reported accuracy / macro-metrics / per-class metrics /
   confusion matrix in the experiment log.

## Determinism

- Splitting uses a fixed seed (default `42`) and its own deterministic RNG;
  `create_dataset_split.py` reproduces identical splits for the same
  manifest + seed.
- Validation and evaluation are deterministic given the same inputs.