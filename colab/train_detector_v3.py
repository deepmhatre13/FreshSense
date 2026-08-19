# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SmartFreshAI — Controlled V3 Experiment: YOLO11n on V3 (exclusion dataset)
#
# This notebook runs the **controlled YOLO experiment**:
#
#     V3 dataset = V2 dataset minus the 14 unresolved TRAIN/VALIDATION blockers
#     Model      = YOLO11n, imgsz=640, epochs=50, batch=16, seed=42, patience=10
#     Benchmark  = the UNCHANGED V2 TEST set (never V3 test)
#
# ## Workflow
# 1. Verify CUDA / Tesla T4 GPU
# 2. Ensure `data/detection` (frozen V2) is present
# 3. Build `data/detection_v3` deterministically with the repo exclusion builder
# 4. Verify V3 counts
# 5. Train YOLO11n on V3 -> `models/detection/v3` (isolated, never touches best.pt)
# 6. Evaluate on the V2 TEST set -> `reports/detection_v3_yolo11n_test.json`
# 7. Compare against the frozen baseline -> `reports/yolo/EXP_V3_YOLO11N_COMPARISON.md`
#
# The frozen baseline at `models/detection/detector/weights/best.pt` is NEVER
# overwritten.

# %% [markdown]
# ## Stage A — Verify GPU

# %%
import sys, platform, torch, ultralytics, os, json
print(f"Python:       {platform.python_version()}")
print(f"PyTorch:      {torch.__version__}")
print(f"Ultralytics:  {ultralytics.__version__}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    print(f"CUDA device:  {name}")
    print(f"CUDA mem:     {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("Tesla T4 detected:", "Tesla T4" in name)
else:
    print("WARNING: No GPU. Aborting.")
    sys.exit(1)

# %% [markdown]
# ## Stage B — Repository & frozen V2 dataset

# %%
if not os.path.isdir("data/detection") or not os.path.isfile("data/detection/data.yaml"):
    print("Frozen V2 dataset not found. Reconstruct data/detection first.")
    print(os.getcwd(), os.listdir(".")[:20])
    raise SystemExit(1)
print("Frozen V2 detected:", len(os.listdir("data/detection/train/images")), "train images")

# %% [markdown]
# ## Stage C — Build V3 deterministically (exclusion builder)

# %%
# data/detection_v3 = V2 minus the 14 blocker train/valid samples.
# --force allows an idempotent rebuild over an existing V3 (deterministic).
!python scripts/build_detection_v3.py --exclusion-build --force

# %% [markdown]
# ## Stage D — Verify V3 counts

# %%
from pathlib import Path
def count_images(d):
    return len(list(Path(d).glob("*.jpg"))) if Path(d).is_dir() else -1
for sp in ("train", "valid", "test"):
    print(f"V3 {sp}: {count_images(f'data/detection_v3/{sp}/images')} images")

# %% [markdown]
# ## Stage E — Train YOLO11n on V3

# %%
EPOCHS, BATCH, IMGSZ, PATIENCE, SEED = 50, 16, 640, 10, 42
!python scripts/train_detector_v3.py \
# %% [markdown]
# ## Stage F — Locate the V3 best/last checkpoints

# %%
run_dir = "models/detection/v3/exp_v3_yolo11n_640"
weights_dir = os.path.join(run_dir, "weights")
print("weights dir:", os.listdir(weights_dir) if os.path.isdir(weights_dir) else "NO WEIGHTS")
best = os.path.join(weights_dir, "best.pt")
last = os.path.join(weights_dir, "last.pt")
print("best exists:", os.path.exists(best))
print("last exists:", os.path.exists(last))
print("run files:", sorted(os.listdir(run_dir)))

# %% [markdown]
# ## Stage G — Evaluate on the UNCHANGED V2 TEST set (NOT V3 test)

# %%
# Benchmark is the original V2 data.yaml / test split. V3 test == V2 test, but
# we evaluate against data/detection/data.yaml to be provably safe.
TEST_REPORT = "reports/detection_v3_yolo11n_test.json"
!python scripts/evaluate_detector.py \
    --model {best} \
    --data data/detection/data.yaml \
    --split test \
    --output {TEST_REPORT}

with open(TEST_REPORT, encoding="utf-8") as f:
    test_metrics = json.load(f)
print("V3 on V2 test set:")
print(f"  precision : {test_metrics['precision']:.4f}")
print(f"  recall    : {test_metrics['recall']:.4f}")
print(f"  mAP50     : {test_metrics['map50']:.4f}")
print(f"  mAP50-95  : {test_metrics['map50_95']:.4f}")

# %% [markdown]
# ## Stage H — Generate comparison vs frozen baseline

# %%
!python scripts/compare_yolo_results.py \
    --baseline reports/detection_baseline_test.json \
    --experiment {TEST_REPORT} \
    --output reports/yolo/EXP_V3_YOLO11N_COMPARISON.md
print("Comparison written: reports/yolo/EXP_V3_YOLO11N_COMPARISON.md")

# %% [markdown]
# ## Stage I — Package V3 artifacts (download locally)

# %%
import shutil, zipfile
ARTIFACT = "/content/v3_artifacts"
os.makedirs(f"{ARTIFACT}/weights", exist_ok=True)
os.makedirs(f"{ARTIFACT}/training", exist_ok=True)
os.makedirs(f"{ARTIFACT}/reports", exist_ok=True)
for f in (best, last):
    if os.path.exists(f):
        shutil.copy(f, f"{ARTIFACT}/weights/{os.path.basename(f)}")
for f in ("results.csv", "args.yaml", "results.png"):
    src = os.path.join(run_dir, f)
    if os.path.exists(src):
        shutil.copy(src, f"{ARTIFACT}/training/{f}")
for f in (TEST_REPORT, "reports/yolo/EXP_V3_YOLO11N_COMPARISON.md",
          "reports/detection_v3_exclusion_report.json"):
    if os.path.exists(f):
        shutil.copy(f, f"{ARTIFACT}/reports/{os.path.basename(f)}")
z = zipfile.ZipFile("/content/SmartFreshAI_V3_yolo11n_artifacts.zip", "w", zipfile.ZIP_DEFLATED)
for root, _, fs in os.walk(ARTIFACT):
    for fn in fs:
        p = os.path.join(root, fn)
        z.write(p, os.path.relpath(p, ARTIFACT))
z.close()
print("Packaged: /content/SmartFreshAI_V3_yolo11n_artifacts.zip")

# %%
from google.colab import files
files.download("/content/SmartFreshAI_V3_yolo11n_artifacts.zip")

# %% [markdown]
# ## Summary
# After download, restore locally:
# - `models/detection/experiments/exp_v3_yolo11n_640/best.pt` (and last.pt)
# - Evaluate locally against the V2 test set:
#   `python scripts/evaluate_detector.py --model
#    models/detection/experiments/exp_v3_yolo11n_640/best.pt \
#    --data data/detection/data.yaml --split test --output
#    reports/detection_v3_yolo11n_test_local.json`
#
# Do NOT move this checkpoint over `models/detection/detector/weights/best.pt`.
# Only adopt V3 as production if it meets the Phase-9 criteria.
    --data data/detection_v3/ \
    --model yolo11n.pt \
    --epochs {EPOCHS} \
    --batch {BATCH} \
    --imgsz {IMGSZ} \
    --device 0 \
    --workers 4 \
    --patience {PATIENCE} \
    --seed {SEED} \
    --project models/detection/v3 \
    --name exp_v3_yolo11n_640
