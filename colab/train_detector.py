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
# # SmartFreshAI — YOLO Fruit Detector Training (GPU / Google Colab)
#
# This notebook trains a YOLO11n detector on the Roboflow **fruits-test-ajvf8-duncc**
# dataset (Version 1, 10 classes, 1106 images).
#
# ## Workflow
# 1. Upload repository ZIP → extract → install dependencies
# 2. Configure the Roboflow API key
# 3. Download & validate the dataset
# 4. Train YOLO11n on GPU
# 5. Evaluate on val + test splits
# 6. Package trained artifacts for download
#
# ## After training
# Download `SmartFreshAI_yolo_artifacts.zip` from Colab and restore locally:
# - Place `best.pt` → `models/detection/detector/weights/best.pt`
# - Place `last.pt` → `models/detection/detector/weights/last.pt`
# - Evaluation reports → `reports/`
#
# ---
# **No API keys are hardcoded in this notebook.**
# You will be prompted to provide your `ROBOFLOW_API_KEY` via Colab secrets or
# a text input.
# ---

# %% [markdown]
# ## Stage A — Verify GPU

# %%
import sys
import platform
import torch
import ultralytics

print(f"Python:       {platform.python_version()}")
print(f"PyTorch:      {torch.__version__}")
print(f"Ultralytics:  {ultralytics.__version__}")
print(f"CUDA avail:   {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device:  {torch.cuda.get_device_name(0)}")
    print(f"CUDA mem:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Training will be very slow on CPU.")
    response = input("Continue on CPU? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted. Enable GPU in Runtime → Change runtime type.")
        sys.exit(1)

# %% [markdown]
# ## Stage B — Repository Setup

# %%
# Mount Google Drive (optional, for persistent storage)
from google.colab import drive
drive.mount("/content/drive")

# %%
# Upload and extract the repository ZIP
import zipfile
import os

ZIP_PATH = "/content/SmartFreshAI.zip"
if not os.path.exists(ZIP_PATH):
    from google.colab import files
    uploaded = files.upload()
    ZIP_PATH = list(uploaded.keys())[0]

print(f"Extracting {ZIP_PATH} ...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall("/content/")
print("Extraction complete.")

# %%
%cd /content/SmartFreshAI

# %% [markdown]
# ## Stage C — Install Dependencies

# %%
# Install Colab-specific requirements (GPU torch, ultralytics, etc.)
# %% [markdown]
# ## Stage F — Train YOLO Detector

# %%
# Default training parameters (override via --flags)
EPOCHS = 50
BATCH = 16
IMGSZ = 640
PATIENCE = 10
DEVICE = "0"  # first GPU

!python -m scripts.train_detector \
    --epochs {EPOCHS} \
    --batch {BATCH} \
    --imgsz {IMGSZ} \
    --patience {PATIENCE} \
    --device {DEVICE}

# %% [markdown]
# ## Stage G — Evaluate on Validation Set

# %%
BEST_PT = "models/detection/detector/weights/best.pt"
VAL_REPORT = "reports/detection_evaluation_val.json"

!python -m scripts.evaluate_detector \
    --model {BEST_PT} \
    --split val \
    --output {VAL_REPORT}

# Display validation report
import json
with open(VAL_REPORT) as f:
    val_metrics = json.load(f)
print(json.dumps(val_metrics, indent=2))

# %% [markdown]
# ## Stage H — Evaluate on Test Set

# %%
TEST_REPORT = "reports/detection_evaluation_test.json"

!python -m scripts.evaluate_detector \
    --model {BEST_PT} \
    --split test \
    --output {TEST_REPORT}

with open(TEST_REPORT) as f:
    test_metrics = json.load(f)
print(json.dumps(test_metrics, indent=2))

# %% [markdown]
# ## Stage I — Package Artifacts

# %%
import shutil
from datetime import datetime

ARTIFACT_DIR = "/content/artifacts"
os.makedirs(f"{ARTIFACT_DIR}/detector/weights", exist_ok=True)
os.makedirs(f"{ARTIFACT_DIR}/reports", exist_ok=True)
os.makedirs(f"{ARTIFACT_DIR}/training", exist_ok=True)
os.makedirs(f"{ARTIFACT_DIR}/metadata", exist_ok=True)

# Copy model weights
shutil.copy("models/detection/detector/weights/best.pt", f"{ARTIFACT_DIR}/detector/weights/best.pt")
shutil.copy("models/detection/detector/weights/last.pt", f"{ARTIFACT_DIR}/detector/weights/last.pt")

# Copy evaluation reports
shutil.copy(VAL_REPORT, f"{ARTIFACT_DIR}/reports/")
shutil.copy(TEST_REPORT, f"{ARTIFACT_DIR}/reports/")

# Copy training results
training_dir = "models/detection/detector"
if os.path.exists(f"{training_dir}/results.csv"):
    shutil.copy(f"{training_dir}/results.csv", f"{ARTIFACT_DIR}/training/results.csv")
for plot in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
             "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"]:
    src = f"{training_dir}/{plot}"
    if os.path.exists(src):
        shutil.copy(src, f"{ARTIFACT_DIR}/training/{plot}")

# Write metadata
metadata = {
    "timestamp": datetime.utcnow().isoformat(),
    "model": "yolo11n.pt",
    "epochs": EPOCHS,
    "batch": BATCH,
    "imgsz": IMGSZ,
    "dataset_version": 1,
    "dataset_classes": list(val_metrics.get("per_class", {}).keys()) if val_metrics else [],
    "ultralytics_version": ultralytics.__version__,
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
with open(f"{ARTIFACT_DIR}/metadata/training_config.json", "w") as f:
    json.dump(metadata, f, indent=2)

# %%
# Package into ZIP for download
ZIP_OUT = "/content/SmartFreshAI_yolo_artifacts.zip"
with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(ARTIFACT_DIR):
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, ARTIFACT_DIR)
            zf.write(full_path, f"artifacts/{rel_path}")

print(f"Artifacts packaged: {ZIP_OUT}")

# %%
# Offer download link in Colab
from google.colab import files
files.download(ZIP_OUT)

# %% [markdown]
# ## Summary
#
# Training complete. The artifact ZIP contains:
# - `detector/weights/best.pt`
# - `detector/weights/last.pt`
# - `reports/` — JSON evaluation reports
# - `training/` — CSV logs + training plots
# - `metadata/` — training configuration snapshot
#
# ## Next steps (back on the local machine)
#
# ```powershell
# # Restore best.pt → models\detection\detector\weights\best.pt
# python -c "from ultralytics import YOLO; m = YOLO('models/detection/detector/weights/best.pt'); print('Model loaded')"
# python -m scripts.run_webcam
# ```
!pip install -r requirements-colab.txt -q

# Verify key packages
import torch, ultralytics, roboflow, cv2, numpy as np, PIL
print(f"PyTorch:      {torch.__version__}   CUDA: {torch.cuda.is_available()}")
print(f"Ultralytics:  {ultralytics.__version__}")
print(f"Roboflow:     {roboflow.__version__}")
print(f"OpenCV:       {cv2.__version__}")
print(f"NumPy:        {np.__version__}")
print(f"Pillow:       {PIL.__version__}")

# %% [markdown]
# ## Stage D — Configure Roboflow API Key

# %%
import os
from getpass import getpass

# Try Colab secrets first, then prompt.
try:
    from google.colab import userdata
    API_KEY = userdata.get("ROBOFLOW_API_KEY")
except Exception:
    API_KEY = os.environ.get("ROBOFLOW_API_KEY")

if not API_KEY:
    API_KEY = getpass("Enter your Roboflow API key: ")

os.environ["ROBOFLOW_API_KEY"] = API_KEY
print("ROBOFLOW_API_KEY configured.")

# Write .env so the project's own loader finds it
with open(".env", "w") as f:
    f.write(f"ROBOFLOW_API_KEY={API_KEY}\n")
print(".env created.")

# %% [markdown]
# ## Stage E — Download & Validate Dataset

# %%
!python -m scripts.download_detection_dataset

# %%
!python -m scripts.validate_detection_dataset