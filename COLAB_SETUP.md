# FreshSense Colab Setup Guide

This guide provides step-by-step instructions for training FreshSense Phase 1 on Google Colab.

## Prerequisites

- Google account
- Google Drive (for dataset storage)
- Basic familiarity with Google Colab

---

## Step 1: Prepare Your Dataset

1. Organize your dataset in the following structure:
```
data/
└── raw/
    ├── fresh/
    │   ├── apple_01.jpg
    │   └── banana_02.png
    ├── stale/
    │   └── ...
    └── rotten/
        └── ...
```

2. Compress the dataset:
```bash
zip -r freshsense_dataset.zip data/
```

---

## Step 2: Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder named `FreshSense`
3. Upload `freshsense_dataset.zip`
4. Upload the entire `FreshSense` project folder (or clone directly in Colab)

---

## Step 3: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **New Notebook**
3. Go to **Runtime > Change runtime type**
4. Select **GPU** as hardware accelerator
5. Click **Save**

---

## Step 4: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify mount
import os
os.listdir('/content/drive/MyDrive')
```

---

## Step 5: Upload Project (Alternative Method)

If you prefer to upload the project directly:

1. In Colab, click the **Files** tab (folder icon on the left)
2. Click **Upload**
3. Select the `FreshSense` folder
4. Wait for upload to complete

**OR** clone directly from GitHub:
```python
!git clone https://github.com/YOUR_USERNAME/FreshSense.git
%cd FreshSense
```

---

## Step 6: Extract Dataset

```python
# Extract dataset
!unzip -q '/content/drive/MyDrive/FreshSense/freshsense_dataset.zip' -d data/

# Verify extraction
import os
print("Classes:", os.listdir('data/raw'))
```

---

## Step 7: Install Dependencies

```python
# Install PyTorch with CUDA support
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
!pip install -r requirements.txt

# Verify installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Step 8: Validate Project

```python
# Run validation script
!python scripts/validate_project.py
```

Expected output:
```
✓ Imports: All modules imported successfully
✓ Paths: No hardcoded absolute paths found
✓ Dataset: Dataset found: X classes
✓ Model Architecture: Architecture verified: 1280-256-6
✓ Configuration: Config valid: FreshSense AI
✓ Preprocessing: Preprocessing pipeline verified
✓ Output Folders: All output folders ready
✓ Requirements: All packages installed
```

---

## Step 9: Configure Training (Optional)

Edit `configs/settings.yaml` if needed:

```python
# View current config
!cat configs/settings.yaml
```

Common Colab adjustments:
```yaml
data:
  num_workers: 2  # Reduce for Colab
  batch_size: 32  # Adjust based on GPU memory

training:
  epochs: 20
  mixed_precision: true  # Keep enabled for faster training
```

---

## Step 10: Run Training

```python
# Start training
!python -m src.main
```

Training will:
- Log to `logs/training.log`
- Save checkpoints to `models/checkpoints/`
- Save best model to `models/checkpoints/best_model.pth`
- Generate training history CSV
- Run evaluation after training

---

## Step 11: Monitor Training

Training progress will be displayed in the output. Key metrics:
- Train/Validation loss
- Train/Validation accuracy
- Learning rate
- Epoch time

---

## Step 12: Save Best Model to Drive

After training completes:

```python
# Copy best model to Google Drive
!cp models/checkpoints/best_model.pth /content/drive/MyDrive/FreshSense/

# Copy training history
!cp models/checkpoints/training_history.csv /content/drive/MyDrive/FreshSense/

# Copy metrics
!cp -r models/metrics /content/drive/MyDrive/FreshSense/

print("Model saved to Google Drive!")
```

---

## Step 13: Verify Training

```python
# Load and inspect checkpoint
import torch

checkpoint = torch.load('models/checkpoints/best_model.pth', map_location='cpu')
print(f"Best epoch: {checkpoint['best_epoch']}")
print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
print(f"Best val acc: {checkpoint['best_val_acc']:.2f}%")
print(f"Classes: {checkpoint['class_names']}")
print(f"Checkpoint version: {checkpoint.get('checkpoint_version', 'N/A')}")
print(f"Architecture: {checkpoint.get('classifier_type', 'N/A')}")
```

---

## Step 14: Download Model (Alternative)

If you didn't save to Drive, download directly:

```python
from google.colab import files
files.download('models/checkpoints/best_model.pth')
```

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `configs/settings.yaml`:
```yaml
data:
  batch_size: 16  # or 8
```

### Dataset Not Found
Ensure the dataset is extracted correctly:
```python
!ls -la data/raw/
```

### Slow Training
- Ensure GPU is enabled: `Runtime > Change runtime type > GPU`
- Reduce `num_workers` to 2
- Use mixed precision (already enabled by default)

### Import Errors
Restart runtime: `Runtime > Restart runtime`

---

## Colab Session Management

### Save Progress
Always save checkpoints to Google Drive:
```python
!cp models/checkpoints/best_model.pth /content/drive/MyDrive/FreshSense/
```

### Resume Training
If session disconnects:
1. Re-run setup steps 1-8
2. Resume from last checkpoint:
```python
!python -m src.main resume_from=models/checkpoints/last_model.pth
```

### Session Timeout
Colab sessions timeout after ~12 hours of inactivity. Train in chunks:
```yaml
training:
  epochs: 10  # Train in batches of 10 epochs
  resume_from: models/checkpoints/last_model.pth  # Resume between sessions
```

---

## Post-Training

### Download All Artifacts
```python
# Create a ZIP of all results
!zip -r freshtraining_results.zip models/ logs/

# Download
from google.colab import files
files.download('freshtraining_results.zip')
```

### Use the Model
The trained model can be used for:
- Single-image inference (Phase 1)
- Real-time webcam inference (Phase 2)
- Deployment to production

---

## Quick Reference

| Task | Command |
|------|---------|
| Mount Drive | `drive.mount('/content/drive')` |
| Extract dataset | `!unzip dataset.zip -d data/` |
| Install deps | `!pip install -r requirements.txt` |
| Validate | `!python scripts/validate_project.py` |
| Train | `!python -m src.main` |
| Save to Drive | `!cp models/checkpoints/best_model.pth /content/drive/MyDrive/FreshSense/` |
| Download model | `files.download('models/checkpoints/best_model.pth')` |

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check the documentation in `README.md`
- Review `TESTING.md` for troubleshooting