# FreshSense

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-red)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![Version](https://img.shields.io/badge/Version-0.1.0-orange)

AI-powered Fruit & Vegetable Freshness Analysis Platform — Phase 1

## Overview

FreshSense is a production-ready machine learning pipeline for classifying the freshness of fruits and vegetables. Phase 1 implements a complete training and evaluation pipeline using EfficientNet-B0 transfer learning with mixed precision training, comprehensive logging, checkpointing, and evaluation metrics.

The system automatically validates image quality, handles dataset splitting, and provides detailed evaluation metrics including confusion matrices, ROC curves, and misclassification analysis.

## Features

### Training Pipeline
- **EfficientNet-B0** transfer learning with ImageNet pretrained weights
- **Mixed precision training** (AMP) with automatic CPU fallback
- **Two-stage training**: Classifier warmup followed by backbone unfreezing
- **Gradient clipping** for training stability
- **ReduceLROnPlateau** learning rate scheduling
- **Early stopping** on validation loss with patience
- **Checkpointing**: Best model, last model, and periodic epoch checkpoints
- **Resume training** from any checkpoint with full state restoration

### Dataset Management
- **Automatic layout detection** supporting multiple dataset structures (flat, split, nested)
- **Image quality filtering**: Resolution, blur detection (Laplacian variance), exposure validation
- **Stratified splitting** with fixed random seed for reproducibility
- **Class imbalance reporting** with detailed statistics
- **Flexible format support**: JPG, JPEG, PNG, BMP, WebP

### Evaluation & Metrics
- Comprehensive metrics: Accuracy, Precision, Recall, F1-score
- **Per-class metrics** for detailed performance analysis
- **Confusion matrix** visualization
- **ROC curves** (one-vs-rest) with AUC scores
- **Precision-Recall curves** with AUC scores
- **Confidence distribution** histogram
- **Misclassified images** grid with true/predicted labels

### Inference
- **Single-image prediction** with automatic class recovery
- **Checkpoint-based loading** with full metadata preservation
- **Production-ready** preprocessing matching training pipeline exactly

### Configuration & Logging
- **YAML-based configuration** with frozen dataclass validation
- **Three-log system**: Console (INFO+), training.log (rotating), errors.log (ERROR+)
- **Full reproducibility** with seed_everything() for deterministic training
- **Runtime optimization** for Windows, CPU, and CUDA environments

## Architecture

```
Input Image
   ↓
OpenCV Image Validation (quality checks)
   ↓
OpenCV Preprocessing (resize + BGR→RGB)
   ↓
Albumentations (augmentation + ImageNet normalization)
   ↓
EfficientNet-B0 (transfer learning)
   ↓
Fresh / Stale / Rotten classification
```

## Directory Structure

```
FreshSense/
├── configs/
│   ├── config.py          # Frozen dataclass configs + YAML loading/validation
│   └── settings.yaml      # Human-readable configuration (source of truth)
├── src/
│   ├── main.py            # Entry point: python -m src.main
│   ├── inference/
│   │   ├── predict.py     # Single-image inference
│   │   └── camera.py      # Webcam capture (Phase 2)
│   ├── models/
│   │   └── efficientnet.py # EfficientNet-B0 implementation
│   ├── preprocessing/
│   │   ├── augmentation.py # Albumentations pipelines
│   │   ├── dataset.py      # Dataset loading and splitting
│   │   ├── preprocess.py   # OpenCV preprocessing
│   │   └── quality.py      # Image quality validation
│   ├── training/
│   │   ├── trainer.py      # Training loop with checkpointing
│   │   ├── evaluate.py     # Evaluation and metrics
│   │   └── losses.py       # Loss functions
│   └── utils/
│       ├── logger.py       # Production logging setup
│       ├── metrics.py      # Metric computation helpers
│       └── visualization.py # Plotting utilities
├── data/                  # Dataset (gitignored)
│   ├── raw/               # Place class folders here
│   └── processed/         # Reserved for future phases
├── models/                # Model artifacts (gitignored)
│   ├── checkpoints/       # Best, last, and epoch checkpoints
│   └── metrics/           # Evaluation plots and JSON
├── logs/                  # Log files (gitignored)
├── .github/               # GitHub templates and workflows
├── requirements.txt       # Pinned dependencies
├── README.md             # This file
├── CONTRIBUTING.md       # Contribution guidelines
├── CODE_OF_CONDUCT.md    # Community guidelines
├── CHANGELOG.md          # Version history
├── ROADMAP.md            # Future plans
├── TESTING.md            # Testing guide
├── LICENSE               # MIT License
└── VERSION               # Current version (0.1.0)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch 2.13.0 |
| **Model** | EfficientNet-B0 (torchvision) |
| **Image Processing** | OpenCV 5.0.0 |
| **Augmentation** | Albumentations 2.0.8 |
| **Metrics** | scikit-learn 1.6.1 |
| **Logging** | Python logging with RotatingFileHandler |
| **Configuration** | PyYAML + Python dataclasses |
| **Visualization** | Matplotlib 3.11.0 |

## Installation

### Requirements

- Python **3.10+** (tested on 3.12)
- pip
- CUDA-capable GPU (optional, training works on CPU)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/FreshSense.git
cd FreshSense

# 2. Create a virtual environment (recommended)
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Verify the installation
python -c "import torch, albumentations, cv2, sklearn; print('OK')"
```

> **Note:** `requirements.txt` pins exact versions for reproducibility. If you plan to use a GPU, install the CUDA build of PyTorch before running `pip install -r requirements.txt` (see [PyTorch install guide](https://pytorch.org/get-started/locally/)).

## Dataset

Place images in class folders under `data/raw`. **Each subfolder name is a class**:

```
data/raw/
├── fresh/
│   ├── apple_01.jpg
│   └── banana_02.png
├── stale/
│   └── ...
└── rotten/
    └── ...
```

### Supported Formats
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### Quality Filtering
Invalid images are automatically skipped:
- **Resolution**: Minimum 224×224 pixels
- **Blur**: Laplacian variance < 100.0
- **Exposure**: Mean brightness < 40 (underexposed) or > 220 (overexposed)

### Dataset Layouts Supported
1. **Flat**: `data/raw/<class>/*.jpg`
2. **Split**: `data/raw/{train,val,test}/<class>/*.jpg`
3. **Nested**: Automatically detects train/test folders at any depth

## Training

```bash
python -m src.main
```

Everything is driven by `configs/settings.yaml` — **no code changes required**.

### Configuration

Edit `configs/settings.yaml` to customize:

```yaml
data:
  image_size: 224
  batch_size: 32
  num_workers: 4

model:
  pretrained: true
  freeze_backbone: true
  dropout: 0.30

training:
  epochs: 20
  learning_rate: 0.001
  mixed_precision: true
  patience: 5
  resume_from: null  # e.g. "models/checkpoints/epoch_5.pth"
```

### Key Features

- **Deterministic**: Fixed seed ensures reproducible results
- **Resumable**: Interrupt and resume training from checkpoints
- **Efficient**: Mixed precision, gradient clipping, and optimized DataLoader
- **Monitorable**: Real-time logging to console and files

### Training on CPU

The pipeline automatically runs on CPU when CUDA is unavailable. For CPU-only machines:
- Set `training.mixed_precision: false` (automatically disabled on CPU)
- Reduce `num_workers` if you hit memory limits

## Evaluation

Evaluation runs automatically after training and produces:

| Artifact | Location |
|----------|----------|
| Metrics JSON | `models/metrics/evaluation_metrics.json` |
| Confusion matrix | `models/metrics/confusion_matrix.png` |
| ROC curves | `models/metrics/roc_curves.png` |
| PR curves | `models/metrics/pr_curves.png` |
| Confidence distribution | `models/metrics/confidence_distribution.png` |
| Misclassified images | `models/metrics/misclassified.png` |
| Training history | `models/checkpoints/training_history.csv` |

## Results

After a successful training run:

```
logs/training.log                      <- Full training log
logs/errors.log                        <- Errors only
models/checkpoints/best_model.pth      <- Best validation loss
models/checkpoints/last_model.pth      <- Latest epoch
models/checkpoints/epoch_<N>.pth       <- Periodic checkpoints
models/checkpoints/training_history.csv
models/metrics/*.json + *.png          <- Evaluation artifacts
```

Exit code `0` on success, `1` on failure, `130` on Ctrl+C.

## Project Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Training Pipeline |
| **Phase 2** | 📋 Planned | Real-time Webcam |
| **Phase 3** | 📋 Planned | YOLO Detection |
| **Phase 4** | 📋 Planned | LangGraph AI |
| **Phase 5** | 📋 Planned | FastAPI Backend |
| **Phase 6** | 📋 Planned | React Dashboard |
| **Phase 7** | 📋 Planned | Docker Deployment |
| **Phase 8** | 📋 Planned | Cloud Deployment |

See [ROADMAP.md](ROADMAP.md) for detailed plans.

## Future Work

- Real-time webcam inference with FPS optimization
- Object detection for multi-fruit analysis (YOLO)
- Natural language query interface (LangGraph)
- REST API for remote inference (FastAPI)
- Web dashboard for monitoring (React)
- Containerized deployment (Docker)
- Cloud scalability (AWS/GCP/Azure)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'feat: add feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **EfficientNet** paper: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **Albumentations** library for fast image augmentation
- **PyTorch** team for the deep learning framework
- Open source community for invaluable tools and libraries

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Check existing documentation in [TESTING.md](TESTING.md)
- Review the [ROADMAP.md](ROADMAP.md) for upcoming features

---

**Status**: Phase 1 Production-Ready | **Version**: 0.1.0