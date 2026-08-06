# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-01

### Added

- Initial Phase 1 release
- Training pipeline with EfficientNet-B0 transfer learning
  - Mixed precision training (AMP) with automatic CPU fallback
  - Gradient clipping for training stability
  - ReduceLROnPlateau learning rate scheduling
  - Early stopping on validation loss
  - Two-stage training (classifier warmup + backbone unfreeze)
- Comprehensive logging system
  - Console output (INFO+)
  - Rotating file handler (training.log, 5MB x 3 backups)
  - Error-only log (errors.log)
- Checkpointing system
  - Best model checkpoint (lowest validation loss)
  - Last model checkpoint (latest epoch)
  - Periodic epoch checkpoints (configurable interval)
  - Training history CSV logging
  - Resume training from checkpoint with full state restoration
- Dataset management
  - Automatic layout detection (flat, split, nested structures)
  - Image quality filtering (resolution, blur, exposure)
  - Stratified train/validation/test splits with fixed seed
  - Class imbalance reporting
- Evaluation framework
  - Top-1 accuracy, weighted precision/recall/F1
  - Per-class metrics
  - Confusion matrix visualization
  - ROC curves (one-vs-rest)
  - Precision-Recall curves (one-vs-rest)
  - Confidence distribution histogram
  - Misclassified image grid
- Inference pipeline
  - Single-image prediction with checkpoint loading
  - Automatic class name recovery from checkpoint
  - Mirror training preprocessing exactly
- Configuration system
  - Frozen dataclass configuration hierarchy
  - YAML-based configuration (configs/settings.yaml)
  - Runtime device optimization (Windows/CPU/CUDA)
  - Full reproducibility with seed_everything()