# SmartFresh AI — Phase 1 Testing Checklist

Manual test checklist to validate every pipeline stage. Run each check and
mark it off. All commands assume you are in the project root with the
virtual environment activated.

---

## 0. Environment

- [ ] `python --version` → 3.10+ (tested on 3.12)
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python -c "import torch, albumentations, cv2, sklearn; print('OK')"` → `OK`

---

## 1. Configuration

- [ ] `python -c "from configs.config import Config; c = Config.from_yaml('configs/settings.yaml'); print(c.device)"` loads without error
- [ ] Invalid split (e.g. `train_split: 0.5`) raises `ValueError: Splits must sum to 1.0`
- [ ] `persistent_workers: true` with `num_workers: 0` raises a clear `ValueError`

---

## 2. Dataset

- [ ] `data/raw/<class>/*.jpg` structure is detected (one class per folder)
- [ ] Corrupted / unreadable images are skipped (not fatal)
- [ ] Images below `min_width`/`min_height` are skipped
- [ ] Blurry images (Laplacian variance < `blur_threshold`) are skipped
- [ ] Over/underexposed images are skipped
- [ ] Class imbalance report prints at startup
- [ ] Train/val/test split is stratified and deterministic (same seed → same split)

### Quick dataset smoke test

```bash
python -c "import sys; sys.path.insert(0,'.'); from src.preprocessing.dataset import FreshSenseDatasetLoader; from src.preprocessing.augmentation import AugmentationPipeline; p=AugmentationPipeline(); l=FreshSenseDatasetLoader('data/raw', p.get_transforms('train'), p.get_transforms('val'), p.get_transforms('test')); l.verify(); print('DATASET OK')"
```

- [ ] Prints `Verified train/val/test split` for each split
- [ ] Batch shape is `(B, 3, 224, 224)`, labels shape `(B,)`

---

## 3. Model

- [ ] `python -c "import sys; sys.path.insert(0,'.'); from src.models.efficientnet import FreshSenseEfficientNet; m=FreshSenseEfficientNet(3, pretrained=False, freeze_backbone=True); print(m.trainable_parameters(), m.frozen_parameters(), m.total_parameters())"` runs
- [ ] With `freeze_backbone=True`, trainable params ≈ classifier only (~3.8k for 3 classes)
- [ ] With `freeze_backbone=False`, trainable params ≈ full model (~4M)
- [ ] One forward pass works: `m(torch.randn(2,3,224,224))` → `(2, 3)`
- [ ] `model.summary()` prints (torchinfo) or falls back gracefully

---

## 4. Preprocessing & Augmentation

- [ ] `ImagePreprocessor` returns RGB uint8 `(224, 224, 3)`
- [ ] `AugmentationPipeline.get_transforms('train')` builds without error
- [ ] `get_transforms('val')` and `get_transforms('test')` build without error
- [ ] `get_transforms('bogus')` raises `ValueError`
- [ ] Train transform output is a normalized CHW tensor `(3, 224, 224)`

---

## 5. Training

- [ ] One training epoch runs without error
- [ ] Validation runs without error
- [ ] `best_model.pth` is saved when val loss improves
- [ ] `last_model.pth` is saved every epoch
- [ ] `epoch_<N>.pth` periodic checkpoints are saved
- [ ] `training_history.csv` is written with correct columns
- [ ] `logs/training.log` and `logs/errors.log` are created
- [ ] Early stopping triggers after `patience` epochs without improvement
- [ ] `ReduceLROnPlateau` reduces LR on plateau
- [ ] Gradient clipping is applied (no NaN explosion)
- [ ] Mixed precision works on CUDA (or is disabled on CPU)
- [ ] `KeyboardInterrupt` (Ctrl+C) saves a checkpoint and exits cleanly

### One-epoch smoke test

```bash
python -c "import sys; sys.path.insert(0,'.'); import torch; from src.models.efficientnet import FreshSenseEfficientNet; from src.training.trainer import Trainer; from src.training.losses import build_criterion; from torch.utils.data import DataLoader, TensorDataset; m=FreshSenseEfficientNet(3, pretrained=False, freeze_backbone=False); d=TensorDataset(torch.randn(16,3,224,224), torch.randint(0,3,(16,))); dl=DataLoader(d, batch_size=4); opt=torch.optim.AdamW(m.parameters(), lr=1e-3); t=Trainer(m, dl, dl, build_criterion(3), opt, None, torch.device('cpu'), epochs=1, checkpoint_dir='models/checkpoints'); h=t.fit(); print('TRAIN OK', h.best_epoch)"
```

- [ ] Prints `TRAIN OK` and writes checkpoints + CSV

---

## 6. Resume

- [ ] Set `training.resume_from` to `models/checkpoints/epoch_1.pth`
- [ ] Run `python -m src.main` → logs `Resumed from ... at epoch 1`
- [ ] History and patience counter are restored

---

## 7. Evaluation

- [ ] `evaluation_metrics.json` is written
- [ ] `confusion_matrix.png` is written
- [ ] `roc_curves.png` and `pr_curves.png` are written (multi-class)
- [ ] `confidence_distribution.png` is written
- [ ] `misclassified.png` is written (if any misclassifications)
- [ ] Metrics include accuracy, top-1, precision, recall, F1, per-class metrics
- [ ] `Evaluator.save_all()` works after `evaluate()` (no re-inference)

---

## 8. Inference

- [ ] `Predictor` loads `best_model.pth` without error
- [ ] `predict()` returns `{'class', 'confidence', 'probabilities'}`
- [ ] Prediction matches training preprocessing (normalized input)
- [ ] Missing image raises `FileNotFoundError`
- [ ] Corrupted image raises `ValueError`

```bash
python -c "import sys; sys.path.insert(0,'.'); from src.inference.predict import Predictor; p=Predictor('models/checkpoints/best_model.pth', ['fresh','stale','rotten']); print(p.predict('path/to/image.jpg'))"
```

---

## 9. Full Pipeline

- [ ] `python -m src.main` runs end-to-end with exit code 0
- [ ] All artifacts exist (checkpoints, CSV, JSON, PNGs, logs)
- [ ] Re-running with the same seed produces the same split
- [ ] `logs/errors.log` is empty on a successful run

---

## 10. Failure Modes

- [ ] Empty `data/raw` → clear `ValueError: No class directories found`
- [ ] All images rejected → clear `ValueError: No valid images found`
- [ ] Missing `configs/settings.yaml` → clear `FileNotFoundError`
- [ ] Wrong checkpoint format to `Predictor` → clear `RuntimeError`