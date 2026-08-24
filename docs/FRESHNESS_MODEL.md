# Freshness Model (`docs/FRESHNESS_MODEL.md`)

## Overview

The freshness subsystem is an **ML image classifier** over a canonical
16-class taxonomy. It is the *only* source of a `fresh` / `rotten` /
`uncertain` prediction. It never guesses freshness from fruit appearance,
colour, filenames, or YOLO confidence.

> **Freshness is an ML classification result only for fruits covered by the
> trained freshness model.**

## Supported fruits (have a validated freshness model)

| Fruit          | Model classes                 | Availability status |
|----------------|-------------------------------|---------------------|
| Apple          | `Apple_fresh` / `Apple_rotten`| AVAILABLE           |
| Banana         | `banana_fresh` / `banana_rotten`| AVAILABLE          |
| Grape          | `Grape_fresh` / `Grape_rotten`| AVAILABLE           |
| Guava          | `guava_fresh` / `guava_rotten`| AVAILABLE           |
| Orange         | `Orange_fresh` / `Orange_rotten`| AVAILABLE          |
| Strawberry     | `Strawberry_fresh` / `Strawberry_rotten`| AVAILABLE  |

Jujube and Pomegranate also have valid fresh/rotten training data
(200 fresh / 200 rotten each, Mendeley), and the classifier includes their
16 classes. However the frozen YOLO detector does not emit a Jujube or
Pomegranate bounding box, so production image/webcam detection never
surfaces them; the API returns `data_not_available` for them. This is
intentional and consistent with the availability registry
(`configs/freshness_availability.json`, status `AVAILABLE_DETECTOR_UNSUPPORTED`).

## Unavailable fruits (no validated freshness model)

| Fruit | Reason | Production output |
|---|---|---|
| Kiwi   | No valid fresh training data (only 0 fresh / 6 rotten residuals) | `data_not_available` |
| Mango  | No valid fresh training data (only 0 fresh / 3 rotten residuals) | `data_not_available` |
| Cherry | Insufficient residuals (4 fresh / 5 rotten) | `data_not_available` |
| Chickoo| No valid fresh/rotten training data at all | `data_not_available` |

For these fruits the detector still succeeds (fruit identity + detection
confidence + nutrition are returned) but freshness is NOT guessed.

## Class taxonomy

Deterministic, serialized in `data/freshness/class_mapping.json`:

```
0  Apple_fresh        1  Apple_rotten
2  banana_fresh       3  banana_rotten
4  Grape_fresh        5  Grape_rotten
6  guava_fresh        7  guava_rotten
8  Jujube_fresh       9  Jujube_rotten
10 Orange_fresh      11 Orange_rotten
12 Pomegranate_fresh 13 Pomegranate_rotten
14 Strawberry_fresh  15 Strawberry_rotten
```

No Mango / Kiwi / Cherry / Chickoo classes exist because no valid training
data exists. The classifier class count is derived dynamically from this file.

## Preprocessing

- Image size: `224 x 224` (matches the existing pipeline `image_size`).
- Deterministic inference transform — no training-time augmentation at
  inference.
- Same normalization / channel order as the established FreshSense pipeline.

## Training

- Architecture: EfficientNet-B0 transfer learning (backbone frozen for the
  transfer phase), same architecture family as the existing production
  pipeline.
- Optimizer: AdamW with the repository's established parameter-group scheme.
- Loss: `CrossEntropyLoss` over 16 classes.
- LR schedule: `ReduceLROnPlateau`.
- Early stopping + best-checkpoint saving (validation loss monitored).
- Deterministic seed (`42`).
- Mixed precision enabled when CUDA is available.
- **Immutable baseline**: `models/checkpoints/best_model.pth` is never
  overwritten. The expanded model trains to a versioned path
  `models/checkpoints/freshness_efficientnet_b0_16class.pth`.

## Uncertainty policy

A prediction is considered *confident* when the top-1 softmax probability is at
or above the calibrated uncertainty threshold. Below it the classifier returns
`uncertain` rather than guessing. The threshold is calibrated on the validation
set (never the test set) and exposed internally via
`PredictionResult.is_uncertain`, `top2_probabilities`, `raw_logits`, and
`uncertainty_threshold`.

The diagnostic fields exist so production can explain *why* a prediction became
uncertain: raw logits, softmax probabilities, predicted class, top-2
probabilities, crop dimensions are all available for that investigation.

## Evaluation results (final, held-out test set)

Checkpoint: `freshness_efficientnet_b0_16class.pth`, best epoch 3,
validation accuracy **95.21%** (480 validation images), held-out test
accuracy **93.75%**, macro precision **94.30%**, macro recall **93.75%**, macro F1 **93.54%**
(480 test images, 16-class top-1; see reports/freshness/training_metrics.json).

Per-fruit held-out test recall (from `reports/freshness/per_fruit_recall.json`):

| fruit | fresh_recall | rotten_recall |
|---|---|---|
| Apple | 0.933 | 1.000 |
| Grape | 0.967 | 0.967 |
| Jujube | 1.000 | 0.900 |
| Orange | 0.867 | 0.933 |
| Pomegranate | 1.000 | 0.633 |
| Strawberry | 0.967 | 1.000 |
| banana | 0.967 | 0.967 |
| guava | 1.000 | 0.933 |

Calibrated uncertainty threshold: **0.6635** (5th percentile of correct
prediction softmax confidences on the *validation* set; p(conf<0.66)=7.7%).
Stored inside the checkpoint metadata and loaded automatically by the
production `Predictor`.

### Real-image end-to-end API validation

Held-out test images were POSTed to the live API
(`reports/freshness/real_image_validation.json`). Findings:

* Supported fruits detected by YOLO are correctly graded (e.g. apple fresh,
  banana fresh, grape fresh: 100% fresh through the full API path).
* The frozen YOLO detector does not detect many Mendeley studio images at all
  (detector domain gap). Those requests legitimately return zero detections -
  nothing is fabricated.
* Some fruits are detected under a different detector label (e.g. a guava or
  jujube image labelled "Mango"). Because Mango has no freshness model, the
  API returns `data_not_available` for them - exactly the required behavior:
  the system refuses to grade a crop whose detector identity has no model.

Known limitation (documented honestly): direct-model accuracy on full frames
is higher than API-path accuracy because production crops to the YOLO box.
The box crop can include background/context that shifts the input away from
the training distribution (full-frame studio shots). This is an inherent
detector/classifier domain gap, not a threshold problem; lowering thresholds
to mask it is prohibited.

## Model version

- `model_version`: `v20260823`
- checkpoint: `models/checkpoints/freshness_efficientnet_b0_16class.pth`

## Provenance & license

- Source: Mendeley "Original Image" dataset, license CC BY 4.0.
- Canonical derived dataset: `data/freshness/` (train/valid/test).
- No fabricated labels; no synthetic shelf-life ground truth.