# YOLO Training Baseline & Reconnaissance Report

**Generated**: 2026-08-18  
**Repository**: SmartFreshAI  
**Target Module**: Detection Pipeline Reconnaissance & Baseline Evaluation  

---

## 1. Executive Summary & Findings Table

| Requirement / Query | Findings / Value | Source / Provenance |
| :--- | :--- | :--- |
| **1. Ultralytics Version** | `8.4.117` in `requirements.txt`; `8.4.120` installed at runtime | `requirements.txt:L80`, runtime inspection |
| **2. Base Architecture (`best.pt`)** | **YOLO11n** (`yolo11n.yaml` / `DetectionModel`) | `models/detection/detector/weights/best.pt` metadata |
| **3. Target Image Size (`imgsz`)** | **640** | `configs/settings.yaml:L91`, `scripts/train_detector.py` |
| **4. Training Batch Size** | **16** | `configs/settings.yaml:L90`, `scripts/train_detector.py` |
| **5. Optimizer / Hyperparameters** | `UNKNOWN — NOT RECOVERABLE FROM REPOSITORY` | Default Ultralytics auto-optimizer / hyperparameters used in `train_detector.py` |
| **6. Custom Augmentation** | `UNKNOWN — NOT RECOVERABLE FROM REPOSITORY` | Default Ultralytics training augmentations used; no custom pipeline specified |
| **7. Class Weighting** | None (Default equal weighting) | `scripts/train_detector.py` does not pass `cls` weight adjustments |
| **8. Early Stopping** | Enabled (`patience=10`) | `configs/settings.yaml:L95`, `scripts/train_detector.py:L34` |
| **9. Pretrained Weights** | Enabled (`pretrained=True`, base weights: `yolo11n.pt`) | `configs/settings.yaml:L88`, `scripts/train_detector.py:L74` |
| **10. Script Safety** | Safe & reusable via parameterized arguments | `scripts/train_detector.py` uses configurable `data_yaml` |
| **11. Evaluation Trustworthiness** | Safe & compliant; uses Ultralytics `model.val()` API | `scripts/evaluate_detector.py` |
| **12. Historical Metric Reports** | Recovered via baseline evaluation execution | `reports/detection_baseline_val.json` & `detection_baseline_test.json` |

---

## 2. Dataset Audit & Integrity Validation (V2 Baseline)

### 2.1 Split & Label Summary
- **Location**: `data/detection/` (Roboflow V2, frozen).
- **Split Breakdown**:
  - `train`: 777 images / 3,884 annotated objects
  - `valid`: 218 images / 1,105 annotated objects
  - `test`: 111 images / 612 annotated objects
  - **Total**: 1,106 images / 5,601 objects.
- **Label Integrity**: `reports/detection_dataset_validation.json` status = **`PASS`**. Zero missing image/label pairs, zero corrupt rows.

### 2.2 Class Imbalance & Duplicate Analysis
- **Class Imbalance Ratio**: **6.072** (Most represented: `Apple` with 1,184 objects; Least represented: `chickoo` with 195 objects).
- **Exact Duplicates**: 0 pairs.
- **Near-Duplicates / Cross-Split Leakage**: 0 cross-split exact duplicate pairs.

---

## 3. Verified Baseline Model Benchmark (`best.pt`)

Evaluation conducted using `scripts/evaluate_detector.py` on the frozen V2 dataset:

### 3.1 Aggregate Metrics

| Evaluation Split | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 | Report Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Validation (`valid`)** | 0.7338 | 0.6943 | **0.7100** | 0.5138 | `reports/detection_baseline_val.json` |
| **Test (`test`)** | 0.7878 | 0.6605 | **0.7155** | 0.5456 | `reports/detection_baseline_test.json` |

### 3.2 Per-Class Breakdown (Test Split)

| Class ID | Class Name | Precision | Recall | AP@50 | Weakness / Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | Apple | 0.7446 | 0.6181 | 0.5133 | Missed on shadowed or non-red varieties |
| 1 | Grape | 0.7410 | **0.2264** | **0.1722** | Severe recall failure; cluster vs berry confusion |
| 2 | Kiwi | 0.9325 | 0.5741 | 0.6654 | Moderate recall |
| 3 | Mango | 0.6206 | 0.8800 | 0.7281 | Low precision; confused with Guava/Chickoo |
| 4 | Orange | 0.8902 | 0.7679 | 0.6808 | Solid performance |
| 5 | Strawberry | 0.8074 | 0.8077 | 0.5751 | Good overall detection |
| 6 | banana | 0.8497 | 0.8483 | 0.6832 | Good overall detection |
| 7 | cherry | 0.5165 | 0.6724 | **0.3027** | Low precision & AP@50 (small red sphere confusion) |
| 8 | chickoo | 0.9802 | 0.6842 | 0.6822 | Low representation (195 objects) |
| 9 | guava | 0.7949 | 0.5263 | 0.4534 | Confused with green Apple & unripe Mango |

---

## 4. Summary & Next Steps

1. **Established Baseline**: The frozen V2 baseline benchmark is **mAP@50 = 0.7155** on `test` and **0.7100** on `valid`.
2. **Primary Weakness Identified**: Grape (AP50=17.2%), Cherry (AP50=30.3%), and Guava (AP50=45.3%) suffer from severe recall and precision deficits.
3. **Pipeline Readiness**: The dataset validation and baseline benchmark evaluation layers are fully functional and verifiable.
