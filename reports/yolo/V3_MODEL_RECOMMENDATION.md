# V3 Model Recommendation Report

**Status**: PENDING — V3 production adoption not decided
**Generated**: placeholder (V3 training has not been run yet)

---

## Purpose

This report documents the **explicit rule used to decide whether the V3-trained
YOLO11n model replaces the frozen V2 baseline**. It is generated after the V3
controlled experiment completes and the V3 test metrics are available.

## Baseline (frozen YOLO11n, evaluated on the unchanged V2 TEST set)

- Precision: 0.7878
- Recall: 0.6605
- mAP50: 0.7155
- mAP50-95: 0.5456
- Per-class AP50: Apple 0.5133, Grape 0.1722, Kiwi 0.6654, Mango 0.7281,
  Orange 0.6808, Strawberry 0.5751, banana 0.6832, cherry 0.3027,
  chickoo 0.6822, guava 0.4534

## V3 metrics (fill after training + evaluation on V2 test set)

- Precision, Recall, mAP50, mAP50-95 and all ten per-class AP50 values from
  `reports/detection_v3_yolo11n_test.json`.

## Recommendation rule

**V3 replaces the baseline ONLY IF all of the following hold:**

1. mAP50 improves **meaningfully** (>= +0.01 absolute).
2. mAP50-95 does **not regress materially** (>= -0.01 absolute).
3. Overall precision and recall remain acceptable (no large drop vs baseline).
4. Critical weak classes (Grape, Cherry, Guava, Apple) do **not** regress badly.
5. **No data leakage** is detected (V3 train/valid share nothing with the V2
   test set; V2 test is byte-identical between V2 and V3).
6. The benchmark **test set is unchanged** (confirmed: 111 test images, 612
   objects, byte-identical).

Otherwise the frozen YOLO11n baseline is retained as the active model.

## Absolute / percentage deltas (V3 - Baseline)

| Metric | Baseline | V3 | Abs Delta | % Delta |
| --- | ---: | ---: | ---: | ---: |
| precision | 0.7878 | (after) | | |
| recall | 0.6605 | (after) | | |
| mAP50 | 0.7155 | (after) | | |
| mAP50-95 | 0.5456 | (after) | | |
| Apple AP50 | 0.5133 | (after) | | |
| Grape AP50 | 0.1722 | (after) | | |
| Cherry AP50 | 0.3027 | (after) | | |
| Guava AP50 | 0.4534 | (after) | | |

## Limitations

- The V3 experiment only *removes* 14 unresolved annotation blockers from
  train/valid; it does not repair them. Any accuracy change reflects the absence
  of these problematic samples during training/validation, evaluated on the
  exact same V2 test set.
- No human annotations were fabricated.