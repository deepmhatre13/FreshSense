# SmartFreshAI Detection V3 — Annotation Policy

This document defines the annotation policy for building the **V3 detection
dataset** (`data/detection_v3/`). It is derived from the Dataset-V3 audit
(`reports/detection_dataset_audit.{json,md}`), the annotation review
(`reports/audit_review/`), and the intended **production use case**: fruit-level
detection for a live webcam application.

The policy is deliberately consistent with the current SmartFreshAI detector and
with the V2 class list:

**10 classes:** `Apple`, `Grape`, `Kiwi`, `Mango`, `Orange`, `Strawberry`,
`banana`, `cherry`, `chickoo`, `guava`.

> **Status:** This is the *proposed* policy that governs how the V3 builder will
> create `data/detection_v3/`. Nothing under `data/detection/` is changed by
> this document. V2 remains reproducible and untouched.

---

## 1. Object definition

An **object** is a *visually distinct fruit instance* (whole fruit, or a coherent
cluster/bunch of fruit) that a human can confidently recognise in the image.

The purpose of V3 is **fruit-level detection for live webcam usage** — identifying
and localising fruit so the downstream application knows *what fruit is present
and where it is*. It is **not** meant to count individual berries or estimate
fruit mass.

Therefore the unit of annotation is the **fruit object as seen**, not the
individual botanical unit inside a cluster.

## 2. Grape policy

Grapes are annotated with **one bounding box around each visually distinct grape
bunch/cluster**.

- Do **not** annotate each individual berry when the image contains a coherent
  bunch — even if the berries are individually visible, they are a single grape
  object for the purposes of this detector.
- Annotate multiple separate boxes **only** when the image genuinely contains
  *separate grape objects* (e.g. two distinct bunches separated in space, or
  individual floating grapes that are not part of a coherent bunch).
- The class label is always `Grape` for both bunches and detached individual
  grapes.

**Rationale:** the webcam pipeline detects fruit regions; a bunch is the natural
unit for this use case. Annotating every berry would (a) be impractical to label
correctly, (b) inflate the number of tiny boxes the model must learn, and (c) not
serve the fruit-level detection objective. This policy directly addresses the
**Grape annotation-policy inconsistency** found by the audit (Grape has low test
recall: P=0.741, R=0.226, AP50=0.377).

## 3. Bounding-box rules

- Use **axis-aligned bounding boxes (AABB)** in YOLO normalized format:
  `class_id x_center y_center width height` (all in `[0,1]`).
- The box must tightly enclose the **visible extent** of the fruit object.
- The box should be as **tight** as practical — do not pad the box with large
  margins of background. A box that spans ~the whole image is a **huge box** and
  must be rejected (see Section 10).
- Boxes must have **positive, non-zero** width and height.
- Box centre + half-extents must lie within `[0,1]`; the box must not extend
  outside the image frame.

## 4. Handling partially occluded fruit

- Annotate fruit that is **partially occluded** (by hands, other fruit, leaves,
  or objects) as long as the visible portion lets a human recognise it and place
  a meaningful box.
- The bounding box covers the full fruit whenever the occlusion is only at the
  edge; otherwise cover the **visible, recognisable extent** tightly.
- **Do not** annotate a fruit that is so heavily occluded that its class cannot
  be reasonably determined, or whose extent is not distinguishable.

## 5. Handling overlapping fruit

- When two or more fruit overlap, annotate **each fruit with its own box** if the
  individual extents are visually distinguishable.
- If two fruit overlap so completely that they cannot be separated, annotate them
  as **one box** containing the combined visible mass.
- Do not draw a single sloppy box over several clearly separate objects just to
  save time.

## 6. Handling fruit touching image borders

- Fruit that touches the image border is still annotated with its **visible
  portion** only. The box must stay within `[0,1]`.
- Do not clamp a box beyond the border; extend it only to the visible edge of the
  fruit inside the frame.
- A fruit entirely outside the frame is not annotated.

## 7. Handling tiny objects

- **Tiny objects** were flagged by the audit (`<0.5%` image area, 56 cases) and
  are generally poor annotations for a webcam detector.
- A fruit smaller than ~1% of the image area is unlikely to be usable by the live
  webcam detector and should generally be **excluded from V3** unless it is
  unambiguous and clearly visible.
- If a tiny object is kept, the box must still be tight and correctly labelled.
- When in doubt, flag for human review; **do not auto-delete** and **do not
  auto-keep**.

## 8. Handling ambiguous classes

The audit flagged 200 ambiguous-class candidates (classes frequently confused:
`Apple`/`Mango`/`Cherry`/`Chickoo`/`Guava`/`Grape`).

- An ambiguous annotation must be resolved by **human review**, using the
  explicit decision schema in `scripts/adjudicate_detection_annotations.py`.
- A labelled fruit stays with its **current class** only if a human confirms the
  existing label. Otherwise it is either re-classified by a human or excluded.
- **Never** silently auto-"fix" an ambiguous class label based on heuristics.

## 9. Handling empty-label images

The audit found 8 images with empty/missing label files that contain **visible
fruit** (verified by human review). These are **not** background images.

- Such an image is given decision `annotate` +
  `manual_annotation_required` in the human-decisions manifest.
- It is **not** converted into a background/negative image.
- It is **not** auto-deleted.
- A real, human-verified bounding box must be supplied before V3 construction.
- **No bounding-box coordinates are fabricated** by any script in this phase.

## 10. What constitutes an invalid annotation

An annotation is **invalid** and must be corrected or excluded if any of the
following hold:

1. **Box geometry invalid** — NaN/negative/zero width or height; centre or
   extents outside `[0,1]`; malformed YOLO line (not exactly 5 numeric values).
2. **Class ID out of range** — `class_id` not in `[0, 9]` for the 10-class model.
3. **Wrong label content** — label file is empty or missing an object that is
   clearly present (empty-label issue); or the file references no objects when
   fruit is visible.
4. **Huge box** — a box covering ~the whole image (`area ratio >= 0.95*0.95`)
   where the object does not fill the frame. Must be reviewed: keep, tighten, or
   manual review.
5. **Tiny box** — a box smaller than ~0.5% of the image area (flagged by audit)
   that is unsuitable or unverifiable.
6. **Incorrect class** — box present but class label is wrong (ambiguous or
   mislabeled).
7. **Over-annotated cluster** — individual berries annotated inside a single
   grape bunch (violates the Grape bunch policy, Section 2). The bunch should be
   one box.
8. **Background annotated** — a box drawn with no discernible fruit.
9. **Missing image/label pairing** — an image with no label when fruit is present,
   or a label with no image.

Any of these cases is routed through the adjudication layer and requires an
explicit human decision before V3 is built. The V3 builder applies **only
validated corrections**; it must never overwrite `data/detection/`.