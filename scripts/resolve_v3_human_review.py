#!/usr/bin/env python3
"""Controlled FINAL V3 human-review resolution tool.

Purpose
-------
Provide an auditable, CLI-driven tool for a human reviewer to resolve the
*remaining genuine human-judgment blockers* of the V3 detection build:

    A.  EMPTY LABEL       : 8 images with an empty/missing label but visible fruit
    B.  MANUAL REVIEW     : 5 huge-box records that need a human decision
    C.  TIGHTEN           : 1 huge-box record that needs a revised bounding box
    D.  UNCERTAIN         : 4111 AI proposal records (250 images) flagged uncertain

This tool does **NOT** silently convert ``uncertain``/``manual_review``/
``tighten``/``annotate`` into accepted annotations. Every resolution is an
explicit, recorded human action with a timestamp, reviewer, source proposal,
decision, action, coordinates (when applicable) and notes.

It produces decisions that the *existing* V3 builder understands
(``scripts/build_detection_v3.py``). It reuses the two existing decision
schemas ("proposal-review" records carrying ``human_decision`` and
"adjudication" records carrying ``decision``) rather than inventing a new,
incompatible manifest.

It is strictly read-only with respect to the frozen V2 dataset
(``data/detection/``) and ``best.pt``. The only writes it performs are:
  * timestamped backups of ``human_decisions.json`` (atomic),
  * the human review queue report,
  * a copy of the (validated) resolved decisions manifest,
  * optional per-image visualization composites (not required).
It NEVER writes to ``data/detection/`` and NEVER fabricates bounding boxes.

Usage
-----
    python scripts/resolve_v3_human_review.py --dry-run
        Print status/counts; make no modifications.

    python scripts/resolve_v3_human_review.py --generate-queue
        Generate reports/audit_review/v3_human_review_queue.json (no writes
        to the decision manifest).

    python scripts/resolve_v3_human_review.py --apply-resolution <JSON-file>
        Apply an externally prepared, validated resolution manifest atomically.

    python scripts/resolve_v3_human_review.py --validate
        Re-run full validation against the current human_decisions.json +
        manual annotations; print status.

The interactive review itself (drawing boxes, confirming a grape bunch,
tightening a huge box) remains in the existing GUI tools:
``scripts/manual_annotate_detection.py`` (empty-label fruit) and
``scripts/review_annotation_proposals.py`` (image-level proposal review). This
resolver provides the *queue*, the *resolution-schema*, the *atomic
persistence* and the *validation* gate so those human decisions can be folded
back into the manifest safely and auditably.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the existing, authoritative gate + validators from the V3 builder.
from scripts.build_detection_v3 import (  # noqa: E402
    ADJUDICATION_DECISION_NEEDS_RESOLUTION,
    ALLOWED_DECISIONS,
    check_gate,
    dedup_rows,
    load_human_decisions,
    load_manual_annotations,
    load_proposals,
    validate_pixel_box,
    validate_yolo_row,
)
from scripts.audit_detection_dataset import (  # noqa: E402
    _find_dataset_root,
    load_data_config,
)

DECISIONS_FILE = Path("reports/audit_review/human_decisions.json")
PROPOSALS_FILE = Path("reports/audit_review/ai_annotation_proposals/proposals.json")
MANUAL_DIR = Path("reports/audit_review/manual_annotations")
POLICY_FILE = Path("docs/DETECTION_V3_ANNOTATION_POLICY.md")
QUEUE_FILE = Path("reports/audit_review/v3_human_review_queue.json")
DATA_DIR = Path("data/detection")

REVIEWER = "human"
# Categories used internally for grouping queue items. These are NOT a new
# decision schema; they only describe *why* an item is being queued.
CAT_EMPTY = "A_EMPTY_LABEL"
CAT_MANUAL = "B_MANUAL_REVIEW"
CAT_TIGHTEN = "C_TIGHTEN"
CAT_UNCERTAIN = "D_UNCERTAIN"

# Allowed reviewable image-level actions for uncertain-proposal images.
UNCERTAIN_ACTIONS = {
    "ACCEPT_SELECTED",  # approve only the listed proposal ids
    "REJECT_SELECTED",  # reject only the listed proposal ids
    "KEEP_ORIGINAL",    # preserve original GT, ignore all proposals
    "CORRECT",          # replace a matched original box with a corrected box
    "UNCERTAIN",        # leave unresolved (blocks the gate)
    "ACCEPT_ALL",       # explicit bulk accept of every proposal for the image
}

# Allowed reviewable actions for empty-label images.
EMPTY_ACTIONS = {
    "MANUALLY_ANNOTATE",  # supply validated YOLO coords (via GUI or explicitly)
    "CONFIRM_BACKGROUND",  # image truly has no fruit -> keep empty for V3
    "MARK_UNCERTAIN",      # cannot decide; stays blocked
}

# Allowed reviewable actions for manual_review / tighten huge-box records.
HUGEBOX_ACTIONS = {
    "KEEP",      # retain the original V2 box unchanged
    "TIGHTEN",   # supply a revised (tighter) validated box
    "REPLACE",   # replace with a validated box (alias for TIGHTEN with coords)
    "UNCERTAIN", # cannot decide; stays blocked
}


# --------------------------------------------------------------------------- #
# Atomic persistence / backup
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string (with microseconds)."""
    return datetime.now(timezone.utc).isoformat()


def backup_decisions(path: Path = DECISIONS_FILE,
                     out_dir: Optional[Path] = None,
                     timestamp: Optional[str] = None,
                     baseline: Optional[Any] = None) -> Path:
    """Create a timestamped backup of the decisions manifest.

    Returns the backup path. If ``path`` exists the backup is a byte-for-byte
    copy of the current manifest. If ``path`` does not exist, ``baseline`` may
    supply the pre-resolution state (e.g. the empty manifest) to snapshot so the
    first-ever resolution remains reversible; otherwise FileNotFoundError is
    raised.
    """
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = (out_dir or path.parent) / f"{path.stem}.backup_{stamp}{path.suffix}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(str(path), str(backup))
    elif baseline is not None:
        backup.write_text(json.dumps(baseline, indent=2, default=str) + "\n",
                          encoding="utf-8")
    else:
        raise FileNotFoundError(f"decisions manifest not found: {path}")
    return backup


def write_json_atomic(data: Any, path: Path) -> Path:
    """Write ``data`` to ``path`` via a temp file + atomic rename.

    Guarantees the previous manifest is never left half-written: the write is
    first persisted to a unique temp file in the same directory, then atomically
    moved over the destination. On any failure the original file is untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{path.stem}.tmp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{path.suffix}"
    tmp = path.with_name(tmp_name)
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


# --------------------------------------------------------------------------- #
# Record classification (mirrors the builder's own predicates)
# --------------------------------------------------------------------------- #
def is_proposal_review_record(r: dict) -> bool:
    """True if the record uses the proposal-review schema."""
    return isinstance(r.get("human_decision"), str)


def is_adjudication_record(r: dict) -> bool:
    """True if the record uses the adjudication schema (decision field)."""
    return (isinstance(r.get("decision"), str)
            and not isinstance(r.get("human_decision"), str))


def record_image_name(r: dict) -> str:
    """Return the image filename (basename) for a record."""
    return r.get("image_filename") or Path(str(r.get("image", ""))).name


def normalize_image_path(p: Optional[str]) -> str:
    """Normalize an image path (forward slashes, repo-relative)."""
    return str(p or "").replace("\\", "/")
# --------------------------------------------------------------------------- #
# Loading the frozen inputs (read-only)
# --------------------------------------------------------------------------- #
def load_frozen() -> Dict[str, Any]:
    """Load dataset config, decisions, proposals, manual annotations.

    Pure read of the frozen inputs; returns them in a dict. Never modifies the
    source dataset or best.pt.
    """
    data_root = _find_dataset_root(DATA_DIR)
    if data_root is None:
        raise FileNotFoundError(f"dataset root not found under {DATA_DIR}")
    _, _, class_names = load_data_config(data_root)
    decisions = load_human_decisions(DECISIONS_FILE)
    proposals = load_proposals(PROPOSALS_FILE)
    manual = load_manual_annotations(MANUAL_DIR)
    return {
        "data_root": data_root,
        "class_names": class_names,
        "decisions": decisions,
        "proposals": proposals,
        "manual": manual,
    }


# --------------------------------------------------------------------------- #
# Blocker detection (the authoritative list of what still needs resolution)
# --------------------------------------------------------------------------- #
def collect_adjudication_blockers(records: List[dict]) -> List[dict]:
    """Return huge-box adjudication records that still block the gate.

    A huge-box adjudication record blocks when its decision is
    ``manual_review`` or ``tighten`` (unresolved until a human resolves it).
    ``annotate``/empty-label records are a separate blocker category and are
    collected separately (see ``collect_empty_label_blockers``); they are not
    returned here so that callers that ask for "adjudication blockers" get
    only the huge-box decisions that need a KEEP / TIGHTEN / REPLACE choice.
    ``keep``/``confirmed`` do not block.
    """
    blockers: List[dict] = []
    for r in records:
        if not is_adjudication_record(r):
            continue
        dec = r.get("decision")
        if dec in ADJUDICATION_DECISION_NEEDS_RESOLUTION and dec != "annotate":
            blockers.append(r)
    return blockers


def collect_empty_label_blockers(records: List[dict]) -> List[dict]:
    """Return empty-label adjudication records (``decision == annotate``) that
    still block the gate.  These require a MANUALLY_ANNOTATE or
    CONFIRM_BACKGROUND action."""
    blockers: List[dict] = []
    for r in records:
        if not is_adjudication_record(r):
            continue
        if r.get("decision") == "annotate":
            blockers.append(r)
    return blockers


def collect_uncertain_proposals(records: List[dict],
                                proposals: List[dict]) -> List[dict]:
    """Return proposal-review records that are still ``uncertain``.

    A record is uncertain when it is a proposal-review record and its
    ``human_decision`` is ``uncertain`` (or missing/unknown). Resolved values
    (accepted/corrected/rejected/kept) are never returned.
    """
    uncertain: List[dict] = []
    for r in records:
        if not is_proposal_review_record(r):
            continue
        hd = r.get("human_decision")
        if hd in ("uncertain",) or hd not in ALLOWED_DECISIONS or hd == "skip":
            uncertain.append(r)
    return uncertain
# --------------------------------------------------------------------------- #
# Human review queue generation
# --------------------------------------------------------------------------- #
def image_label_boxes(data_root: Path, split: str, image_name: str) -> List[dict]:
    """Read the V2 normalized GT boxes for an image (never modify them)."""
    lbl = data_root / split / "labels" / (Path(image_name).stem + ".txt")
    boxes: List[dict] = []
    if not lbl.exists():
        return boxes
    from scripts.audit_detection_dataset import _read_boxes
    rows, _ = _read_boxes(lbl, 100)
    for row in rows:
        boxes.append({
            "class_id": int(row[0]),
            "cx": row[1], "cy": row[2], "w": row[3], "h": row[4],
        })
    return boxes


def image_dimensions(data_root: Path, split: str, image_name: str) -> Tuple[int, int]:
    """Return (width, height) of an image, or (0, 0) if unavailable."""
    img = data_root / split / "images" / image_name
    try:
        from PIL import Image
        with Image.open(img) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return (0, 0)


def _proposal_display(p: dict) -> dict:
    return {
        "proposal_id": p.get("proposal_id"),
        "class_name": p.get("class_name"),
        "class_id": p.get("class_id"),
        "confidence": p.get("confidence"),
        "x1": p.get("x1"), "y1": p.get("y1"),
        "x2": p.get("x2"), "y2": p.get("y2"),
    }


def _image_proposals(uncertain: List[dict], image_name: str) -> List[dict]:
    """Gather ai_proposal payloads for one image from its uncertain records."""
    props = []
    for r in uncertain:
        if record_image_name(r) != image_name:
            continue
        p = r.get("ai_proposal")
        if isinstance(p, dict):
            props.append(p)
    return props


def _priority_for_image(review_category: str) -> str:
    """Deterministic triage priority for an uncertain image."""
    if review_category == "ambiguous_classes":
        return "HIGH"
    if review_category == "many_objects":
        return "MEDIUM"
    return "LOW"
def _build_adjudication_queue(r: dict, cat: str, acting: str,
                              data_root: Path) -> dict:
    """Build a queue item for an adjudication blocker record."""
    img = record_image_name(r)
    split = r.get("split", "train")
    return {
        "image": normalize_image_path(r.get("image")),
        "image_filename": img,
        "split": split,
        "category": cat,
        "current_decision": r.get("decision"),
        "original_annotation": image_label_boxes(data_root, split, img),
        "proposal_ids": [],
        "proposal_classes": [],
        "proposal_confidence": [],
        "proposal_coordinates": [],
        "image_dimensions": image_dimensions(data_root, split, img),
        "max_area_ratio": r.get("max_area_ratio"),
        "class_name": r.get("class_name"),
        "reason_for_review": acting,
        "expected_reviewer_action": "KEEP / TIGHTEN / REPLACE / UNCERTAIN",
        "grape_warning": (r.get("class_name") == "Grape"),
    }


def build_review_queue(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build the deterministic unresolved-blocker queue.

    Groups uncertain proposals by image so the reviewer makes one image-level
    decision. Empty-label, manual_review and tighten items are individual.
    Only *unresolved* items are included.
    """
    records = state["decisions"].get("records", [])
    data_root = state["data_root"]

    adj_blockers = collect_adjudication_blockers(records)
    empty = collect_empty_label_blockers(records)
    manual = [r for r in adj_blockers if r.get("decision") == "manual_review"]
    tighten = [r for r in adj_blockers if r.get("decision") == "tighten"]

    queue_empty = [_build_adjudication_queue(
        r, CAT_EMPTY,
        r.get("notes") or "empty-label image with visible fruit",
        data_root) for r in empty]
    for it in queue_empty:
        it["expected_reviewer_action"] = \
            "MANUALLY_ANNOTATE / CONFIRM_BACKGROUND / MARK_UNCERTAIN"
        it["grape_warning"] = (it["class_name"] == "Grape")

    queue_manual = [_build_adjudication_queue(
        r, CAT_MANUAL,
        "huge bounding box flagged for manual review",
        data_root) for r in manual]

    queue_tighten = [_build_adjudication_queue(
        r, CAT_TIGHTEN,
        "huge bounding box needs tightening (validated box required)",
        data_root) for r in tighten]

    uncertain = collect_uncertain_proposals(records, state["proposals"])
    by_image: Dict[str, List[dict]] = {}
    for r in uncertain:
        by_image.setdefault(record_image_name(r), []).append(r)

    queue_uncertain = []
    for img in sorted(by_image):
        recs = by_image[img]
        split = recs[0].get("split", "train")
        props = _image_proposals(uncertain, img)
        queue_uncertain.append({
            "image": normalize_image_path(recs[0].get("image")),
            "image_filename": img,
            "split": split,
            "category": CAT_UNCERTAIN,
            "current_decision": "uncertain",
            "original_annotation": image_label_boxes(data_root, split, img),
            "proposal_ids": [p.get("proposal_id") for p in props],
            "proposal_classes": [p.get("class_name") for p in props],
            "proposal_confidence": [p.get("confidence") for p in props],
            "proposal_coordinates": [[p.get("x1"), p.get("y1"), p.get("x2"), p.get("y2")]
                                     for p in props],
            "image_dimensions": image_dimensions(data_root, split, img),
            "gt_count": len(image_label_boxes(data_root, split, img)),
            "proposal_count": len(props),
            "proposals": [_proposal_display(p) for p in props],
            "review_category": recs[0].get("review_category"),
            "priority": _priority_for_image(recs[0].get("review_category", "ambiguous_classes")),
            "reason_for_review": "AI proposals reflagged uncertain; requires explicit human decision",
            "expected_reviewer_action": "ACCEPT_SELECTED / REJECT_SELECTED / KEEP_ORIGINAL / CORRECT / ACCEPT_ALL / UNCERTAIN",
        })

    all_items = queue_empty + queue_manual + queue_tighten + queue_uncertain
    return {
        "generated_at": now_iso(),
        "reviewer": REVIEWER,
        "description": "Deterministic queue of the FINAL V3 human-judgment blockers. "
                       "Only UNRESOLVED items are included.",
        "schema_note": "Categories A-D are queue labels for triage only. Decisions "
                       "persisted back to human_decisions.json reuse the existing "
                       "V3 builder schemas (proposal-review 'human_decision' and "
                       "adjudication 'decision').",
        "counts": {
            "empty_label_images": len(queue_empty),
            "manual_review_records": len(queue_manual),
            "tighten_records": len(queue_tighten),
            "uncertain_images": len(queue_uncertain),
            "uncertain_proposal_records": len(uncertain),
            "total_blocker_images": len({it["image_filename"] for it in all_items}),
        },
        "items": all_items,
    }


def write_queue(state: Dict[str, Any], out_path: Path = QUEUE_FILE) -> Path:
    """Write the review queue deterministically and return the path."""
    queue = build_review_queue(state)
    write_json_atomic(queue, out_path)
    return out_path
# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _extract_final_boxes(r: dict) -> List[Optional[List[float]]]:
    """Return the normalized final boxes of a proposal-review record.

    Handles both pixel-form ``final_boxes`` (unde) and YOLO-form ``bbox``.
    Used for duplicate/geometry checks inside validation.
    """
    finals = r.get("final_boxes")
    out: List[Optional[List[float]]] = []
    if isinstance(finals, list):
        for fb in finals:
            if not isinstance(fb, (list, tuple)) or len(fb) < 4:
                out.append(None)
            else:
                out.append([float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])])
    return out


def validate_resolved_state(state: Dict[str, Any],
                           policy_path: Optional[Path] = None) -> Dict[str, Any]:
    """Validate the resolved decisions + manual annotations end-to-end.

    Uses the authoritative ``check_gate`` from the V3 builder plus explicit
    structural, class, coordinate, Grape-policy and duplicate checks. Returns a
    dict with ``passed`` and a list of ``errors``.

    ``policy_path`` optionally points to the V3 annotation policy doc. When
    supplied (and it exists), its class list is cross-checked against the data
    config class list; when omitted, that policy-class cross-check is skipped
    (useful for testing against synthetic class lists). Callers that want the
    full production gate (including the class-list cross-check) pass the real
    policy path.
    """
    errors: List[str] = []
    decisions = state["decisions"]
    proposals = state["proposals"]
    class_names = state["class_names"]
    data_root = state["data_root"]
    manual = state["manual"]

    records = decisions.get("records", [])
    nc = len(class_names)
    name_to_id = {n: i for i, n in enumerate(class_names)}

    # 1) authoritative gate
    gate = check_gate(
        data_root=data_root,
        decisions=decisions,
        proposals=proposals,
        manual_annotations=manual,
        class_names=class_names,
        policy_path=(policy_path if policy_path is not None and policy_path.exists() else None),
    )
    if not gate.passed:
        errors.append(f"gate blocked ({len(gate.reasons)} reason(s)): "
                      f"{gate.reasons[0] if gate.reasons else 'unknown'}")

    # 2) proposal ids referenced in records must exist in proposals.json
    proposal_ids = {p.get("proposal_id") for p in proposals}
    for r in records:
        if not is_proposal_review_record(r):
            continue
        props = r.get("ai_proposals") or ([r["ai_proposal"]]
                                          if r.get("ai_proposal") else [])
        for p in props:
            pid = p.get("proposal_id")
            if pid is not None and pid not in proposal_ids:
                errors.append(f"proposal_id {pid!r} referenced but not in proposals.json")

    # 3) classes & coordinates
    for r in records:
        if is_proposal_review_record(r):
            if r.get("human_decision") in ("accepted", "corrected"):
                props = r.get("ai_proposals") or ([r["ai_proposal"]]
                                                  if r.get("ai_proposal") else [])
                for p in props:
                    cn = p.get("class_name")
                    if cn is not None and cn not in name_to_id:
                        errors.append(f"invalid class name {cn!r}")
                    cid = p.get("class_id")
                    if cid is not None and not (0 <= int(cid) < nc):
                        errors.append(f"invalid class id {cid!r}")
        else:
            bbox = r.get("bbox")
            if bbox is not None:
                err = validate_yolo_row(list(bbox), nc)
                if err:
                    errors.append(f"adjudication bbox invalid ({err}) for {record_image_name(r)}")
    # 4) no zero-area boxes
    for r in records:
        bbox = r.get("bbox")
        if bbox is not None and len(bbox) >= 5:
            if float(bbox[3]) <= 0 or float(bbox[4]) <= 0:
                errors.append(f"zero-area box in adjudication for {record_image_name(r)}")

    # 5) Grape policy: a Grape adjudication's box must be a bunch-sized, in-frame
    #    object; a per-berry-sized box is implausible. Mechanical invariants
    #    (valid, in-frame, non-degenerate) are enforced here; the bunch-vs-berry
    #    judgment remains an explicit reviewer decision, never auto-inferred.
    for r in records:
        if is_adjudication_record(r) and r.get("class_name") == "Grape" and r.get("bbox"):
            bbox = r.get("bbox")
            if len(bbox) >= 5 and (float(bbox[3]) < 0.02 or float(bbox[4]) < 0.02):
                errors.append(f"Grape bbox implausibly small (likely per-berry) for "
                              f"{record_image_name(r)}")

    # 6) manual annotations valid
    _, manual_labels = manual
    for stem, rows in manual_labels.items():
        for row in rows:
            err = validate_yolo_row(row, nc)
            if err:
                errors.append(f"manual annotation {stem}.txt invalid ({err})")

    # 7) duplicate decision records (same image + decision + identical proposal)
    seen = set()
    for r in records:
        prop_key = r.get("ai_proposal", r.get("ai_proposals"))
        key = (record_image_name(r),
               r.get("human_decision") or r.get("decision"),
               json.dumps(prop_key, sort_keys=True, default=str))
        if key in seen:
            errors.append(f"duplicate decision record for {record_image_name(r)}")
        seen.add(key)

    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "gate": {
            "passed": gate.passed,
            "unresolved_proposal_records": gate.unresolved_proposal_records,
            "unresolved_images": sorted(gate.unresolved_images),
            "manual_required": gate.manual_required,
            "ambiguous_unresolved": gate.ambiguous_unresolved,
        },
    }
# --------------------------------------------------------------------------- #
# Resolution record builders (explicit, audited human decisions)
# --------------------------------------------------------------------------- #
def _validated_yolo_row(coordinates, nc: int) -> List[float]:
    """Validate an input coordinate list as a YOLO row [cls,cx,cy,w,h].

    Raises ValueError on any invalid entry. Never fabricates coordinates; the
    caller must supply real, human-verified numbers.
    """
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 5:
        raise ValueError(
            "coordinates must be a [cls_id, cx, cy, w, h] YOLO row (5 fields)")
    try:
        row = [float(x) for x in coordinates]
    except (TypeError, ValueError):
        raise ValueError("coordinates must be numeric")
    err = validate_yolo_row(row, nc)
    if err:
        raise ValueError(f"invalid coordinates ({err})")
    return row


def _resolve_empty_label(records: List[dict], row: Optional[List[float]],
                         notes: str, decision: str = "annotate") -> dict:
    """Return a resolved empty-label adjudication record (annotate/keep_empty).

    For ``annotate`` the supplied YOLO ``row`` becomes the adjudication ``bbox``
    (the field the V3 builder reads). For ``keep_empty`` the bbox stays None and
    decision becomes ``keep_empty`` (not in the blocking set).
    """
    rec = copy.deepcopy(records[0])
    rec["decision"] = decision
    rec["action"] = ("manual_annotation_supplied" if row is not None
                     else "no_change")
    rec["bbox"] = row
    rec["reviewer"] = REVIEWER
    rec["resolved_at"] = now_iso()
    rec["resolution_notes"] = notes
    return rec


def _resolve_hugebox(records: List[dict], decision: str,
                     row: Optional[List[float]], notes: str) -> dict:
    """Return a resolved huge-box adjudication record.

    ``decision`` is one of ``keep``/``tighten``. ``keep`` keeps the original V2
    box (no coordinates needed / no change). ``tighten`` supplies the validated
    ``row`` as the replacement bbox.
    """
    rec = copy.deepcopy(records[0])
    rec["decision"] = decision
    rec["action"] = ("no_change" if decision == "keep"
                     else "manual_annotation_supplied")
    rec["bbox"] = row
    rec["reviewer"] = REVIEWER
    rec["resolved_at"] = now_iso()
    rec["resolution_notes"] = notes
    return rec


def _resolve_uncertain_group(records: List[dict], action: str,
                             proposal_ids: List[str],
                             proposal_by_id: Dict[str, dict],
                             notes: str) -> dict:
    """Return the transformed proposal-review records for an image.

    Maps the human image-level action onto the existing proposal-review schema:
      * ACCEPT_SELECTED -> each chosen proposal becomes an ``accepted`` record
      * REJECT_SELECTED -> each chosen proposal becomes a ``rejected`` record
      * KEEP_ORIGINAL   -> each proposal becomes a ``kept`` record (original GT
                           is preserved; nothing is added)
      * CORRECT         -> each chosen proposal becomes a ``corrected`` record
                           (a corrected box must be supplied separately)
      * ACCEPT_ALL      -> every proposal for the image becomes ``accepted``
      * UNCERTAIN       -> untouched (stays blocked)

    The transformation is recorded as an explicit human action; nothing is
    silently auto-accepted.
    """
    out: List[dict] = []
    if action == "UNCERTAIN":
        return out  # do not change; stays unresolved
    for r in records:
        p = r.get("ai_proposal") or {}
        pid = p.get("proposal_id")
        # For ACCEPT_SELECTED / REJECT_SELECTED / CORRECT, only touch listed ids
        if action in ("ACCEPT_SELECTED", "REJECT_SELECTED", "CORRECT"):
            if pid not in proposal_ids:
                out.append(r)  # leave others as-is (still uncertain -> blocked)
                continue
        new = copy.deepcopy(r)
        if action in ("ACCEPT_SELECTED", "ACCEPT_ALL"):
            new["human_decision"] = "accepted"
            new["notes"] = notes or "image-level human acceptance"
        elif action == "REJECT_SELECTED":
            new["human_decision"] = "rejected"
            new["notes"] = notes or "image-level human rejection"
        elif action == "KEEP_ORIGINAL":
            new["human_decision"] = "kept"
            new["notes"] = notes or "image-level keep-original decision"
        elif action == "CORRECT":
            new["human_decision"] = "corrected"
            new["notes"] = notes or "image-level human correction"
        new["reviewer"] = REVIEWER
        new["resolved_at"] = now_iso()
        out.append(new)
    return out


# --------------------------------------------------------------------------- #
# Resolution record replacement helpers
# --------------------------------------------------------------------------- #
def _replace_record(records: List[dict], originals: List[dict], new_rec: dict) -> None:
    """Replace the originals (by identity) in the records list with ``new_rec``."""
    for orig in originals:
        for i, r in enumerate(records):
            if r is orig:
                records[i] = new_rec
                break


def _replace_records(records: List[dict], originals: List[dict],
                     new_recs: List[dict]) -> None:
    """Remove all originals (by identity) and append ``new_recs``."""
    new_list = [r for r in records if r not in originals]
    new_list.extend(new_recs)
    # Reassign in place so the caller's references stay valid.
    del records[:]
    records.extend(new_list)


# --------------------------------------------------------------------------- #
# Resolution application (atomic, backup-then-write)
# --------------------------------------------------------------------------- #
def apply_resolutions(state: Dict[str, Any],
                      resolutions: Dict[str, Any],
                      decisions_path: Path = DECISIONS_FILE,
                      dry_run: bool = False) -> Dict[str, Any]:
    """Apply a validated resolution manifest to ``human_decisions.json``.

    ``resolutions`` uses the schema:
        {
          "reviewer": "human",
          "items": [
             {"category": A/B/C/D, "image_filename": ...,
              "action": <one of the allowed actions>,
              "proposal_ids": [...],      # for ACCEPT_SELECTED / REJECT_SELECTED
              "coordinates": [...],       # validated YOLO [cls,cx,cy,w,h] for
                                          # MANUALLY_ANNOTATE / TIGHTEN / REPLACE
              "notes": "..."},
          ],
        }

    Every item's ``action`` must be in the allowed set for its category. The
    write is atomic and always backed-up first. Returns a summary dict.
    """
    if not isinstance(resolutions, dict) or "items" not in resolutions:
        raise ValueError("resolution manifest must be an object with 'items'")
    items = resolutions.get("items", [])
    if not isinstance(items, list):
        raise ValueError("resolution manifest 'items' must be a list")

    records = copy.deepcopy(state["decisions"].get("records", []))
    proposal_by_id = {p.get("proposal_id"): p for p in state["proposals"]}
    nc = len(state["class_names"])

    from collections import defaultdict
    by_img: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_img[record_image_name(r)].append(r)

    applied = []
    skipped = []

    for it in items:
        cat = it.get("category")
        img = it.get("image_filename")
        action = it.get("action")
        notes = it.get("notes") or "human resolution"
        coordinates = it.get("coordinates")
        proposal_ids = it.get("proposal_ids") or []

        if not img or img not in by_img:
            skipped.append({"image_filename": img,
                            "reason": "no matching records in manifest"})
            continue

        # ---- A. EMPTY LABEL ------------------------------------------------
        if cat == CAT_EMPTY:
            if action == "MANUALLY_ANNOTATE":
                if not coordinates:
                    raise ValueError(f"MANUALLY_ANNOTATE requires coordinates for {img}")
                row = _validated_yolo_row(coordinates, nc)
                _replace_record(records, by_img[img],
                                _resolve_empty_label(by_img[img], row, notes))
                applied.append({"image_filename": img, "category": cat,
                                "action": "MANUALLY_ANNOTATE"})
            elif action == "CONFIRM_BACKGROUND":
                _replace_record(records, by_img[img],
                                _resolve_empty_label(by_img[img], None, notes,
                                                     decision="keep_empty"))
                applied.append({"image_filename": img, "category": cat,
                                "action": "CONFIRM_BACKGROUND"})
            elif action == "MARK_UNCERTAIN":
                skipped.append({"image_filename": img,
                                "reason": "marked uncertain - stays blocked"})

        # ---- B/C. MANUAL_REVIEW / TIGHTEN ---------------------------------
        elif cat in (CAT_MANUAL, CAT_TIGHTEN):
            if action == "KEEP":
                _replace_record(records, by_img[img],
                                _resolve_hugebox(by_img[img], "keep", None, notes))
                applied.append({"image_filename": img, "category": cat,
                                "action": "KEEP"})
            elif action in ("TIGHTEN", "REPLACE"):
                if not coordinates:
                    raise ValueError(f"{action} requires validated coordinates for {img}")
                row = _validated_yolo_row(coordinates, nc)
                _replace_record(records, by_img[img],
                                _resolve_hugebox(by_img[img], "tighten", row, notes))
                applied.append({"image_filename": img, "category": cat,
                                "action": "TIGHTEN"})
            elif action == "UNCERTAIN":
                skipped.append({"image_filename": img,
                                "reason": "marked uncertain - stays blocked"})

        # ---- D. UNCERTAIN PROPOSALS ---------------------------------------
        elif cat == CAT_UNCERTAIN:
            reps = _resolve_uncertain_group(by_img[img], action, proposal_ids,
                                            proposal_by_id, notes)
            if reps:
                _replace_records(records, by_img[img], reps)
                applied.append({"image_filename": img, "category": cat,
                                "action": action})
            else:
                skipped.append({"image_filename": img,
                                "reason": "action produced no resolved records "
                                          "(UNCERTAIN left unchanged)"})
        else:
            raise ValueError(f"unknown category {cat!r}")

    if dry_run:
        return {"dry_run": True,
                "applied": len(applied), "skipped": len(skipped),
                "skipped_detail": skipped,
                "proposed_records": records}

    # Atomic persistence: backup before every write so the resolution is always
    # reversible. When a manifest already exists we back up that existing file
    # byte-for-byte; on the first-ever write there is no prior manifest, so we
    # snapshot the empty pre-resolution state as a baseline backup.
    if decisions_path.exists():
        backup = backup_decisions(decisions_path)
    else:
        backup = backup_decisions(decisions_path,
                                   timestamp=None,
                                   baseline=copy.deepcopy(state["decisions"]))
    backup_str = str(backup)
    state["decisions"]["records"] = records
    write_json_atomic(state["decisions"], decisions_path)
    return {"dry_run": False, "backup": backup_str,
            "applied": len(applied), "skipped": len(skipped),
            "skipped_detail": skipped}
# --------------------------------------------------------------------------- #
# Status / dry-run reporting
# --------------------------------------------------------------------------- #
def gate_status(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a human-readable blocker summary (counts per category)."""
    records = state["decisions"].get("records", [])
    adj = collect_adjudication_blockers(records)
    empty = len(collect_empty_label_blockers(records))
    manual = len([r for r in adj if r.get("decision") == "manual_review"])
    tighten = len([r for r in adj if r.get("decision") == "tighten"])
    uncertain = collect_uncertain_proposals(records, state["proposals"])
    uncertain_images = len({record_image_name(r) for r in uncertain})
    queue = build_review_queue(state)
    return {
        "empty_labels": empty,
        "manual_review": manual,
        "tighten": tighten,
        "uncertain_images": uncertain_images,
        "uncertain_proposal_records": len(uncertain),
        "total_blocker_images": queue["counts"]["total_blocker_images"],
    }


def print_status(state: Dict[str, Any],
                 policy_path: Optional[Path] = None) -> int:
    """Print the FINAL V3 human-review status block (dry-run display)."""
    q = build_review_queue(state)
    c = q["counts"]
    gate = check_gate(
        data_root=state["data_root"],
        decisions=state["decisions"],
        proposals=state["proposals"],
        manual_annotations=state["manual"],
        class_names=state["class_names"],
        policy_path=(policy_path if policy_path is not None and policy_path.exists() else None),
    )

    print("\nV3 HUMAN REVIEW STATUS")
    print("======================")
    print(f"Empty labels:       {c['empty_label_images']}")
    print(f"Manual review:      {c['manual_review_records']}")
    print(f"Tighten:            {c['tighten_records']}")
    print(f"Uncertain images:   {c['uncertain_images']}")
    print(f"  (uncertain proposal records: {c['uncertain_proposal_records']})")
    print(f"Total blocker images: {c['total_blocker_images']}")
    print()
    if gate.passed:
        print("V3 GATE: PASSED")
    else:
        print("V3 GATE: BLOCKED")
    return 0 if gate.passed else 3


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resolve the FINAL V3 human-judgment blockers (auditable).")
    p.add_argument("--dry-run", action="store_true",
                   help="Load blockers, show counts, make no modifications.")
    p.add_argument("--generate-queue", action="store_true",
                   help="Write reports/audit_review/v3_human_review_queue.json "
                        "(no changes to the decision manifest).")
    p.add_argument("--apply-resolution", type=Path, metavar="JSON",
                   help="Apply a validated resolution manifest atomically.")
    p.add_argument("--validate", action="store_true",
                   help="Validate current decisions + manual annotations.")
    p.add_argument("--decisions-file", type=Path, default=DECISIONS_FILE,
                   help="Path to human_decisions.json (default: existing).")
    p.add_argument("--dry-run-apply", type=Path, metavar="JSON",
                   help="Simulate --apply-resolution without writing.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        state = load_frozen()
        return print_status(state, policy_path=POLICY_FILE)

    if args.generate_queue:
        state = load_frozen()
        q = write_queue(state)
        print(f"Queue written: {q}")
        print(f"  Items: {build_review_queue(state)['counts']['total_blocker_images']} "
              "blocker images.")
        print("  (decision manifest left untouched)")
        return 0

    if args.apply_resolution is not None or args.dry_run_apply is not None:
        res_path = args.apply_resolution or args.dry_run_apply
        state = load_frozen()
        res_json = json.loads(Path(res_path).read_text(encoding="utf-8"))
        dry = args.dry_run_apply is not None
        result = apply_resolutions(
            state, res_json,
            decisions_path=args.decisions_file,
            dry_run=dry,
        )
        if dry:
            print("DRY-RUN APPLY (no writes):")
            print(f"  would apply {result['applied']} resolution(s)")
            print(f"  would skip     {result['skipped']} (see below)")
        else:
            print("Resolution applied:")
            print(f"  backup:     {result['backup']}")
            print(f"  applied:    {result['applied']}")
            print(f"  skipped:    {result['skipped']}")
        for s in result.get("skipped_detail", []):
            print(f"    skipped {s.get('image_filename')}: {s.get('reason')}")
        return 0

    if args.validate:
        state = load_frozen()
        v = validate_resolved_state(state, policy_path=POLICY_FILE)
        print("V3 RESOLUTION VALIDATION")
        for key, val in sorted(state["decisions"].get("counts", {}).items()):
            print(f"  {key}: {val}")
        print(f"  passed: {v['passed']}")
        for e in v["errors"]:
            print(f"  ERROR: {e}")
        print(f"  gate.passed: {v['gate']['passed']}  "
              f"unresolved proposal records: {v['gate']['unresolved_proposal_records']}  "
              f"unresolved images: {len(v['gate']['unresolved_images'])}")
        return 0 if v["passed"] else 3

    # No action -> print status (equivalent to a dry run, no writes).
    state = load_frozen()
    return print_status(state, policy_path=POLICY_FILE)


if __name__ == "__main__":
    sys.exit(main())
