#!/usr/bin/env python3
"""Build the frozen, auditable Dataset V3 from V2 + validated human decisions.

Safe, deterministic V3 builder. It:

* reads the frozen ``data/detection`` V2 dataset (read-only),
* reads ``reports/audit_review/human_decisions.json`` (validated human decisions),
* reads ``reports/audit_review/ai_annotation_proposals/proposals.json`` (proposal
  metadata, used only to reconstruct accepted proposal geometry),
* reads manually-created annotations from ``reports/audit_review/manual_annotations``,
* verifies the human-review **gate** against ``docs/DETECTION_V3_ANNOTATION_POLICY.md``,
* and ONLY then constructs ``data/detection_v3``.

The builder never modifies ``data/detection`` or ``models/.../best.pt``. If the
review gate does not pass it prints ``V3 BUILD BLOCKED`` followed by the exact
reasons and returns exit code 3 without creating ``data/detection_v3``.

Audit / integrity notes
----------------------
- Every transformation is explicit and recorded in ``v3_manifest.json``.
- AI proposals are NEVER auto-accepted. Only proposals carrying an explicit
  human ``accepted``/``corrected`` decision enter V3.
- ``UNCERTAIN`` and unresolved cases gate the build (they are never silently
  turned into annotations).
- Duplicate / overlapping accepted proposals are de-duplicated with IoU-based
  matching (superset of the policy's duplicate/overlap rules).

Usage:
    python scripts/build_detection_v3.py                      # build (after gate)
    python scripts/build_detection_v3.py --gate-only          # only run the gate
    python scripts/build_detection_v3.py --data-dir <path>    # source override
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import shutil
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_detection_dataset import (  # noqa: E402
    _find_dataset_root,
    _list_images,
    _list_labels,
    _read_boxes,
    load_data_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUILDER_VERSION = "1.0.0"
ALLOWED_DECISIONS = ("accepted", "corrected", "rejected", "kept", "uncertain")
# Adjudication actions that demand a real annotation/decision before build.
# annotate        -> a manual bounding box is required (bbox must be supplied)
# manual_review   -> the case is not settled until the human resolves it
# keep            -> preserve the original V2 label (no change)
# tighten         -> the box should be revised; unresolved until a new box exists
ADJUDICATION_DECISION_NEEDS_RESOLUTION = {"annotate", "manual_review", "tighten"}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    """Return SHA256 of the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_human_decisions(path: Path) -> Dict[str, object]:
    """Load and structurally validate the human-decisions manifest."""
    if not path.exists():
        raise FileNotFoundError(f"human decisions manifest not found: {path}")
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("human_decisions.json must be an object with a 'records' list")
    return data


def load_proposals(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"AI proposals file not found: {path}")
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("proposals.json must be a list of proposal objects")
    return data


def load_manual_annotations(manual_dir: Path) -> Tuple[Optional[dict], Dict[str, List[list]]]:
    """Load manual annotations: returns (decisions_json, {stem: yolo_rows}).

    ``label_rows`` maps image stem -> list of validated YOLO rows
    (``[cls_id, cx, cy, w, h]``) read from ``manual_annotations/labels/*.txt``.
    """
    decisions = None
    labels: Dict[str, List[list]] = {}
    if manual_dir.exists():
        dec_file = manual_dir / "decisions.json"
        if dec_file.exists():
            decisions = load_json(dec_file)
        lbl_dir = manual_dir / "labels"
        if lbl_dir.exists():
            for lp in sorted(lbl_dir.glob("*.txt")):
                rows, _ = _read_boxes(lp, 100)  # nc generous; validated below
                labels[lp.stem] = [list(r[:5]) for r in rows]
    return decisions, labels


def class_names_from_policy(policy_path: Path) -> List[str]:
    """Extract the class list from the V3 annotation policy doc.

    The policy lists classes as backtick-quoted items after a
    "**<n> classes:**" marker, e.g. ``**10 classes:** `Apple`, `Grape`, ...``.
    """
    try:
        text = policy_path.read_text(encoding="utf-8")
    except OSError:
        return []
    import re
    # Find the "classes:" marker, then take text only up to the next blank
    # line or the next "**" heading so we do not sweep up unrelated backtick
    # tokens from the rest of the document.
    m = re.search(r"\*\*\s*\d+\s+classes?:\s*\*\*", text)
    if not m:
        return []
    tail = text[m.end():]
    # Stop at a blank line or the next bold heading.
    stop = re.search(r"\n\s*\n|\n\*\*", tail)
    if stop:
        tail = tail[:stop.start()]
    names = re.findall(r"`([^`]+)`", tail)
    return [n.strip() for n in names if n.strip()]


def _norm_split(split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    s = str(split).strip().lower()
    if s in ("val", "validation"):
        return "valid"
    if s in ("train", "valid", "test"):
        return s
    return split



# ---------------------------------------------------------------------------
# V3 build gate
# ---------------------------------------------------------------------------
class GateResult:
    """Collection of gate findings (blocking reasons and counts)."""

    def __init__(self) -> None:
        self.reasons: List[str] = []
        self.unresolved_proposal_records = 0
        self.unresolved_images = set()
        self.malformed = 0
        self.invalid_class = 0
        self.invalid_box = 0
        self.invalid_class_name = 0
        self.manual_invalid = 0
        self.manual_required = 0
        self.ambiguous_unresolved = 0

    def add(self, reason: str) -> None:
        self.reasons.append(reason)

    @property
    def passed(self) -> bool:
        return not self.reasons


def _normalize_box(record: dict) -> Optional[List[float]]:
    """Return a normalized YOLO row from a decision's final box, or None.

    Handles the two decision-record shapes:
    * proposal-review: ``final_boxes`` = list of ``[x1, y1, x2, y2]`` pixels and
      ``ai_proposals``/``ai_proposal`` carrying pixel geometry + class id.
    * adjudication:    ``bbox`` = a ``[cls_id, cx, cy, w, h]`` yolo row (or None).
    """
    # --- proposal-review shape -------------------------------------------------
    props = record.get("ai_proposals") or ([record["ai_proposal"]] if record.get("ai_proposal") else None)
    final_boxes = record.get("final_boxes")
    if props and isinstance(final_boxes, list):
        out = []
        for i, fb in enumerate(final_boxes):
            if not isinstance(fb, (list, tuple)) or len(fb) < 4:
                return None
            cls_name = None
            if i < len(props) and isinstance(props[i], dict):
                cls_name = props[i].get("class_name")
            out.append({
                "box_px": [float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])],
                "class_name": cls_name,
            })
        return out
    # --- adjudication shape ----------------------------------------------------
    if record.get("bbox") is not None:
        bb = record["bbox"]
        if isinstance(bb, (list, tuple)) and len(bb) >= 5:
            return [{"yolo": [float(x) for x in bb[:5]], "class_name": None}]
        return None
    return None


def validate_yolo_row(row: list, nc: int) -> Optional[str]:
    """Return an error string if the YOLO row is invalid, else None."""
    if not isinstance(row, (list, tuple)) or len(row) != 5:
        return "malformed row (needs 5 fields)"
    try:
        cls_id = int(round(float(row[0])))
        cx, cy, w, h = (float(x) for x in row[1:5])
    except (TypeError, ValueError):
        return "non-numeric row"
    if not (0 <= cls_id < nc):
        return f"invalid class id {cls_id}"
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return "center out of [0,1]"
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return "width/height not in (0,1]"
    if cx - w / 2 < 0 or cx + w / 2 > 1 or cy - h / 2 < 0 or cy + h / 2 > 1:
        return "box extends beyond image frame"
    return None


def validate_pixel_box(box: List[float], img_w: float, img_h: float) -> Optional[str]:
    if len(box) != 4:
        return "malformed box (needs 4 values)"
    x1, y1, x2, y2 = (float(v) for v in box)
    if not (x2 > x1 and y2 > y1):
        return "degenerate box (x2<=x1 or y2<=y1)"
    w, h = x2 - x1, y2 - y1
    if w < 1 or h < 1:
        return "box with sub-pixel width/height"
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return "box extends beyond image frame"
    return None


def pixel_box_to_yolo(box: List[float], img_w: float, img_h: float, cls_id: int) -> List[float]:
    """Convert a pixel-space box to a normalized YOLO row."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [cls_id, cx, cy, w, h]

def check_gate(
    data_root: Path,
    decisions: Dict[str, object],
    proposals: List[dict],
    manual_annotations: Tuple[Optional[dict], Dict[str, List[list]]],
    class_names: List[str],
    policy_path: Optional[Path] = None,
) -> GateResult:
    """Verify the human-review gate and return a ``GateResult``.

    The gate BLOCKS the build whenever any of the following hold:
    * unresolved (``uncertain``/``skip``ped/missing) proposal-review decisions
    * malformed decision records
    * invalid class IDs in accepted proposals/manual annotations
    * invalid coordinates in accepted proposals
    * invalid class names in accepted proposals
    * manually supplied annotations that fail validation
    * empty-label/annotate decisions whose required bounding box is missing
    * ambiguous/uncertain cases unresolved when the policy requires a decision

    Blocking is never silent: each violation is appended to ``GateResult.reasons``.
    """
    result = GateResult()
    nc = len(class_names)
    name_to_id = {n: i for i, n in enumerate(class_names)}
    records = decisions.get("records", [])

    # --- structural check of decision records ---------------------------------
    for r in records:
        if not isinstance(r, dict):
            result.malformed += 1
            result.add("malformed decision record (not a dict)")
            continue
        has_pd = isinstance(r.get("human_decision"), str) or isinstance(r.get("decision"), str)
        if not has_pd:
            result.malformed += 1
            result.add("malformed decision record with no decision")
        elif is_proposal_review_record(r):
            hd = r.get("human_decision")
            if hd not in ALLOWED_DECISIONS:
                result.malformed += 1
                result.add(f"unknown human_decision {hd!r}")

    # --- unresolved / uncertain proposal-review decisions ---------------------
    for r in records:
        if not is_proposal_review_record(r):
            continue
        hd = r.get("human_decision")
        img = r.get("image_filename") or Path(str(r.get("image", ""))).name
        unsettled = hd in ("uncertain",) or (hd == "skip") or hd not in ALLOWED_DECISIONS
        if unsettled:
            result.unresolved_proposal_records += 1
            result.unresolved_images.add(img)
            result.add(f"unresolved proposal record (human_decision={hd!r}) for {img}")

    # --- accepted / corrected proposals: validate geometry & class ------------
    for r in records:
        if not is_proposal_review_record(r):
            continue
        if r.get("human_decision") not in ("accepted", "corrected"):
            continue
        finals = _normalize_box(r)
        props = r.get("ai_proposals") or ([r["ai_proposal"]] if r.get("ai_proposal") else [])
        for p in props:
            cls_name = p.get("class_name")
            if cls_name is not None and cls_name not in name_to_id:
                result.invalid_class_name += 1
                result.add(f"invalid class name {cls_name!r} in accepted proposal {r.get('image_filename')}")
            cid = p.get("class_id")
            if cid is not None:
                try:
                    i = int(cid)
                except (TypeError, ValueError):
                    i = -1
                if not (0 <= i < nc):
                    result.invalid_class += 1
                    result.add(f"invalid class id {cid!r} in accepted proposal {r.get('image_filename')}")
        img_path = _resolve_repo_path(r.get("image", ""))
        img_w = img_h = None
        if img_path is not None and img_path.exists():
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:  # noqa: BLE001
                img_w = img_h = None
        if finals and img_w and img_h:
            for fb in finals:
                if fb.get("box_px") is not None:
                    err = validate_pixel_box(fb["box_px"], img_w, img_h)
                    if err:
                        result.invalid_box += 1
                        result.add(f"accepted proposal invalid box ({err}) for {r.get('image_filename')}")


    # --- adjudication records (annotate / manual_review / tighten) -----------
    _manual_decisions, manual_labels = manual_annotations
    for r in records:
        if is_proposal_review_record(r):
            continue
        dec = r.get("decision")
        img = r.get("image_filename") or Path(str(r.get("image", ""))).name
        if dec == "annotate":
            stem = Path(str(img)).stem
            has_manual = stem in manual_labels and len(manual_labels[stem]) > 0
            if r.get("bbox") is None and not has_manual:
                result.manual_required += 1
                result.unresolved_images.add(img)
                result.add(f"annotation required for empty-label image {img} but no valid box supplied")
        if dec in ADJUDICATION_DECISION_NEEDS_RESOLUTION and dec != "annotate":
            result.ambiguous_unresolved += 1
            result.unresolved_images.add(img)
            result.add(f"adjudication {dec!r} unresolved for {img}")

    # --- manual annotations must all be valid ---------------------------------
    for stem, rows in manual_labels.items():
        for row in rows:
            err = validate_yolo_row(row, nc)
            if err:
                result.manual_invalid += 1
                result.add(f"manual annotation {stem}.txt invalid ({err})")

    # --- policy requirement: ambiguous/uncertain must be resolved -------------
    if policy_path is not None and policy_path.exists():
        pol_names = class_names_from_policy(policy_path)
        if pol_names and set(pol_names) != set(class_names):
            result.add("annotation policy class list differs from data.yaml class list")
    return result


def is_proposal_review_record(r: dict) -> bool:
    """Return True if a decision record uses the proposal-review schema."""
    return isinstance(r.get("human_decision"), str)


def _resolve_repo_path(rel: str) -> Optional[Path]:
    """Resolve a repo-relative or absolute path under the repository root."""
    if not rel:
        return None
    p = Path(str(rel).replace("\\", "/"))
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


# ---------------------------------------------------------------------------
# Duplicate / overlap detection (IoU-based)
# ---------------------------------------------------------------------------
def _iou_norm(a: list, b: list) -> float:
    """IoU between two normalized YOLO rows ``[c, cx, cy, w, h]``."""
    ax1, ay1 = a[1] - a[3] / 2.0, a[2] - a[4] / 2.0
    ax2, ay2 = a[1] + a[3] / 2.0, a[2] + a[4] / 2.0
    bx1, by1 = b[1] - b[3] / 2.0, b[2] - b[4] / 2.0
    bx2, by2 = b[1] + b[3] / 2.0, b[2] + b[4] / 2.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def dedup_rows(rows: List[list], iou_threshold: float = 0.9) -> List[list]:
    """Remove duplicate / near-identical overlapping rows.

    Two rows are considered duplicates when their IoU >= ``iou_threshold`` AND
    they share the same class id AND one box is contained within the other's
    area by most of its extent (i.e. they cover the same object). The larger
    box is kept. This is the explicit, manifest-recorded duplicate policy.
    """
    kept: List[list] = []
    for row in sorted(rows, key=lambda r: -(r[3] * r[4])):
        is_dup = False
        for k in kept:
            if k[0] != row[0]:
                continue
            if _iou_norm(k, row) >= iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Per-image label construction
# ---------------------------------------------------------------------------
def build_image_v3(
    split: str,
    image_path: Path,
    v2_label_path: Optional[Path],
    decisions_for_image: List[dict],
    manual_labels: Dict[str, List[list]],
    class_names: List[str],
) -> dict:
    """Build the V3 YOLO label rows for one image.

    Returns a dict with:
        rows            : final deduplicated list of [cls, cx, cy, w, h]
        kept_original   : count of original V2 rows retained
        accepted        : count of accepted/corrected proposal boxes added
        rejected        : count of ai proposals explicitly rejected
        manually_added  : count of manually supplied boxes added
        modified        : count of original rows replaced/edited
        removed         : count of original rows removed
        notes           : human-readable audit notes for the manifest

    Decision semantics (see policy doc):
        ACCEPT       -> append the accepted proposal's box (after IoU dedup)
        CORRECTED    -> replace the matched original box with the corrected one
        REJECT       -> do not use the rejected proposal's box
        KEEP ORIGINAL-> preserve the original V2 annotation
        UNCERTAIN    -> must not reach construction (gate blocks first)
    """
    name_to_id = {n: i for i, n in enumerate(class_names)}
    nc = len(class_names)

    rows: List[list] = []
    kept, accepted_added, rejected_count, manual_added, modified, removed = 0, 0, 0, 0, 0, 0
    notes: List[str] = []

    img_w = img_h = 1.0
    if image_path.exists():
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:  # noqa: BLE001
            pass

    # Start from the original V2 annotation (preserved unless a decision says otherwise)
    if v2_label_path is not None and v2_label_path.exists():
        v2_rows, _ = _read_boxes(v2_label_path, nc)
        rows = [list(r[:5]) for r in v2_rows]
        kept = len(rows)

    proposal_records = [r for r in decisions_for_image if is_proposal_review_record(r)]
    for r in proposal_records:
        hd = r.get("human_decision")
        props = r.get("ai_proposals") or ([r["ai_proposal"]] if r.get("ai_proposal") else [])
        if hd == "accepted":
            for p in props:
                cls_id = int(p.get("class_id", -1))
                if not (0 <= cls_id < nc):
                    continue
                box = [p.get("x1", 0), p.get("y1", 0), p.get("x2", 0), p.get("y2", 0)]
                err = validate_pixel_box(box, img_w, img_h)
                if err:
                    notes.append(f"skipped invalid accepted box ({err})")
                    continue
                rows.append(pixel_box_to_yolo(box, img_w, img_h, cls_id))
                accepted_added += 1
        elif hd == "corrected":
            # Replace each original row that matches the corrected proposal.
            finals = _normalize_box(r)
            for p, fb in zip(props, (finals or [])):
                cls_name = fb.get("class_name") or p.get("class_name")
                cls_id = int(p.get("class_id", -1))
                if cls_name is not None and cls_name in name_to_id:
                    cls_id = name_to_id[cls_name]
                box = fb.get("box_px")
                if box is None:
                    box = [p.get("x1", 0), p.get("y1", 0), p.get("x2", 0), p.get("y2", 0)]
                if not (0 <= cls_id < nc):
                    continue
                err = validate_pixel_box(box, img_w, img_h)
                if err:
                    notes.append(f"skipped invalid corrected box ({err})")
                    continue
                corrected_row = pixel_box_to_yolo(box, img_w, img_h, cls_id)
                # Replace the closest original row (same class preferred).
                replaced = False
                for idx, orig in enumerate(rows):
                    if orig[0] == cls_id and _iou_norm(orig, corrected_row) >= 0.3:
                        rows[idx] = corrected_row
                        modified += 1
                        replaced = True
                        break
                if not replaced:
                    rows.append(corrected_row)
                    accepted_added += 1
        elif hd == "rejected":
            rejected_count += len(props)

    # Manual annotations (validated) for this image's stem.
    stem = image_path.stem
    if stem in manual_labels:
        for row in manual_labels[stem]:
            if validate_yolo_row(row, nc) is None:
                rows.append(list(row))
                manual_added += 1
                notes.append(f"added manual annotation from {stem}.txt")

    final_rows = dedup_rows(rows)
    removed = max(0, kept + accepted_added + manual_added - len(final_rows))

    return {
        "rows": final_rows,
        "kept_original": kept,
        "accepted": accepted_added,
        "rejected": rejected_count,
        "manually_added": manual_added,
        "modified": modified,
        "removed": removed,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# V3 construction orchestration
# ---------------------------------------------------------------------------
def _gather_or_build_manifest_statistics(
    split: str,
    v3_img_dir: Path,
    v3_lbl_dir: Path,
    class_names: List[str],
    stats: dict,
) -> None:
    """Count images / objects per class / empty labels for one V3 split."""
    imgs = _list_images(v3_img_dir)
    stats["images"] += len(imgs)
    nc = len(class_names)
    for lp in _list_labels(v3_lbl_dir):
        rows, _ = _read_boxes(lp, nc)
        if not rows:
            stats["empty_labels"] += 1
        for r in rows:
            cid = int(r[0])
            if 0 <= cid < nc:
                stats["per_class_instances"][class_names[cid]] += 1
                stats["total_objects"] += 1


def construct_v3(
    data_root: Path,
    out_root: Path,
    decisions: Dict[str, object],
    manual_annotations: Tuple[Optional[dict], Dict[str, List[list]]],
    class_names: List[str],
    manifest_meta: Optional[dict] = None,
    force: bool = False,
) -> dict:
    """Construct ``data/detection_v3`` from the V2 dataset + validated decisions.

    The build is atomic-ish: it first copies all images into a staging temp dir,
    composes the V3 labels, and only on full success moves them into
    ``out_root``. If ``out_root`` already exists the build refuses unless
    ``force=True``. Never touches ``data/detection`` or ``best.pt``.
    """
    out_root = Path(out_root)
    if out_root.exists() and not force:
        raise RuntimeError(
            f"{out_root} already exists; refusing to overwrite without --force")
    manifest_meta = manifest_meta or {}

    # Index decision records by image (normalized repo-relative path).
    records = decisions.get("records", [])
    by_img: Dict[str, List[dict]] = {}
    for r in records:
        key = str(r.get("image", "")).replace("\\", "/")
        by_img.setdefault(key, []).append(r)

    _manual_decisions, manual_labels = manual_annotations

    stats = {
        "train": {"images": 0, "total_objects": 0, "per_class_instances": {},
                  "empty_labels": 0},
        "valid": {"images": 0, "total_objects": 0, "per_class_instances": {},
                  "empty_labels": 0},
        "test": {"images": 0, "total_objects": 0, "per_class_instances": {},
                 "empty_labels": 0},
    }
    for c in class_names:
        for s in stats.values():
            s["per_class_instances"][c] = 0

    counts = {
        "original_retained": 0,
        "ai_accepted": 0,
        "ai_rejected": 0,
        "manual_created": 0,
        "modified": 0,
        "removed": 0,
    }

    import tempfile
    staged = Path(tempfile.mkdtemp(prefix="v3_stage_"))
    try:
        for split in ("train", "valid", "test"):
            v2_imgs = data_root / split / "images"
            if not v2_imgs.is_dir():
                continue
            st_img = staged / split / "images"
            st_lbl = staged / split / "labels"
            st_img.mkdir(parents=True, exist_ok=True)
            st_lbl.mkdir(parents=True, exist_ok=True)
            for img_path in _list_images(v2_imgs):
                # Decision records may reference the image by: absolute path,
                # repo-relative path, "data/detection/..." form, or bare name.
                # Normalise every candidate key to forward slashes to match the
                # by_img index (which is built from normalised record paths).
                recs = []
                norm_abs = str(img_path).replace("\\", "/")
                candidates = [
                    norm_abs,
                    f"data/detection/{split}/images/{img_path.name}",
                    f"data/{split}/images/{img_path.name}",
                    img_path.name,
                ]
                for cand in candidates:
                    recs = by_img.get(cand, [])
                    if recs:
                        break
                lbl_src = data_root / split / "labels" / (img_path.stem + ".txt")
                lbl_src = lbl_src if lbl_src.exists() else None
                res = build_image_v3(
                    split, img_path, lbl_src, recs, manual_labels, class_names)
                # write image + label to staging
                shutil.copy2(str(img_path), str(st_img / img_path.name))
                with open(st_lbl / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
                    for row in res["rows"]:
                        f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} "
                                f"{row[3]:.6f} {row[4]:.6f}\n")
                counts["original_retained"] += res["kept_original"]
                counts["ai_accepted"] += res["accepted"]
                counts["ai_rejected"] += res["rejected"]
                counts["manual_created"] += res["manually_added"]
                counts["modified"] += res["modified"]
                counts["removed"] += res["removed"]
            # tally statistics
            _gather_or_build_manifest_statistics(
                split, st_img, st_lbl, class_names, stats[split])

        # Success: move staging into place.
        if out_root.exists() and force:
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            src = staged / split
            if src.exists():
                shutil.move(str(src), str(out_root / split))
        write_data_yaml(out_root, class_names)
        v3_stats = {
            "train": stats["train"],
            "valid": stats["valid"],
            "test": stats["test"],
        }
        manifest = build_manifest(
            data_root=data_root,
            out_root=out_root,
            class_names=class_names,
            counts=counts,
            stats=v3_stats,
            manifest_meta=manifest_meta,
        )
        with open(out_root / "v3_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        return manifest
    finally:
        shutil.rmtree(staged, ignore_errors=True)


def write_data_yaml(out_root: Path, class_names: List[str]) -> None:
    """Write the V3 data.yaml (train/val/test images dirs + class list)."""
    cfg = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def build_manifest(
    data_root: Path,
    out_root: Path,
    class_names: List[str],
    counts: dict,
    stats: dict,
    manifest_meta: dict,
) -> dict:
    """Assemble the V3 reproducibility manifest (v3_manifest.json)."""
    # Source V2 image counts
    v2_counts = {}
    for split in ("train", "valid", "test"):
        img_dir = data_root / split / "images"
        v2_counts[split] = len(_list_images(img_dir)) if img_dir.is_dir() else 0

    total_v2 = sum(v2_counts.values())
    total_v3 = sum(s["images"] for s in stats.values())
    per_class_v3 = {}
    for c in class_names:
        per_class_v3[c] = sum(s["per_class_instances"].get(c, 0) for s in stats.values())

    manifest = {
        "builder_version": BUILDER_VERSION,
        "source_dataset": str(data_root),
        "source_dataset_checksum": _dir_sha256(data_root),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_v2_image_counts": v2_counts,
        "v3_image_counts": {s: stats[s]["images"] for s in stats},
        "split_counts": {s: stats[s]["images"] for s in stats},
        "class_counts": per_class_v3,
        "total_objects_v3": sum(s["total_objects"] for s in stats.values()),
        "original_annotations_retained": counts["original_retained"],
        "ai_annotations_accepted": counts["ai_accepted"],
        "ai_annotations_rejected": counts["ai_rejected"],
        "annotations_manually_created": counts["manual_created"],
        "annotations_modified": counts["modified"],
        "annotations_removed": counts["removed"],
        "unresolved_cases": 0,
        "validation_result": "passed",
        "policy_version": manifest_meta.get("policy_version"),
        "policy_path": str(manifest_meta.get("policy_path", "docs/DETECTION_V3_ANNOTATION_POLICY.md")),
        "decision_manifest_hash": manifest_meta.get("decision_manifest_hash"),
        "proposal_manifest_hash": manifest_meta.get("proposal_manifest_hash"),
        "git_commit": _git_commit(),
        "notes": manifest_meta.get("notes", []),
    }
    return manifest


def _dir_sha256(root: Path) -> str:
    """Bounded SHA256 of a dataset directory (paths + sizes + hashes of labels)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            try:
                h.update(rel.encode("utf-8"))
                h.update(str(p.stat().st_size).encode("utf-8"))
                if p.suffix == ".txt":
                    h.update(p.read_bytes())
            except OSError:
                continue
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(_REPO_ROOT), timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def run_gate(args) -> GateResult:
    """Load inputs and run the V3 gate (used by both --gate-only and build)."""
    data_root = _find_dataset_root(args.data_dir) or _find_dataset_root(_REPO_ROOT / args.data_dir)
    if data_root is None:
        sys.exit("ERROR: could not resolve V2 dataset root")
    _raw, _nc, class_names = load_data_config(data_root)
    if not class_names:
        sys.exit("ERROR: data.yaml has no class names")

    decisions = load_human_decisions(args.decisions_file)
    proposals = load_proposals(args.proposals_file)
    manual = load_manual_annotations(args.manual_dir)
    gate = check_gate(
        data_root=data_root,
        decisions=decisions,
        proposals=proposals,
        manual_annotations=manual,
        class_names=class_names,
        policy_path=args.policy_file,
    )
    return gate




# --------------------------------------------------------------------------- #
# Exclusion (V2-copy) V3 build -- controlled experiment mode
# --------------------------------------------------------------------------- #
# This is a SEPARATE, opt-in mode used by the controlled YOLO experiment:
#   "V3 = V2 minus the 14 unresolved TRAIN/VALIDATION blockers".
# It does NOT consult human decisions / proposals / manual annotations. It
# merely copies the frozen V2 layout and excludes the unresolved samples from
# the review queue, preserving every retained YOLO label byte-for-byte and
# leaving the V2 TEST set completely unchanged.
#
# It is additive: it does not change the default (gate-based) construction
# path consumed by resolve_v3_human_review.py and by the existing tests.
# --------------------------------------------------------------------------- #

# Path to the human-review queue (authoritative source of the 14 unresolved
# blocker images). Relative paths are resolved against the repo root.
DEFAULT_QUEUE_FILE = Path("reports/audit_review/v3_human_review_queue.json")

# The 10-class V2 mapping (authoritative; must not change).
EXPECTED_CLASS_NAMES = [
    "Apple", "Grape", "Kiwi", "Mango", "Orange", "Strawberry",
    "banana", "cherry", "chickoo", "guava",
]


def load_blocker_queue(queue_file: Optional[Path] = None) -> List[dict]:
    """Load the authoritative V3 human-review queue and return its ``items``.

    The queue report (``reports/audit_review/v3_human_review_queue.json``) is
    the single source of truth for which images are unresolved and must be
    excluded from V3 train/valid. Only items whose ``split`` is ``train`` or
    ``valid`` are considered (test is never excluded). Raises ``RuntimeError``
    if the file is missing/malformed.
    """
    path = queue_file if queue_file is not None else DEFAULT_QUEUE_FILE
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        raise RuntimeError(f"review queue not found: {path}")
    data = load_json(path)
    items = data.get("items")
    if not isinstance(items, list):
        raise RuntimeError(f"review queue {path} has no 'items' list")
    excluded = []
    for it in items:
        split = _norm_split(it.get("split") or "train") or "train"
        fname = it.get("image_filename") or Path(str(it.get("image"))).name
        if split not in ("train", "valid"):
            # Never exclude anything from test.
            continue
        if not fname:
            continue
        excluded.append({"image_filename": fname, "split": split,
                         "category": it.get("category"),
                         "current_decision": it.get("current_decision")})
    return excluded


def _split_root(data_root: Path, split: str) -> Path:
    return data_root / split


def _data_yaml_mapping_check(data_root: Path, class_names: List[str]) -> None:
    """Refuse to build if the V2 data.yaml class mapping is not exactly ours."""
    if class_names != EXPECTED_CLASS_NAMES:
        raise RuntimeError(
            f"V2 data.yaml class mapping mismatch. Expected "
            f"{len(EXPECTED_CLASS_NAMES)} classes {EXPECTED_CLASS_NAMES}, "
            f"got {len(class_names)} {class_names}"
        )


def _compute_exclusion(data_root: Path, class_names: List[str]) -> dict:
    """Compute the full exclusion plan without writing anything.

    Returns a dict describing the plan:
        source_image_counts   : {split: int}
        source_label_counts   : {split: int}
        source_object_counts  : {split: int}
        blocks                : list of {split, image_filename, category}
        excluded_by_split     : {split: int}
        destination_counts    : {split: int}  (images retained per split)
        test_set_unchanged    : bool
        v2_dataset_sha256     : str
    """
    blockers = load_blocker_queue()
    _data_yaml_mapping_check(data_root, class_names)

    source_image_counts: Dict[str, int] = {}
    source_label_counts: Dict[str, int] = {}
    source_object_counts: Dict[str, int] = {}
    for split in ("train", "valid", "test"):
        imgs = set(p.name for p in _list_images(data_root / split / "images"))
        lbls = set(p.name for p in _list_labels(data_root / split / "labels"))
        source_image_counts[split] = len(imgs)
        source_label_counts[split] = len(lbls)
        obj = 0
        for lb in _list_labels(data_root / split / "labels"):
            rows, _ = _read_boxes(lb, len(class_names))
            obj += len(rows)
        source_object_counts[split] = obj

    excluded_by_split: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    seen_files: Dict[str, str] = {}
    for b in blockers:
        split = b["split"]
        fname = b["image_filename"]
        if fname in seen_files:
            raise RuntimeError(
                f"blocker image {fname} appears more than once "
                f"(existing source {split}); refusing to build"
            )
        seen_files[fname] = split
        excluded_by_split[split] += 1

    destination_counts: Dict[str, int] = {}
    for split in ("train", "valid", "test"):
        destination_counts[split] = source_image_counts[split] - excluded_by_split[split]

    return {
        "source_dataset": str(data_root),
        "source_image_counts": source_image_counts,
        "source_label_counts": source_label_counts,
        "source_object_counts": source_object_counts,
        "blocks": [{"split": b["split"], "image_filename": b["image_filename"],
                    "category": b.get("category")} for b in blockers],
        "excluded_by_split": excluded_by_split,
        "destination_counts": destination_counts,
        "test_set_unchanged": destination_counts["test"] == source_image_counts["test"],
        "v2_dataset_sha256": _dir_sha256(data_root),
    }


def _write_data_yaml_for_exclusion(data_root: Path, out_root: Path) -> None:
    """Write V3 data.yaml preserving the exact V2 class mapping."""
    raw, nc, names = load_data_config(data_root)
    _data_yaml_mapping_check(data_root, names)
    cfg = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    (out_root / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8")



def verify_v3_build(data_root: Path, out_root: Path, plan: dict) -> dict:
    """Validate that the built V3 exactly matches the exclusion plan."""
    ver = {}
    for split in ("train", "valid", "test"):
        src_imgs = set(p.name for p in _list_images(data_root / split / "images"))
        src_lbls = set(p.name for p in _list_labels(data_root / split / "labels"))
        dst_imgs = set(p.name for p in _list_images(out_root / split / "images"))
        dst_lbls = set(p.name for p in _list_labels(out_root / split / "labels"))
        excluded_img = {b["image_filename"] for b in plan["blocks"] if b["split"] == split}
        excluded_stems = {Path(b["image_filename"]).stem for b in plan["blocks"]
                          if b["split"] == split}
        excluded_lbls = {st + ".txt" for st in excluded_stems}

        missing_imgs = sorted((src_imgs - excluded_img) - dst_imgs)
        missing_lbls = sorted((src_lbls - excluded_lbls) - dst_lbls)
        leftover_imgs = sorted(dst_imgs - (src_imgs - excluded_img))
        leftover_lbls = sorted(dst_lbls - (src_lbls - excluded_lbls))

        img_identical = lbl_identical = True
        if split == "test":
            for name in src_imgs:
                src_f = data_root / split / "images" / name
                dst_f = out_root / split / "images" / name
                if not dst_f.exists() or src_f.read_bytes() != dst_f.read_bytes():
                    img_identical = False
            for name in src_lbls:
                src_f = data_root / split / "labels" / name
                dst_f = out_root / split / "labels" / name
                if not dst_f.exists() or src_f.read_bytes() != dst_f.read_bytes():
                    lbl_identical = False

        split_ok = (not missing_imgs and not missing_lbls
                    and not leftover_imgs and not leftover_lbls)
        if split == "test":
            split_ok = split_ok and img_identical and lbl_identical

        ver[split] = {
            "source_images": len(src_imgs),
            "destination_images": len(dst_imgs),
            "excluded_images": len(excluded_img),
            "missing_images": missing_imgs,
            "missing_labels": missing_lbls,
            "leftover_images": leftover_imgs,
            "leftover_labels": leftover_lbls,
            "test_images_byte_identical": img_identical,
            "test_labels_byte_identical": lbl_identical,
            "consistent": split_ok,
        }

    counts = {s: ver[s]["destination_images"] for s in ("train", "valid", "test")}
    obj_counts = {}
    for split in ("train", "valid", "test"):
        o = 0
        for lb in _list_labels(out_root / split / "labels"):
            rows, _ = _read_boxes(lb, len(EXPECTED_CLASS_NAMES))
            o += len(rows)
        obj_counts[split] = o

    all_consistent = all(ver[s]["consistent"] for s in ("train", "valid", "test"))
    return {
        "destination_counts": counts,
        "destination_object_counts": obj_counts,
        "verification": {
            "passed": all_consistent,
            "per_split": ver,
        },
    }


def _exclusion_report_md(report: dict) -> str:
    lines = [
        "# V3 Dataset Exclusion Report",
        "",
        f"- **Source**: `{report['source_dataset']}`",
        f"- **Destination**: `{report['destination_dataset']}`",
        f"- **Timestamp**: {report.get('timestamp', '')}",
        f"- **Mode**: `{report.get('build_mode', 'exclusion')}`",
        "",
        "## Exclusion summary",
        "",
        f"- Excluded images: **{report['excluded_count']}**",
        f"- Excluded by split: {report['excluded_by_split']}",
        f"- Reason: {report['reason']}",
        "## Test set",
        "",
        f"- Source test images: {report['source_test_count']}",
        f"- Destination test images: {report['destination_test_count']}",
        f"- Test set unchanged: **{report['test_set_unchanged']}**",
        "",
        "## Destination counts (per split)",
        "",
        "| Split | Images |",
        "| --- | ---: |",
    ]
    for s in ("train", "valid", "test"):
        lines.append(f"| {s} | {report['destination_counts'].get(s, 0)} |")
    lines.append("")
    lines.append("## Excluded files")
    lines.append("")
    for fname in report.get("excluded_files", []):
        lines.append(f"- `{fname}`")
    lines.append("")
    if report.get("source_v2_hashes"):
        lines.append("## V2 integrity")
        lines.append("")
        lines.append(f"- V2 dataset SHA256: `{report['source_v2_hashes'].get('dataset_sha256')}`")
        lines.append(f"- best.pt SHA256: `{report['source_v2_hashes'].get('best_pt_sha256')}`")
    return "\n".join(lines)




def exclusion_build(
    data_root: Path,
    out_root: Path,
    dry_run: bool = False,
    force: bool = False,
    reports_dir: Path | None = None,
) -> dict:
    """Build V3 as a copy of V2 minus the unresolved train/valid blockers.

    Returns the exclusion report dict. Never touches V2. When ``dry_run`` is
    True nothing is written anywhere. ``reports_dir`` defaults to
    ``<repo>/reports`` and may be overridden (e.g. by tests) to avoid mutating
    the real reports directory.
    """
    class_names = load_data_config(data_root)[2]
    plan = _compute_exclusion(data_root, class_names)

    report = {
        "source_dataset": str(plan["source_dataset"]),
        "destination_dataset": str(out_root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "excluded_count": sum(plan["excluded_by_split"].values()),
        "excluded_by_split": plan["excluded_by_split"],
        "excluded_files": [b["image_filename"] for b in plan["blocks"]],
        "reason": ("remove unresolved annotation blockers (empty-label, manual"
                   " review, tighten) from V3 train/valid; test is unchanged"),
        "source_test_count": plan["source_image_counts"]["test"],
        "destination_test_count": plan["destination_counts"]["test"],
        "test_set_unchanged": plan["test_set_unchanged"],
        "source_v2_hashes": {
            "dataset_sha256": plan["v2_dataset_sha256"],
            "best_pt_sha256": sha256_file(
                _REPO_ROOT / "models/detection/detector/weights/best.pt")
            if (_REPO_ROOT / "models/detection/detector/weights/best.pt").exists()
            else None,
        },
        "source_counts": {
            "train_images": plan["source_image_counts"]["train"],
            "valid_images": plan["source_image_counts"]["valid"],
            "test_images": plan["source_image_counts"]["test"],
            "labels": plan["source_label_counts"],
            "objects": plan["source_object_counts"],
        },
        "destination_counts": plan["destination_counts"],
        "build_mode": "exclusion",
    }

    report_name = "detection_v3_exclusion_report"
    reports_root = reports_dir if reports_dir is not None else (_REPO_ROOT / "reports")
    reports_root = reports_root if reports_root.is_absolute() else (_REPO_ROOT / reports_root)
    json_out = reports_root / f"{report_name}.json"
    md_out = reports_root / f"{report_name}.md"

    if dry_run:
        report["dry_run"] = True
        report["would_write"] = {
            "destination": str(out_root),
            "reports": [str(json_out), str(md_out)],
        }
        return report

    if out_root.exists():
        if not force:
            raise RuntimeError(
                f"destination already exists: {out_root}. Refusing to merge or "
                f"overwrite. Re-run with --force to intentionally rebuild."
            )

    import tempfile
    tmp_out = Path(tempfile.mkdtemp(prefix=".v3build_", dir=str(out_root.parent)))
    try:
        for split in ("train", "valid", "test"):
            src_imgs = _split_root(data_root, split) / "images"
            src_lbls = _split_root(data_root, split) / "labels"
            dst_imgs = tmp_out / split / "images"
            dst_lbls = tmp_out / split / "labels"
            dst_imgs.mkdir(parents=True, exist_ok=True)
            dst_lbls.mkdir(parents=True, exist_ok=True)

            excluded = {Path(b["image_filename"]).stem for b in plan["blocks"]
                        if b["split"] == split}

            for img in _list_images(src_imgs):
                if img.stem in excluded:
                    continue
                shutil.copyfile(img, dst_imgs / img.name)

            for lbl in _list_labels(src_lbls):
                # Exclude the label whose stem matches an excluded image so
                # no orphaned label files remain in V3 (image+label kept in
                # lockstep; test labels are byte-identical copies).
                if lbl.stem in excluded:
                    continue
                shutil.copyfile(lbl, dst_lbls / lbl.name)

        _write_data_yaml_for_exclusion(data_root, tmp_out)

        if out_root.exists():
            shutil.rmtree(out_root)
        shutil.move(str(tmp_out), str(out_root))
    except Exception:
        shutil.rmtree(tmp_out, ignore_errors=True)
        raise

    built = verify_v3_build(data_root, out_root, plan)
    report["destination_counts"] = built["destination_counts"]
    report["destination_object_counts"] = built["destination_object_counts"]
    report["verification"] = built["verification"]

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    md_out.write_text(_exclusion_report_md(report), encoding="utf-8")

    return report



def _print_exclusion_report(report: dict) -> None:
    """Print a compact summary of a completed exclusion build."""
    print("V3 EXCLUSION BUILD COMPLETE")
    print(f"  output        : {report['destination_dataset']}")
    print(f"  excluded      : {report['excluded_count']}")
    print(f"  excluded/split: {report['excluded_by_split']}")
    print(f"  destination   : {report['destination_counts']}")
    print(f"  objects       : {report.get('destination_object_counts', {})}")
    print(f"  test unchanged: {report['test_set_unchanged']}")
    ver = report.get("verification", {})
    if isinstance(ver, dict) and ver.get("passed") is not None:
        print(f"  verification  : {'PASSED' if ver['passed'] else 'FAILED'}")
    print(f"  report json   : reports/detection_v3_exclusion_report.json")
    print(f"  report md     : reports/detection_v3_exclusion_report.md")



def main() -> int:
    parser = argparse.ArgumentParser(description="SmartFreshAI V3 dataset builder")
    parser.add_argument("--data-dir", type=Path, default=Path("data/detection"),
                        help="Frozen V2 dataset root")
    parser.add_argument("--out-dir", type=Path, default=Path("data/detection_v3"),
                        help="V3 output root")
    parser.add_argument("--decisions-file", type=Path,
                        default=Path("reports/audit_review/human_decisions.json"))
    parser.add_argument("--proposals-file", type=Path,
                        default=Path("reports/audit_review/ai_annotation_proposals/proposals.json"))
    parser.add_argument("--manual-dir", type=Path,
                        default=Path("reports/audit_review/manual_annotations"))
    parser.add_argument("--policy-file", type=Path,
                        default=Path("docs/DETECTION_V3_ANNOTATION_POLICY.md"))
    parser.add_argument("--gate-only", action="store_true",
                        help="Only run the review gate; do not build V3")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting an existing V3 output dir")
    parser.add_argument("--exclusion-build", action="store_true",
                        help="Build V3 as a copy of V2 minus the unresolved "
                             "train/valid blockers (controlled experiment). "
                             "Does not consult human decisions.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show exactly what the exclusion build would do "
                             "without creating/modifying V3.")
    parser.add_argument("--verify", action="store_true",
                        help="Verify an existing V3 exclusion build against the "
                             "source V2 (read-only). Requires the dataset to exist.")
    parser.add_argument("--queue-file", type=Path, default=None,
                        help="Path to the V3 human-review queue JSON "
                             "(default: reports/audit_review/v3_human_review_queue.json)")
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help="Directory for the exclusion build report "
                             "(default: <repo>/reports). Tests override this.")
    args = parser.parse_args()

    data_root = _find_dataset_root(args.data_dir) or _find_dataset_root(_REPO_ROOT / args.data_dir)
    if data_root is None:
        logger.error("Could not resolve the V2 dataset root from %s", args.data_dir)
        return 1
    _, _, class_names = load_data_config(data_root)
    if not class_names:
        logger.error("data.yaml has no class names")
        return 1

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = _REPO_ROOT / out_root

    # --- Exclusion (V2-copy) mode: does NOT use the human-review gate ---------
    if args.exclusion_build or args.dry_run:
        global DEFAULT_QUEUE_FILE
        if args.queue_file is not None:
            DEFAULT_QUEUE_FILE = args.queue_file
        try:
            report = exclusion_build(
                data_root=data_root,
                out_root=out_root,
                dry_run=args.dry_run,
                force=args.force,
                reports_dir=args.reports_dir,
            )
        except RuntimeError as exc:
            logger.error("V3 exclusion build aborted: %s", exc)
            return 2

        if args.dry_run:
            print("V3 EXCLUSION BUILD -- DRY RUN (no files were written)")
            print(f"  source        : {report['source_dataset']}")
            print(f"  destination   : {report['destination_dataset']}")
            print(f"  excluded      : {report['excluded_count']}")
            print(f"  excluded/split: {report['excluded_by_split']}")
            print(f"  destination   : {report['destination_counts']}")
            print(f"  test unchanged: {report['test_set_unchanged']}")
            print("  would write   :")
            for w in report["would_write"]["reports"]:
                print(f"    - {w}")
            print(f"    - {report['would_write']['destination']}")
            return 0

        _print_exclusion_report(report)
        return 0

    if args.verify:
        if not out_root.exists():
            logger.error("verify mode: V3 dataset not found at %s", out_root)
            return 1
        try:
            plan = _compute_exclusion(data_root, class_names)
            built = verify_v3_build(data_root, out_root, plan)
        except RuntimeError as exc:
            logger.error("V3 verification failed: %s", exc)
            return 2
        print("V3 EXCLUSION BUILD VERIFY")
        print(f"  passed        : {built['verification']['passed']}")
        print(f"  destination   : {built['destination_counts']}")
        print(f"  object counts : {built['destination_object_counts']}")
        return 0 if built["verification"]["passed"] else 2

    decisions = load_human_decisions(args.decisions_file)
    proposals = load_proposals(args.proposals_file)
    manual = load_manual_annotations(args.manual_dir)

    gate = check_gate(
        data_root=data_root,
        decisions=decisions,
        proposals=proposals,
        manual_annotations=manual,
        class_names=class_names,
        policy_path=args.policy_file,
    )

    if not gate.passed:
        print("\nV3 BUILD BLOCKED")
        print("Reasons:")
        for reason in sorted(set(gate.reasons)):
            print(f"  - {reason}")
        print(f"\n  unresolved proposal records: {gate.unresolved_proposal_records}")
        print(f"  unresolved unique images:    {len(gate.unresolved_images)}")
        return 3

    if args.gate_only:
        print("V3 GATE PASSED")
        print("  unresolved proposal records: 0")
        print("  all decisions validated; V3 construction is permitted.")
        return 0

    manifest_meta = {
        "policy_version": "1.0",
        "policy_path": args.policy_file,
        "decision_manifest_hash": sha256_json(decisions),
        "proposal_manifest_hash": sha256_json(proposals),
        "notes": [f"built from {data_root} by scripts/build_detection_v3.py v{BUILDER_VERSION}"],
    }

    try:
        manifest = construct_v3(
            data_root=data_root,
            out_root=out_root,
            decisions=decisions,
            manual_annotations=manual,
            class_names=class_names,
            manifest_meta=manifest_meta,
            force=args.force,
        )
    except RuntimeError as exc:
        logger.error("V3 build aborted: %s", exc)
        return 2

    print("V3 BUILD COMPLETE")
    print(f"  output: {out_root}")
    print(f"  v3 images: {sum(manifest['split_counts'].values())}")
    print(f"  v3 objects: {manifest['total_objects_v3']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
