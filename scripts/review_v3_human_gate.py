#!/usr/bin/env python3
"""Interactive V3 human-review gate GUI.

This tool is intentionally read-only with respect to the frozen V2 dataset and
model weights. It only writes one file:

    reports/audit_review/v3_human_resolutions.json

The output manifest is designed to be consumed directly by:

    python scripts/resolve_v3_human_review.py --apply-resolution <manifest>

Safety constraints enforced by this module:
- Never writes under data/detection.
- Never writes under models/detection/detector/weights.
- Never creates data/detection_v3.
- Never fabricates boxes: all annotation coordinates come from explicit
  human GUI operations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageTk

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_detection_dataset import _find_dataset_root, load_data_config  # noqa: E402
from scripts.resolve_v3_human_review import apply_resolutions  # noqa: E402

QUEUE_PATH = Path("reports/audit_review/v3_human_review_queue.json")
DECISIONS_PATH = Path("reports/audit_review/human_decisions.json")
PROPOSALS_PATH = Path("reports/audit_review/ai_annotation_proposals/proposals.json")
PRIORITY_PATH = Path("reports/audit_review/review_priority.json")
DATA_ROOT_PATH = Path("data/detection")
MANIFEST_PATH = Path("reports/audit_review/v3_human_resolutions.json")

CAT_EMPTY = "A_EMPTY_LABEL"
CAT_MANUAL = "B_MANUAL_REVIEW"
CAT_TIGHTEN = "C_TIGHTEN"
CAT_UNCERTAIN = "D_UNCERTAIN"

CATEGORY_ORDER = {
    CAT_EMPTY: 0,
    CAT_MANUAL: 1,
    CAT_TIGHTEN: 2,
    CAT_UNCERTAIN: 3,
}

FILTER_ALL = "All"
FILTER_HIGH = "High priority"
FILTER_EMPTY = "Empty labels"
FILTER_MANUAL = "Manual review"
FILTER_TIGHTEN = "Tighten"
FILTER_UNCERTAIN = "Uncertain"
FILTER_UNRESOLVED = "Unresolved"
FILTER_RESOLVED = "Resolved"

SORT_PRIORITY = "priority"
SORT_CATEGORY = "category"
SORT_SPLIT = "split"
SORT_FILENAME = "filename"

EMPTY_ACTIONS = {"MANUALLY_ANNOTATE", "CONFIRM_BACKGROUND", "MARK_UNCERTAIN"}
HUGEBOX_ACTIONS = {"KEEP", "TIGHTEN", "REPLACE", "UNCERTAIN"}
UNCERTAIN_ACTIONS = {
    "ACCEPT_SELECTED",
    "REJECT_SELECTED",
    "KEEP_ORIGINAL",
    "CORRECT",
    "UNCERTAIN",
    "ACCEPT_ALL",
}

GT_COLOR = "#FF4D4D"
AI_COLOR = "#00D4FF"
EDIT_COLOR = "#FFD166"
SELECT_COLOR = "#39FF14"

GRAPE_POLICY_TEXT = (
    "GRAPE ANNOTATION POLICY\n"
    "Annotate one box per distinct grape bunch/cluster.\n"
    "Do NOT create one box per berry."
)


@dataclass
class DrawBox:
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    source: str

    def normalized(self) -> "DrawBox":
        return DrawBox(
            class_id=self.class_id,
            class_name=self.class_name,
            x1=min(self.x1, self.x2),
            y1=min(self.y1, self.y2),
            x2=max(self.x1, self.x2),
            y2=max(self.y1, self.y2),
            source=self.source,
        )


class ManifestValidationError(ValueError):
    """Raised when the resolution manifest is invalid."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def validate_yolo_row(row: List[float], class_count: int) -> Optional[str]:
    if not isinstance(row, (list, tuple)) or len(row) != 5:
        return "row must be [class_id, cx, cy, width, height]"
    try:
        class_id = int(row[0])
        cx = float(row[1])
        cy = float(row[2])
        width = float(row[3])
        height = float(row[4])
    except (TypeError, ValueError):
        return "row values must be numeric"
    if class_id < 0 or class_id >= class_count:
        return f"invalid class_id {class_id}"
    if not (0.0 <= cx <= 1.0):
        return "cx must be in [0,1]"
    if not (0.0 <= cy <= 1.0):
        return "cy must be in [0,1]"
    if not (0.0 < width <= 1.0):
        return "width must be in (0,1]"
    if not (0.0 < height <= 1.0):
        return "height must be in (0,1]"
    if cx - width / 2.0 < 0.0 or cx + width / 2.0 > 1.0:
        return "box extends beyond x bounds"
    if cy - height / 2.0 < 0.0 or cy + height / 2.0 > 1.0:
        return "box extends beyond y bounds"
    return None


def pixels_to_yolo(box: DrawBox, image_w: int, image_h: int) -> List[float]:
    b = box.normalized()
    if image_w <= 0 or image_h <= 0:
        raise ManifestValidationError("image dimensions must be positive")
    width = b.x2 - b.x1
    height = b.y2 - b.y1
    if width <= 0 or height <= 0:
        raise ManifestValidationError("box width/height must be positive")
    if b.x1 < 0 or b.y1 < 0 or b.x2 > image_w or b.y2 > image_h:
        raise ManifestValidationError("box must be fully inside image bounds")

    row = [
        int(b.class_id),
        (b.x1 + b.x2) / 2.0 / image_w,
        (b.y1 + b.y2) / 2.0 / image_h,
        width / image_w,
        height / image_h,
    ]
    return row


def yolo_to_pixels(row: Dict[str, Any], image_w: int, image_h: int, class_names: List[str], source: str) -> DrawBox:
    cid = int(row["class_id"])
    cx = float(row["cx"])
    cy = float(row["cy"])
    width = float(row["w"])
    height = float(row["h"])
    x1 = (cx - width / 2.0) * image_w
    y1 = (cy - height / 2.0) * image_h
    x2 = (cx + width / 2.0) * image_w
    y2 = (cy + height / 2.0) * image_h
    class_name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
    return DrawBox(cid, class_name, x1, y1, x2, y2, source=source)


def queue_item_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (str(item.get("image_filename", "")), str(item.get("category", "")))


def manifest_item_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (str(item.get("image_filename", "")), str(item.get("category", "")))


def build_queue_index(items: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for it in items:
        out[queue_item_key(it)] = it
    return out


def summarize_queue(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "empty": 0,
        "manual": 0,
        "tighten": 0,
        "uncertain": 0,
        "total": 0,
    }
    for it in items:
        cat = it.get("category")
        if cat == CAT_EMPTY:
            counts["empty"] += 1
        elif cat == CAT_MANUAL:
            counts["manual"] += 1
        elif cat == CAT_TIGHTEN:
            counts["tighten"] += 1
        elif cat == CAT_UNCERTAIN:
            counts["uncertain"] += 1
    counts["total"] = len({it.get("image_filename") for it in items})
    return counts


def load_review_payloads(
    queue_path: Path = QUEUE_PATH,
    decisions_path: Path = DECISIONS_PATH,
    proposals_path: Path = PROPOSALS_PATH,
    priority_path: Path = PRIORITY_PATH,
    data_root_path: Path = DATA_ROOT_PATH,
) -> Dict[str, Any]:
    if not queue_path.exists():
        raise FileNotFoundError(f"queue not found: {queue_path}")
    if not decisions_path.exists():
        raise FileNotFoundError(f"decisions not found: {decisions_path}")
    if not proposals_path.exists():
        raise FileNotFoundError(f"proposals not found: {proposals_path}")

    queue = load_json(queue_path)
    if not isinstance(queue, dict) or not isinstance(queue.get("items"), list):
        raise ManifestValidationError("queue JSON must contain an items list")

    decisions = load_json(decisions_path)
    proposals = load_json(proposals_path)
    priority = load_json(priority_path) if priority_path.exists() else {}

    data_root = _find_dataset_root(data_root_path)
    if data_root is None:
        raise FileNotFoundError(f"dataset root not found under {data_root_path}")

    _, _, class_names = load_data_config(data_root)

    proposals_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for p in proposals:
        name = Path(str(p.get("image", ""))).name
        proposals_by_image.setdefault(name, []).append(p)

    priority_by_image: Dict[str, Dict[str, Any]] = {}
    for e in priority.get("top20", []):
        priority_by_image[str(e.get("image_filename"))] = e

    return {
        "queue": queue,
        "decisions": decisions,
        "proposals": proposals,
        "proposals_by_image": proposals_by_image,
        "priority_by_image": priority_by_image,
        "class_names": class_names,
        "data_root": data_root,
    }


def _is_obsolete_uncertain(image_filename: str, decisions: Dict[str, Any]) -> bool:
    """Check if a D_UNCERTAIN manifest entry is obsolete because proposals were resolved.

    Per the lifecycle: after human decisions resolve uncertain proposals,
    the D_UNCERTAIN entry in the manifest becomes obsolete and should be skipped
    during queue validation. The human_decisions.json records carry the
    human decisions; if a D_UNCERTAIN image's proposals have a resolved human
    decision, the manifest entry is no longer a valid queue blocker.
    """
    records = decisions.get("records", [])
    for rec in records:
        rec_fn = rec.get("image_filename", "")
        # Match by image filename (basename comparison)
        if rec_fn == image_filename or rec_fn.endswith(image_filename):
            # If the record has been resolved (resolved_at present), the entry is obsolete
            if rec.get("resolved_at") or rec.get("human_decision"):
                return True
    return False


def init_manifest(source_queue: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "reviewer": "human",
        "source_queue": source_queue,
        "items": [],
        # Alias kept for readability in downstream audit tooling.
        "resolutions": [],
        "ui_state": {
            "last_index": 0,
            "last_filter": FILTER_UNRESOLVED,
            "last_sort": SORT_PRIORITY,
        },
    }


def load_or_init_manifest(path: Path, source_queue: str) -> Dict[str, Any]:
    if not path.exists():
        return init_manifest(source_queue=source_queue)
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ManifestValidationError("resolution manifest must be a JSON object")
    payload.setdefault("schema_version", 1)
    payload.setdefault("created_at", now_iso())
    payload.setdefault("updated_at", now_iso())
    payload.setdefault("reviewer", "human")
    payload.setdefault("source_queue", source_queue)
    payload.setdefault("items", [])
    payload.setdefault("ui_state", {"last_index": 0, "last_filter": FILTER_UNRESOLVED, "last_sort": SORT_PRIORITY})
    payload["resolutions"] = list(payload.get("items", []))
    return payload


def validate_manifest_schema(manifest: Dict[str, Any], queue_items: List[Dict[str, Any]], class_names: List[str], decisions: Dict[str, Any] = None) -> None:
    if not isinstance(manifest, dict):
        raise ManifestValidationError("manifest must be an object")
    if manifest.get("schema_version") != 1:
        raise ManifestValidationError("manifest schema_version must be 1")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ManifestValidationError("manifest items must be a list")

    queue_index = build_queue_index(queue_items)
    seen: set[Tuple[str, str]] = set()
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise ManifestValidationError(f"manifest item {idx} must be object")
        image_filename = entry.get("image_filename")
        category = entry.get("category")
        action = entry.get("action")
        if not image_filename or not category or not action:
            raise ManifestValidationError(f"manifest item {idx} missing image_filename/category/action")
        key = (str(image_filename), str(category))
        # Skip obsolete D_UNCERTAIN entries: if proposals were already resolved
        # (per human_decisions.json), the manifest entry is no longer a valid queue blocker.
        if category == "D_UNCERTAIN" and decisions:
            if _is_obsolete_uncertain(image_filename, decisions):
                continue
        if key in seen:
            raise ManifestValidationError(f"duplicate manifest resolution for {image_filename} {category}")
        seen.add(key)
        if key not in queue_index:
            raise ManifestValidationError(f"manifest item does not exist in queue: {key}")

        if category == CAT_EMPTY:
            if action not in EMPTY_ACTIONS:
                raise ManifestValidationError(f"invalid action {action} for {category}")
            if action == "MANUALLY_ANNOTATE":
                boxes = entry.get("boxes")
                if not isinstance(boxes, list) or not boxes:
                    raise ManifestValidationError("empty-label manual annotation requires non-empty boxes")
                for row in boxes:
                    err = validate_yolo_row(row, len(class_names))
                    if err:
                        raise ManifestValidationError(f"invalid YOLO box in empty-label item: {err}")
        elif category in (CAT_MANUAL, CAT_TIGHTEN):
            if action not in HUGEBOX_ACTIONS:
                raise ManifestValidationError(f"invalid action {action} for {category}")
            if action in ("TIGHTEN", "REPLACE"):
                row = entry.get("coordinates")
                err = validate_yolo_row(row, len(class_names)) if row is not None else "missing coordinates"
                if err:
                    raise ManifestValidationError(f"{action} requires valid coordinates: {err}")
        elif category == CAT_UNCERTAIN:
            if action not in UNCERTAIN_ACTIONS:
                raise ManifestValidationError(f"invalid action {action} for {category}")
            if action in ("ACCEPT_SELECTED", "REJECT_SELECTED", "CORRECT"):
                pids = entry.get("proposal_ids")
                if not isinstance(pids, list) or not pids:
                    raise ManifestValidationError(f"{action} requires proposal_ids")
        else:
            raise ManifestValidationError(f"unknown category: {category}")


def manifest_to_resolver_payload(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Convert GUI manifest into resolver-compatible apply payload.

    The resolver expects:
        {"items": [{category, image_filename, action, notes, proposal_ids?, coordinates?}]}

    For empty labels with multiple boxes, resolver currently accepts a single
    coordinate row. This adapter emits one item per box, with the final box
    preserving the intended decision on that image. This keeps compatibility
    while retaining full per-image box audit in the GUI manifest.
    """
    out_items: List[Dict[str, Any]] = []
    for item in manifest.get("items", []):
        cat = item.get("category")
        action = item.get("action")
        base = {
            "category": cat,
            "image_filename": item.get("image_filename"),
            "action": action,
            "notes": item.get("notes", ""),
        }

        if cat == CAT_EMPTY and action == "MANUALLY_ANNOTATE":
            boxes = item.get("boxes", [])
            if not boxes:
                out_items.append(base)
                continue
            # Resolver applies one row per item for this action.
            # Emit one item per box with the same image/category/action.
            for row in boxes:
                row_item = dict(base)
                row_item["coordinates"] = row
                out_items.append(row_item)
            continue

        if item.get("proposal_ids"):
            base["proposal_ids"] = list(item.get("proposal_ids"))
        if item.get("coordinates") is not None:
            base["coordinates"] = list(item.get("coordinates"))
        out_items.append(base)

    return {"items": out_items}


def validate_manifest_with_resolver(manifest: Dict[str, Any], frozen_state: Dict[str, Any]) -> None:
    payload = manifest_to_resolver_payload(manifest)
    # dry_run=True verifies that resolver can parse and apply all actions.
    apply_resolutions(copy.deepcopy(frozen_state), payload, dry_run=True)


def reviewed_keys(manifest: Dict[str, Any]) -> set[Tuple[str, str]]:
    return {manifest_item_key(x) for x in manifest.get("items", []) if isinstance(x, dict)}


def upsert_manifest_item(manifest: Dict[str, Any], item: Dict[str, Any]) -> bool:
    """Insert or replace one manifest item by image_filename+category.

    Returns True when a new key is added, False when an existing item is replaced.
    """
    items = manifest.setdefault("items", [])
    key = manifest_item_key(item)
    for idx, old in enumerate(items):
        if manifest_item_key(old) == key:
            items[idx] = item
            manifest["updated_at"] = now_iso()
            manifest["resolutions"] = list(items)
            return False
    items.append(item)
    manifest["updated_at"] = now_iso()
    manifest["resolutions"] = list(items)
    return True


def filtered_sorted_items(
    queue_items: List[Dict[str, Any]],
    resolved: set[Tuple[str, str]],
    filter_name: str,
    sort_name: str,
) -> List[Dict[str, Any]]:
    items = []
    for it in queue_items:
        key = queue_item_key(it)
        is_resolved = key in resolved
        priority = str(it.get("priority", "MEDIUM")).upper()
        category = it.get("category")

        if filter_name == FILTER_HIGH and priority != "HIGH":
            continue
        if filter_name == FILTER_EMPTY and category != CAT_EMPTY:
            continue
        if filter_name == FILTER_MANUAL and category != CAT_MANUAL:
            continue
        if filter_name == FILTER_TIGHTEN and category != CAT_TIGHTEN:
            continue
        if filter_name == FILTER_UNCERTAIN and category != CAT_UNCERTAIN:
            continue
        if filter_name == FILTER_UNRESOLVED and is_resolved:
            continue
        if filter_name == FILTER_RESOLVED and not is_resolved:
            continue
        items.append(it)

    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    def sort_key(x: Dict[str, Any]) -> Tuple[Any, ...]:
        priority = str(x.get("priority", "MEDIUM")).upper()
        if sort_name == SORT_PRIORITY:
            return (priority_rank.get(priority, 3), CATEGORY_ORDER.get(x.get("category"), 99), x.get("image_filename", ""))
        if sort_name == SORT_CATEGORY:
            return (CATEGORY_ORDER.get(x.get("category"), 99), priority_rank.get(priority, 3), x.get("image_filename", ""))
        if sort_name == SORT_SPLIT:
            return (str(x.get("split", "")), priority_rank.get(priority, 3), x.get("image_filename", ""))
        return (str(x.get("image_filename", "")),)

    items.sort(key=sort_key)
    return items


class BoxEditorState:
    """Mutable box editor state with undo/redo snapshots."""

    def __init__(self, boxes: List[DrawBox]):
        self.boxes: List[DrawBox] = copy.deepcopy(boxes)
        self._undo: List[List[DrawBox]] = []
        self._redo: List[List[DrawBox]] = []

    def snapshot(self) -> None:
        self._undo.append(copy.deepcopy(self.boxes))
        self._redo.clear()

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(copy.deepcopy(self.boxes))
        self.boxes = self._undo.pop()

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(copy.deepcopy(self.boxes))
        self.boxes = self._redo.pop()

    def reset(self, boxes: List[DrawBox]) -> None:
        self.snapshot()
        self.boxes = copy.deepcopy(boxes)


class V3HumanReviewApp:
    """Tkinter GUI for human adjudication of V3 blocker queue."""

    def __init__(
        self,
        root: tk.Tk,
        payloads: Dict[str, Any],
        manifest: Dict[str, Any],
        manifest_path: Path,
    ) -> None:
        self.root = root
        self.root.title("SmartFreshAI - V3 Human Review")
        self.root.geometry("1380x920")

        self.queue_items: List[Dict[str, Any]] = list(payloads["queue"]["items"])
        self.proposals_by_image: Dict[str, List[Dict[str, Any]]] = payloads["proposals_by_image"]
        self.class_names: List[str] = payloads["class_names"]
        self.class_id_by_name = {n: i for i, n in enumerate(self.class_names)}
        self.data_root: Path = payloads["data_root"]

        self.manifest = manifest
        self.manifest_path = manifest_path
        self.resolved_keys = reviewed_keys(self.manifest)

        ui_state = self.manifest.get("ui_state", {})
        self.filter_var = tk.StringVar(value=str(ui_state.get("last_filter", FILTER_UNRESOLVED)))
        self.sort_var = tk.StringVar(value=str(ui_state.get("last_sort", SORT_PRIORITY)))

        self.filtered_items: List[Dict[str, Any]] = []
        self.current_index = int(ui_state.get("last_index", 0))

        self._pil_original: Optional[Image.Image] = None
        self._display_photo: Optional[ImageTk.PhotoImage] = None
        self._image_w = 1
        self._image_h = 1
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        self.editor_mode = False
        self.active_tool = "select"
        self.editor_state = BoxEditorState([])
        self._editor_seed_boxes: List[DrawBox] = []
        self._selected_index: Optional[int] = None
        self._drag_kind: Optional[str] = None
        self._drag_start_canvas: Optional[Tuple[float, float]] = None
        self._drag_start_box: Optional[DrawBox] = None
        self._temp_draw_start: Optional[Tuple[float, float]] = None

        self._build_ui()
        self._bind_keys()
        self._refresh_queue_view(keep_index=True)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 4))

        self.header_var = tk.StringVar(value="SmartFreshAI - V3 Human Review")
        self.header = ttk.Label(top, textvariable=self.header_var, font=("Segoe UI", 15, "bold"))
        self.header.pack(side="left")

        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=8, pady=(2, 6))

        ttk.Label(toolbar, text="Filter:").pack(side="left")
        filter_combo = ttk.Combobox(
            toolbar,
            textvariable=self.filter_var,
            values=[
                FILTER_ALL,
                FILTER_HIGH,
                FILTER_EMPTY,
                FILTER_MANUAL,
                FILTER_TIGHTEN,
                FILTER_UNCERTAIN,
                FILTER_UNRESOLVED,
                FILTER_RESOLVED,
            ],
            state="readonly",
            width=18,
        )
        filter_combo.pack(side="left", padx=(4, 14))
        filter_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_queue_view(keep_index=False))

        ttk.Label(toolbar, text="Sort:").pack(side="left")
        sort_combo = ttk.Combobox(
            toolbar,
            textvariable=self.sort_var,
            values=[SORT_PRIORITY, SORT_CATEGORY, SORT_SPLIT, SORT_FILENAME],
            state="readonly",
            width=12,
        )
        sort_combo.pack(side="left", padx=(4, 12))
        sort_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_queue_view(keep_index=True))

        ttk.Button(toolbar, text="Prev [P]", command=self.prev_item).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Next [N]", command=self.next_item).pack(side="left", padx=2)

        ttk.Button(toolbar, text="Accept [A]", command=self.action_accept).pack(side="left", padx=8)
        ttk.Button(toolbar, text="Keep [K]", command=self.action_keep).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Reject [R]", command=self.action_reject).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Uncertain [U]", command=self.action_uncertain).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Edit [E]", command=self.enter_edit_mode).pack(side="left", padx=8)
        ttk.Button(toolbar, text="Skip [S]", command=self.next_item).pack(side="left", padx=2)

        self.progress_var = tk.StringVar(value="Progress: 0 / 0")
        ttk.Label(toolbar, textvariable=self.progress_var).pack(side="right")

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, background="#161616", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._render_current())
        self.canvas.bind("<Button-1>", self._on_canvas_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_up)

        right = ttk.Frame(main, width=360)
        right.pack(side="right", fill="y")

        ttk.Label(right, text="Metadata", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self.meta_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.meta_var, wraplength=340, justify="left").pack(anchor="w")

        ttk.Label(right, text="AI Proposals", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 4))
        self.proposal_list = tk.Listbox(right, selectmode="extended", width=52, height=15)
        self.proposal_list.pack(fill="x")

        self.policy_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.policy_var, foreground="#CC5500", wraplength=340, justify="left").pack(anchor="w", pady=(8, 4))

        edit_frame = ttk.LabelFrame(right, text="Edit Mode")
        edit_frame.pack(fill="x", pady=(8, 2))

        ttk.Button(edit_frame, text="Select", command=lambda: self._set_tool("select")).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(edit_frame, text="Draw", command=lambda: self._set_tool("draw")).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(edit_frame, text="Move", command=lambda: self._set_tool("move")).grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(edit_frame, text="Resize", command=lambda: self._set_tool("resize")).grid(row=0, column=3, sticky="ew", padx=2, pady=2)

        ttk.Button(edit_frame, text="Delete", command=self.delete_selected_box).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(edit_frame, text="Class", command=self.change_selected_class).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(edit_frame, text="Undo", command=self.undo_edit).grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(edit_frame, text="Redo", command=self.redo_edit).grid(row=1, column=3, sticky="ew", padx=2, pady=2)

        ttk.Button(edit_frame, text="Reset", command=self.reset_edit).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(edit_frame, text="Confirm", command=self.confirm_edit).grid(row=2, column=1, columnspan=3, sticky="ew", padx=2, pady=2)

        for i in range(4):
            edit_frame.grid_columnconfigure(i, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var).pack(fill="x", padx=8, pady=(2, 8))

    def _bind_keys(self) -> None:
        self.root.bind("<a>", lambda _e: self.action_accept())
        self.root.bind("<A>", lambda _e: self.action_accept())
        self.root.bind("<k>", lambda _e: self.action_keep())
        self.root.bind("<K>", lambda _e: self.action_keep())
        self.root.bind("<r>", lambda _e: self.action_reject())
        self.root.bind("<R>", lambda _e: self.action_reject())
        self.root.bind("<u>", lambda _e: self.action_uncertain())
        self.root.bind("<U>", lambda _e: self.action_uncertain())
        self.root.bind("<e>", lambda _e: self.enter_edit_mode())
        self.root.bind("<E>", lambda _e: self.enter_edit_mode())
        self.root.bind("<s>", lambda _e: self.next_item())
        self.root.bind("<S>", lambda _e: self.next_item())
        self.root.bind("<n>", lambda _e: self.next_item())
        self.root.bind("<N>", lambda _e: self.next_item())
        self.root.bind("<p>", lambda _e: self.prev_item())
        self.root.bind("<P>", lambda _e: self.prev_item())
        self.root.bind("<Delete>", lambda _e: self.delete_selected_box())
        self.root.bind("<Control-z>", lambda _e: self.undo_edit())
        self.root.bind("<Control-y>", lambda _e: self.redo_edit())

    def _refresh_queue_view(self, keep_index: bool) -> None:
        previous = None
        if keep_index and self.filtered_items and 0 <= self.current_index < len(self.filtered_items):
            previous = queue_item_key(self.filtered_items[self.current_index])

        self.filtered_items = filtered_sorted_items(
            self.queue_items,
            self.resolved_keys,
            self.filter_var.get(),
            self.sort_var.get(),
        )

        if not self.filtered_items:
            self.current_index = 0
            self._render_empty_state()
            self._persist_ui_state()
            return

        if previous is not None:
            for idx, item in enumerate(self.filtered_items):
                if queue_item_key(item) == previous:
                    self.current_index = idx
                    break
            else:
                self.current_index = min(self.current_index, len(self.filtered_items) - 1)
        else:
            self.current_index = min(self.current_index, len(self.filtered_items) - 1)

        self._render_current()
        self._persist_ui_state()

    def _render_empty_state(self) -> None:
        self.header_var.set("SmartFreshAI - V3 Human Review")
        self.meta_var.set("No items match current filter.")
        self.policy_var.set("")
        self.proposal_list.delete(0, tk.END)
        self.canvas.delete("all")
        self.progress_var.set("Progress: 0 / 0")
        self.status_var.set("No queue items in current view")

    def _current_item(self) -> Optional[Dict[str, Any]]:
        if not self.filtered_items:
            return None
        if self.current_index < 0 or self.current_index >= len(self.filtered_items):
            return None
        return self.filtered_items[self.current_index]

    def _render_current(self) -> None:
        item = self._current_item()
        if item is None:
            self._render_empty_state()
            return

        all_total = len(self.queue_items)
        done = len(self.resolved_keys)
        remaining = max(0, all_total - done)
        pct = (done / all_total * 100.0) if all_total else 0.0

        summary = summarize_queue(self.queue_items)
        unresolved_items = filtered_sorted_items(self.queue_items, self.resolved_keys, FILTER_UNRESOLVED, SORT_PRIORITY)
        unresolved_summary = summarize_queue(unresolved_items)

        priority = str(item.get("priority", "MEDIUM")).upper()
        category = item.get("category", "")
        self.header_var.set(
            f"SmartFreshAI - V3 Human Review   Image {self.current_index + 1} / {len(self.filtered_items)}   "
            f"Category: {category}   Priority: {priority}   Remaining: "
            f"Empty: {unresolved_summary['empty']}  Manual: {unresolved_summary['manual']}  "
            f"Tighten: {unresolved_summary['tighten']}  Uncertain: {unresolved_summary['uncertain']}"
        )
        self.progress_var.set(f"Progress: {done} / {all_total}   {pct:.2f}%   Resolved: {done}   Remaining: {remaining}")

        gt_count = len(item.get("original_annotation") or [])
        prop_count = len(item.get("proposals") or self.proposals_by_image.get(item.get("image_filename", ""), []))

        done_mark = "DONE" if queue_item_key(item) in self.resolved_keys else "PENDING"
        self.meta_var.set(
            f"Status: {done_mark}\n"
            f"Image: {item.get('image_filename')}\n"
            f"Split: {item.get('split')}\n"
            f"GT objects: {gt_count}\n"
            f"AI proposals: {prop_count}\n"
            f"Expected action: {item.get('expected_reviewer_action', '-') }"
        )

        self.policy_var.set(GRAPE_POLICY_TEXT if item.get("grape_warning") else "")

        self._load_image(item)
        self._populate_proposal_list(item)
        self.status_var.set("Use actions or keyboard shortcuts to review this image.")

    def _load_image(self, item: Dict[str, Any]) -> None:
        image_rel = str(item.get("image", "")).replace("\\", "/")
        image_path = _REPO_ROOT / image_rel
        self.canvas.delete("all")
        if not image_path.exists():
            self.status_var.set(f"Image missing: {image_path}")
            return
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"Failed to load image: {exc}")
            return

        self._pil_original = img
        self._image_w, self._image_h = img.size

        cw = max(120, self.canvas.winfo_width())
        ch = max(120, self.canvas.winfo_height())
        scale = min(cw / self._image_w, ch / self._image_h, 1.0)
        self._scale = scale
        disp_w = max(1, int(self._image_w * scale))
        disp_h = max(1, int(self._image_h * scale))
        disp = img.resize((disp_w, disp_h), Image.LANCZOS) if scale < 0.999 else img

        self._display_photo = ImageTk.PhotoImage(disp)
        self._offset_x = (cw - disp_w) // 2 if cw > disp_w else 0
        self._offset_y = (ch - disp_h) // 2 if ch > disp_h else 0

        self.canvas.create_image(self._offset_x, self._offset_y, anchor="nw", image=self._display_photo)

        if self.editor_mode:
            self._draw_editor_boxes()
        else:
            self._draw_gt_and_proposals(item)

    def _draw_gt_and_proposals(self, item: Dict[str, Any]) -> None:
        for row in item.get("original_annotation") or []:
            try:
                b = yolo_to_pixels(row, self._image_w, self._image_h, self.class_names, source="gt")
            except Exception:  # noqa: BLE001
                continue
            self._draw_box(b, GT_COLOR, label_prefix="GT")

        proposals = item.get("proposals") or self.proposals_by_image.get(item.get("image_filename", ""), [])
        for p in proposals:
            cls_name = str(p.get("class_name", "?"))
            conf = float(p.get("confidence", 0.0))
            b = DrawBox(
                class_id=int(p.get("class_id", 0)),
                class_name=cls_name,
                x1=float(p.get("x1", 0.0)),
                y1=float(p.get("y1", 0.0)),
                x2=float(p.get("x2", 0.0)),
                y2=float(p.get("y2", 0.0)),
                source="ai",
            )
            self._draw_box(b, AI_COLOR, label_prefix=f"{cls_name} {conf:.2f}")

    def _draw_editor_boxes(self) -> None:
        for idx, box in enumerate(self.editor_state.boxes):
            color = SELECT_COLOR if idx == self._selected_index else EDIT_COLOR
            self._draw_box(box, color, label_prefix=f"{box.class_name} [{idx}]", width=3 if idx == self._selected_index else 2)

    def _draw_box(self, box: DrawBox, color: str, label_prefix: str, width: int = 2) -> None:
        b = box.normalized()
        x1 = self._offset_x + b.x1 * self._scale
        y1 = self._offset_y + b.y1 * self._scale
        x2 = self._offset_x + b.x2 * self._scale
        y2 = self._offset_y + b.y2 * self._scale
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
        self.canvas.create_text(x1 + 2, max(8, y1 - 6), anchor="sw", text=label_prefix, fill=color, font=("Consolas", 9, "bold"))

    def _populate_proposal_list(self, item: Dict[str, Any]) -> None:
        self.proposal_list.delete(0, tk.END)
        proposals = item.get("proposals") or self.proposals_by_image.get(item.get("image_filename", ""), [])
        for idx, p in enumerate(proposals):
            self.proposal_list.insert(
                tk.END,
                f"{idx + 1:3d}. {p.get('class_name', '?'):10s} conf={float(p.get('confidence', 0.0)):.2f} "
                f"box=({int(float(p.get('x1', 0)))} {int(float(p.get('y1', 0)))} {int(float(p.get('x2', 0)))} {int(float(p.get('y2', 0)))})"
            )
            self.proposal_list.selection_set(idx)

    def _persist_ui_state(self) -> None:
        self.manifest["ui_state"] = {
            "last_index": self.current_index,
            "last_filter": self.filter_var.get(),
            "last_sort": self.sort_var.get(),
        }
        atomic_write_json(self.manifest_path, self.manifest)

    def next_item(self) -> None:
        if not self.filtered_items:
            return
        self.current_index = min(self.current_index + 1, len(self.filtered_items) - 1)
        self.editor_mode = False
        self._selected_index = None
        self._render_current()
        self._persist_ui_state()

    def prev_item(self) -> None:
        if not self.filtered_items:
            return
        self.current_index = max(self.current_index - 1, 0)
        self.editor_mode = False
        self._selected_index = None
        self._render_current()
        self._persist_ui_state()

    def _set_tool(self, tool: str) -> None:
        self.active_tool = tool
        self.status_var.set(f"Edit tool: {tool}")

    def enter_edit_mode(self) -> None:
        item = self._current_item()
        if item is None:
            return

        self.editor_mode = True
        self._selected_index = None
        seed = self._seed_edit_boxes(item)
        self._editor_seed_boxes = copy.deepcopy(seed)
        self.editor_state = BoxEditorState(seed)
        self._set_tool("draw" if item.get("category") == CAT_EMPTY else "select")

        self.status_var.set(
            "Edit mode enabled: draw/move/resize boxes, then Confirm to save resolution payload."
        )
        self._load_image(item)

    def _seed_edit_boxes(self, item: Dict[str, Any]) -> List[DrawBox]:
        category = item.get("category")
        boxes: List[DrawBox] = []

        if category in (CAT_EMPTY, CAT_MANUAL, CAT_TIGHTEN):
            for row in item.get("original_annotation") or []:
                boxes.append(yolo_to_pixels(row, self._image_w, self._image_h, self.class_names, source="gt"))
            if category == CAT_EMPTY and not boxes:
                default_name = str(item.get("class_name") or self.class_names[0])
                default_id = self.class_id_by_name.get(default_name, 0)
                boxes = []
                # User must draw real boxes; keep empty until drawn.
            return boxes

        # Uncertain category seeds with AI proposals.
        proposals = item.get("proposals") or self.proposals_by_image.get(item.get("image_filename", ""), [])
        for p in proposals:
            cid = int(p.get("class_id", 0))
            cname = str(p.get("class_name", self.class_names[cid] if cid < len(self.class_names) else cid))
            boxes.append(
                DrawBox(
                    class_id=cid,
                    class_name=cname,
                    x1=float(p.get("x1", 0.0)),
                    y1=float(p.get("y1", 0.0)),
                    x2=float(p.get("x2", 0.0)),
                    y2=float(p.get("y2", 0.0)),
                    source="ai",
                )
            )
        return boxes

    def _canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self._offset_x) / self._scale, (y - self._offset_y) / self._scale)

    def _find_box_at(self, canvas_x: float, canvas_y: float) -> Optional[int]:
        ix, iy = self._canvas_to_image(canvas_x, canvas_y)
        for idx in reversed(range(len(self.editor_state.boxes))):
            b = self.editor_state.boxes[idx].normalized()
            if b.x1 <= ix <= b.x2 and b.y1 <= iy <= b.y2:
                return idx
        return None

    def _on_canvas_down(self, event: tk.Event) -> None:
        if not self.editor_mode:
            return
        idx = self._find_box_at(event.x, event.y)
        if self.active_tool == "draw":
            self._temp_draw_start = (event.x, event.y)
            self._selected_index = None
            return

        if idx is not None:
            self._selected_index = idx
            self._drag_start_canvas = (event.x, event.y)
            self._drag_start_box = copy.deepcopy(self.editor_state.boxes[idx])
            self._drag_kind = self.active_tool if self.active_tool in ("move", "resize") else None
        else:
            self._selected_index = None
            self._drag_kind = None
        self._load_image(self._current_item() or {})

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if not self.editor_mode:
            return
        if self.active_tool == "draw" and self._temp_draw_start is not None:
            # Draw a temporary rectangle overlay.
            self._load_image(self._current_item() or {})
            sx, sy = self._temp_draw_start
            self.canvas.create_rectangle(sx, sy, event.x, event.y, outline=SELECT_COLOR, width=2, dash=(3, 2))
            return

        if self._selected_index is None or self._drag_kind is None or self._drag_start_canvas is None or self._drag_start_box is None:
            return

        ix0, iy0 = self._canvas_to_image(*self._drag_start_canvas)
        ix1, iy1 = self._canvas_to_image(event.x, event.y)
        dx = ix1 - ix0
        dy = iy1 - iy0

        b = copy.deepcopy(self._drag_start_box)
        if self._drag_kind == "move":
            b.x1 += dx
            b.x2 += dx
            b.y1 += dy
            b.y2 += dy
        elif self._drag_kind == "resize":
            b.x2 += dx
            b.y2 += dy

        self.editor_state.boxes[self._selected_index] = b
        self._load_image(self._current_item() or {})

    def _on_canvas_up(self, event: tk.Event) -> None:
        if not self.editor_mode:
            return

        if self.active_tool == "draw" and self._temp_draw_start is not None:
            sx, sy = self._temp_draw_start
            self._temp_draw_start = None
            ix1, iy1 = self._canvas_to_image(sx, sy)
            ix2, iy2 = self._canvas_to_image(event.x, event.y)
            class_name = self._default_class_for_current()
            class_id = self.class_id_by_name.get(class_name, 0)
            new_box = DrawBox(class_id, class_name, ix1, iy1, ix2, iy2, source="manual")
            try:
                _ = pixels_to_yolo(new_box, self._image_w, self._image_h)
            except ManifestValidationError as exc:
                self.status_var.set(f"Rejected box: {exc}")
                self._load_image(self._current_item() or {})
                return
            self.editor_state.snapshot()
            self.editor_state.boxes.append(new_box.normalized())
            self._selected_index = len(self.editor_state.boxes) - 1
            self._load_image(self._current_item() or {})
            return

        if self._selected_index is not None and self._drag_kind in ("move", "resize") and self._drag_start_box is not None:
            try:
                _ = pixels_to_yolo(self.editor_state.boxes[self._selected_index], self._image_w, self._image_h)
            except ManifestValidationError as exc:
                # Revert invalid transformation.
                self.editor_state.boxes[self._selected_index] = self._drag_start_box
                self.status_var.set(f"Invalid geometry reverted: {exc}")
            else:
                self.editor_state.snapshot()
        self._drag_kind = None
        self._drag_start_canvas = None
        self._drag_start_box = None
        self._load_image(self._current_item() or {})

    def _default_class_for_current(self) -> str:
        item = self._current_item() or {}
        if item.get("category") == CAT_EMPTY and item.get("class_name"):
            return str(item.get("class_name"))
        if self.class_names:
            return self.class_names[0]
        return "0"

    def delete_selected_box(self) -> None:
        if not self.editor_mode or self._selected_index is None:
            return
        if self._selected_index < 0 or self._selected_index >= len(self.editor_state.boxes):
            return
        self.editor_state.snapshot()
        del self.editor_state.boxes[self._selected_index]
        self._selected_index = None
        self._load_image(self._current_item() or {})

    def change_selected_class(self) -> None:
        if not self.editor_mode or self._selected_index is None:
            return
        b = self.editor_state.boxes[self._selected_index]
        cls = simpledialog.askstring("Class", "Enter class name", initialvalue=b.class_name, parent=self.root)
        if not cls:
            return
        cls = cls.strip()
        if cls not in self.class_id_by_name:
            messagebox.showerror("Invalid class", f"Class must be one of: {', '.join(self.class_names)}")
            return
        self.editor_state.snapshot()
        b.class_name = cls
        b.class_id = self.class_id_by_name[cls]
        self._load_image(self._current_item() or {})

    def undo_edit(self) -> None:
        if not self.editor_mode:
            return
        self.editor_state.undo()
        self._load_image(self._current_item() or {})

    def redo_edit(self) -> None:
        if not self.editor_mode:
            return
        self.editor_state.redo()
        self._load_image(self._current_item() or {})

    def reset_edit(self) -> None:
        if not self.editor_mode:
            return
        self.editor_state.reset(self._editor_seed_boxes)
        self._selected_index = None
        self._load_image(self._current_item() or {})

    def confirm_edit(self) -> None:
        item = self._current_item()
        if item is None:
            return

        category = item.get("category")
        try:
            yolo_rows = [pixels_to_yolo(b, self._image_w, self._image_h) for b in self.editor_state.boxes]
        except ManifestValidationError as exc:
            messagebox.showerror("Invalid boxes", str(exc))
            return

        if category == CAT_EMPTY and not yolo_rows:
            messagebox.showerror("Missing annotation", "Empty-label images require at least one manually drawn box.")
            return

        if category in (CAT_EMPTY, CAT_MANUAL, CAT_TIGHTEN):
            note = simpledialog.askstring("Notes", "Notes for this resolution", initialvalue="manual edit", parent=self.root) or "manual edit"
            if category == CAT_EMPTY:
                manifest_item = {
                    "image": item.get("image"),
                    "image_filename": item.get("image_filename"),
                    "split": item.get("split"),
                    "category": category,
                    "decision": "annotate",
                    "action": "MANUALLY_ANNOTATE",
                    "boxes": yolo_rows,
                    "notes": note,
                    "timestamp": now_iso(),
                }
            else:
                if len(yolo_rows) != 1:
                    messagebox.showerror(
                        "Single box required",
                        "Manual-review and tighten items require exactly one final box for resolver compatibility.",
                    )
                    return
                manifest_item = {
                    "image": item.get("image"),
                    "image_filename": item.get("image_filename"),
                    "split": item.get("split"),
                    "category": category,
                    "decision": "tighten",
                    "action": "TIGHTEN",
                    "coordinates": yolo_rows[0],
                    "boxes": yolo_rows,
                    "notes": note,
                    "timestamp": now_iso(),
                }
            self._save_resolution(manifest_item)
            self.editor_mode = False
            self.next_item()
            return

        if category == CAT_UNCERTAIN:
            # Editing uncertain proposals is tracked as CORRECT with selected proposal ids.
            selected = self._selected_proposal_ids(item)
            if not selected:
                messagebox.showerror("Selection required", "Select one or more proposal rows for CORRECT action.")
                return
            note = simpledialog.askstring("Notes", "Notes for corrected proposals", initialvalue="corrected in editor", parent=self.root) or "corrected in editor"
            manifest_item = {
                "image": item.get("image"),
                "image_filename": item.get("image_filename"),
                "split": item.get("split"),
                "category": category,
                "decision": "corrected",
                "action": "CORRECT",
                "proposal_ids": selected,
                "boxes": yolo_rows,
                "notes": note,
                "timestamp": now_iso(),
            }
            self._save_resolution(manifest_item)
            self.editor_mode = False
            self.next_item()

    def _selected_proposal_ids(self, item: Dict[str, Any]) -> List[str]:
        proposals = item.get("proposals") or self.proposals_by_image.get(item.get("image_filename", ""), [])
        selected_idx = list(self.proposal_list.curselection())
        ids: List[str] = []
        for i in selected_idx:
            if 0 <= i < len(proposals):
                pid = proposals[i].get("proposal_id")
                if pid is not None:
                    ids.append(str(pid))
        return ids

    def action_accept(self) -> None:
        item = self._current_item()
        if item is None:
            return

        category = item.get("category")
        if category == CAT_EMPTY:
            if not self.editor_mode:
                messagebox.showinfo("Manual required", "Empty-label images require manual annotation mode. Press Edit and draw real boxes.")
                return
            self.confirm_edit()
            return

        if category in (CAT_MANUAL, CAT_TIGHTEN):
            if messagebox.askyesno("Keep or tighten", "For manual/tighten cases, Accept maps to KEEP original box. Continue?"):
                self.action_keep()
            return

        # Uncertain image-level accept with explicit checklist confirmation.
        selected_ids = self._selected_proposal_ids(item)
        proposals = item.get("proposals") or self.proposals_by_image.get(item.get("image_filename", ""), [])
        if not selected_ids:
            messagebox.showerror("No proposals selected", "Select one or more AI proposals before accepting.")
            return

        checklist = (
            f"You are accepting {len(selected_ids)} AI boxes for this image.\n\n"
            "Verify:\n"
            "- objects are real\n"
            "- classes are correct\n"
            "- boxes cover the correct objects\n"
            "- no duplicate boxes\n"
            "- no major missing objects\n\n"
            "Accept?"
        )
        if not messagebox.askyesno("Confirm accept", checklist):
            return

        action = "ACCEPT_ALL" if len(selected_ids) == len(proposals) else "ACCEPT_SELECTED"
        manifest_item = {
            "image": item.get("image"),
            "image_filename": item.get("image_filename"),
            "split": item.get("split"),
            "category": category,
            "decision": "accept",
            "action": action,
            "proposal_ids": selected_ids,
            "boxes": [],
            "notes": "human accepted selected AI proposals",
            "timestamp": now_iso(),
        }
        self._save_resolution(manifest_item)
        self.next_item()

    def action_keep(self) -> None:
        item = self._current_item()
        if item is None:
            return

        category = item.get("category")
        if category == CAT_UNCERTAIN:
            manifest_item = {
                "image": item.get("image"),
                "image_filename": item.get("image_filename"),
                "split": item.get("split"),
                "category": category,
                "decision": "kept",
                "action": "KEEP_ORIGINAL",
                "proposal_ids": [],
                "boxes": [],
                "notes": "human kept original annotations",
                "timestamp": now_iso(),
            }
        elif category == CAT_EMPTY:
            if not messagebox.askyesno("Confirm background", "Mark image as true background (no fruit visible)?"):
                return
            manifest_item = {
                "image": item.get("image"),
                "image_filename": item.get("image_filename"),
                "split": item.get("split"),
                "category": category,
                "decision": "keep_empty",
                "action": "CONFIRM_BACKGROUND",
                "boxes": [],
                "notes": "human confirmed background",
                "timestamp": now_iso(),
            }
        else:
            manifest_item = {
                "image": item.get("image"),
                "image_filename": item.get("image_filename"),
                "split": item.get("split"),
                "category": category,
                "decision": "keep",
                "action": "KEEP",
                "boxes": [],
                "notes": "human kept original box",
                "timestamp": now_iso(),
            }

        self._save_resolution(manifest_item)
        self.next_item()

    def action_reject(self) -> None:
        item = self._current_item()
        if item is None:
            return

        category = item.get("category")
        if category != CAT_UNCERTAIN:
            messagebox.showinfo("Reject action", "Reject is only for uncertain AI proposal images.")
            return

        mode = messagebox.askyesnocancel(
            "Reject AI proposals",
            "Reject all AI proposals and keep original annotations?\n"
            "Yes = Keep original. No = Manually correct (open editor). Cancel = abort.",
        )
        if mode is None:
            return
        if mode:
            manifest_item = {
                "image": item.get("image"),
                "image_filename": item.get("image_filename"),
                "split": item.get("split"),
                "category": category,
                "decision": "rejected",
                "action": "KEEP_ORIGINAL",
                "proposal_ids": [],
                "boxes": [],
                "notes": "rejected AI proposals; kept original",
                "timestamp": now_iso(),
            }
            self._save_resolution(manifest_item)
            self.next_item()
            return

        self.enter_edit_mode()
        self.status_var.set("Manual correction mode: edit boxes and press Confirm.")

    def action_uncertain(self) -> None:
        item = self._current_item()
        if item is None:
            return

        category = item.get("category")
        action = "MARK_UNCERTAIN" if category == CAT_EMPTY else "UNCERTAIN"
        manifest_item = {
            "image": item.get("image"),
            "image_filename": item.get("image_filename"),
            "split": item.get("split"),
            "category": category,
            "decision": "uncertain",
            "action": action,
            "proposal_ids": [],
            "boxes": [],
            "notes": "human marked uncertain",
            "timestamp": now_iso(),
        }
        self._save_resolution(manifest_item)
        self.next_item()

    def _save_resolution(self, item: Dict[str, Any]) -> None:
        # Validate schema first against queue and class rules.
        trial = copy.deepcopy(self.manifest)
        upsert_manifest_item(trial, item)
        validate_manifest_schema(trial, self.queue_items, self.class_names)

        # Persist atomically.
        upsert_manifest_item(self.manifest, item)
        atomic_write_json(self.manifest_path, self.manifest)

        self.resolved_keys = reviewed_keys(self.manifest)
        self.status_var.set(f"Saved resolution for {item.get('image_filename')} [{item.get('category')}].")
        self._refresh_queue_view(keep_index=True)


def make_frozen_state_for_resolver(payloads: Dict[str, Any]) -> Dict[str, Any]:
    """Create resolver-compatible state dict for dry-run compatibility checks."""
    # Manual annotations are loaded by resolver internally in normal flow; for
    # dry-run compatibility check we can pass an empty manual tuple.
    return {
        "data_root": payloads["data_root"],
        "class_names": payloads["class_names"],
        "decisions": payloads["decisions"],
        "proposals": payloads["proposals"],
        "manual": ({}, {}),
    }


def cli_validate_manifest(
    manifest_path: Path,
    queue_path: Path,
    decisions_path: Path,
    priority_path: Path = None,
    data_root_path: Path = None,
) -> int:
    if data_root_path is None:
        data_root_path = Path("data/detection")
    payloads = load_review_payloads(
        queue_path=queue_path,
        decisions_path=decisions_path,
        proposals_path=PROPOSALS_PATH,
        priority_path=priority_path or PRIORITY_PATH,
        data_root_path=data_root_path,
    )
    manifest = load_or_init_manifest(manifest_path, str(queue_path).replace("\\", "/"))

    # Use decisions from payloads to detect obsolete D_UNCERTAIN entries
    decisions = payloads.get("decisions")

    validate_manifest_schema(manifest, payloads["queue"]["items"], payloads["class_names"], decisions)
    validate_manifest_with_resolver(manifest, make_frozen_state_for_resolver(payloads))

    print("Manifest validation passed")
    print(f"  path: {manifest_path}")
    print(f"  items: {len(manifest.get('items', []))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SmartFreshAI V3 human gate review GUI")
    p.add_argument("--queue", type=Path, default=QUEUE_PATH, help="Queue JSON path")
    p.add_argument("--decisions", type=Path, default=DECISIONS_PATH, help="human_decisions.json path")
    p.add_argument("--proposals", type=Path, default=PROPOSALS_PATH, help="proposals.json path")
    p.add_argument("--priority", type=Path, default=PRIORITY_PATH, help="review_priority.json path")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT_PATH, help="Frozen V2 dataset root")
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Resolution manifest output path")
    p.add_argument("--validate-only", action="store_true", help="Validate manifest without opening GUI")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    payloads = load_review_payloads(
        queue_path=args.queue,
        decisions_path=args.decisions,
        proposals_path=args.proposals,
        priority_path=args.priority,
        data_root_path=args.data_root,
    )

    manifest = load_or_init_manifest(args.manifest, str(args.queue).replace("\\", "/"))

    # Use decisions from payloads to detect obsolete D_UNCERTAIN entries
    decisions = payloads.get("decisions")

    # Enforce schema early and persist bootstrap manifest atomically.
    validate_manifest_schema(manifest, payloads["queue"]["items"], payloads["class_names"], decisions)
    atomic_write_json(args.manifest, manifest)

    if args.validate_only:
        return cli_validate_manifest(
            manifest_path=args.manifest,
            queue_path=args.queue,
            decisions_path=args.decisions,
            priority_path=args.priority,
            data_root_path=args.data_root,
        )

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"ERROR: Tkinter UI unavailable: {exc}", file=sys.stderr)
        return 2

    app = V3HumanReviewApp(root, payloads, manifest, args.manifest)
    root.mainloop()

    # Final compatibility check before exit.
    validate_manifest_schema(app.manifest, payloads["queue"]["items"], payloads["class_names"])
    validate_manifest_with_resolver(app.manifest, make_frozen_state_for_resolver(payloads))
    atomic_write_json(args.manifest, app.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
