#!/usr/bin/env python3
"""GUI & CLI tool for human review of AI-assisted annotation proposals.

The reviewer works **one image at a time** (all of that image's AI proposals at
once). A reviewer who decides the entire proposal set for an image is correct
can ACCEPT ALL in a single action — no per-box confirmation is required.

Supported actions per image:
    ACCEPT ALL      : approve every AI proposal as an approved annotation
    REJECT ALL      : reject every AI proposal for this image
    KEEP ORIGINAL   : keep the original ground-truth, ignore AI proposals
    EDIT            : open the box editor (move / resize individual boxes)
    DELETE PROPOSAL : remove one selected AI proposal
    ADD BOX         : append a new (empty) box to the proposal set
    CHANGE CLASS    : change the class of the selected proposal
    MOVE BOX        : move the selected box (drag)
    RESIZE BOX      : resize the selected box (drag)
    MARK UNCERTAIN  : record the image as uncertain (stays unresolved)
    SKIP            : leave the image pending (no decision written)

Decision semantics (Phase 3.6):
    AI proposal != approved annotation.
    Only explicit human action creates approved / corrected / kept / rejected /
    excluded / uncertain. A skipped image remains pending; an uncertain image
    remains unresolved.

Usage:
    python scripts/review_annotation_proposals.py

Review is an interactive, one-image-at-a-time GUI. AI proposals are never
auto-approved: every proposal remains a proposal until a human reviewer
explicitly accepts it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageTk

# Ensure repository root is in the Python path.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.prioritize_annotation_review import (  # noqa: E402
    iou,
    load_proposals_by_image,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROPOSAL_FILE = Path("reports/audit_review/ai_annotation_proposals/proposals.json")
DECISION_FILE = Path("reports/audit_review/human_decisions.json")
VISUALIZATION_DIR = Path("reports/audit_review/ai_annotation_proposals/visualizations")
REPORT_JSON = Path("reports/audit_review/ai_annotation_review_report.json")
REPORT_MD = Path("reports/audit_review/ai_annotation_review_report.md")
PRIORITY_JSON = Path("reports/audit_review/review_priority.json")


def compute_priority(proposal: dict) -> str:
    """Legacy single-proposal priority helper (kept for test compatibility).

    The authoritative priority now comes from the review_priority.json report
    (per-image, evidence-based). This function remains for CLI batch mode.
    """
    conf = proposal.get("confidence", 0.0)
    cat = proposal.get("review_category", "")
    if cat == "ambiguous_classes" or conf >= 0.85:
        return "HIGH"
    if conf >= 0.50:
        return "MEDIUM"
    return "LOW"


# --- Coordinate transforms ---------------------------------------------------
# The displayed image is placed on the canvas at offset (ox, oy) with its
# top-left corner there, scaled by ``scale``. Every overlay (proposal boxes,
# ground-truth boxes, labels, drag handles) and every mouse conversion MUST use
# the SAME transform so that boxes land on the displayed image, never on the
# surrounding canvas margin.
#
#   image coords  --(scale, then +offset)-->  canvas coords
#   canvas coords --( -offset, then /scale)--> image coords


def img_rect_to_canvas(x1, y1, x2, y2, scale, ox, oy):
    """Map a source-image rectangle to canvas coordinates.

    ``(x1, y1, x2, y2)`` are source-image pixel coordinates (unchanged). The
    canvas rectangle is ``ox + x*scale`` / ``oy + y*scale`` for each edge.
    """
    return (ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale)


def canvas_point_to_img(cx, cy, scale, ox, oy):
    """Inverse transform: map a canvas point back to source-image coordinates."""
    return ((cx - ox) / scale, (cy - oy) / scale)


def load_proposals() -> list:
    """Load the flat list of proposals (backwards-compatible)."""
    if not PROPOSAL_FILE.exists():
        logger.error("Proposal file missing: %s", PROPOSAL_FILE)
        return []
    with open(PROPOSAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_human_decisions() -> dict:
    if DECISION_FILE.exists():
        with open(DECISION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": 1,
        "source": "human-adjudication",
        "notes": ["Extended with AI-assisted proposals review decisions."],
        "record_count": 0,
        "records": [],
    }


def save_human_decisions(decisions_data: dict):
    decisions_data["record_count"] = len(decisions_data["records"])
    with open(DECISION_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions_data, f, indent=2)
def load_priority_map() -> dict:
    """image_filename -> per-image priority evidence from review_priority.json."""
    if not PRIORITY_JSON.exists():
        return {}
    with open(PRIORITY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for e in data.get("top20", []):
        result[e["image_filename"]] = {
            "score": e.get("score"),
            "priority": e.get("priority"),
            "category": e.get("category"),
        }
    # Add aggregated evidence for all other images too, if present.
    return result


def group_proposals_by_image(proposals: list, priority_map: dict | None = None) -> list:
    """Group the flat proposal list into per-image review units.

    Each unit groups all of one image's AI proposals. Units are ordered by
    review priority (HIGH first, then score descending) when available.
    """
    priority_map = priority_map or {}
    by_img: dict = {}
    for p in proposals:
        fname = Path(p["image"]).name
        unit = by_img.setdefault(fname, {
            "image": p["image"],
            "image_filename": fname,
            "split": p.get("split"),
            "review_category": p.get("review_category"),
            "proposals": [],
        })
        unit["proposals"].append(p)

    _LEVEL_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    for unit in by_img.values():
        prio = priority_map.get(unit["image_filename"], {})
        unit["priority_level"] = prio.get("priority", "MEDIUM")
        unit["priority_score"] = prio.get("score", 0)
        unit["priority_category"] = prio.get("category", unit.get("review_category"))
        if unit["priority_level"] not in _LEVEL_RANK:
            unit["priority_level"] = "MEDIUM"

    return sorted(
        by_img.values(),
        key=lambda u: (_LEVEL_RANK[u["priority_level"]], -(u["priority_score"] or 0), u["image_filename"]),
    )


def generate_review_report(units: list, decisions: dict):
    """Write the AI annotation review report (per-image summary)."""
    total = len(units)
    dec_records = decisions.get("records", [])

    accepted = sum(1 for r in dec_records if r.get("human_decision") == "accepted")
    corrected = sum(1 for r in dec_records if r.get("human_decision") == "corrected")
    kept = sum(1 for r in dec_records if r.get("human_decision") == "kept")
    rejected = sum(1 for r in dec_records if r.get("human_decision") == "rejected")
    excluded = sum(1 for r in dec_records if r.get("human_decision") == "excluded")
    uncertain = sum(1 for r in dec_records if r.get("human_decision") == "uncertain")

    processed = accepted + corrected + kept + rejected + excluded + uncertain
    pending = max(0, total - processed)

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_unresolved_cases": total,
        "processed": processed,
        "pending": pending,
        "accepted": accepted,
        "corrected": corrected,
        "kept": kept,
        "rejected": rejected,
        "excluded": excluded,
        "uncertain": uncertain,
        "acceptance_rate": round(accepted / processed, 4) if processed > 0 else 0.0,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    md_content = (
        "# AI Annotation Review Report\n\n"
        f"- **Timestamp:** {report_data['timestamp']}\n"
        f"- **Total Unresolved Cases:** {total}\n"
        f"- **Processed:** {processed}\n"
        f"- **Pending:** {pending}\n"
        f"- **Accepted:** {accepted}\n"
        f"- **Corrected:** {corrected}\n"
        f"- **Kept:** {kept}\n"
        f"- **Rejected:** {rejected}\n"
        f"- **Excluded:** {excluded}\n"
        f"- **Uncertain:** {uncertain}\n"
        f"- **Acceptance Rate:** {report_data['acceptance_rate'] * 100:.2f}%\n"
    )
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved review reports to %s and %s", REPORT_JSON, REPORT_MD)
# --- GUI ----------------------------------------------------------------------

class AnnotationReviewGUI:
    """One-image-at-a-time review GUI with a full editing toolset."""

    _LEVEL_COLOR = {"HIGH": "#d9534f", "MEDIUM": "#f0ad4e", "LOW": "#5cb85c"}

    def __init__(self, root, units, decisions_data=None):
        self.root = root
        self.root.title("SmartFreshAI - Annotation Review (per-image)")
        self.root.geometry("1180x820")

        self.units = units
        self.index = 0
        self.decisions_data = decisions_data if decisions_data is not None else load_human_decisions()

        # Display scale for overlaying proposals on the image.
        self.canvas_img = None
        self._pil_image = None
        self._img_w = 1
        self._img_h = 1
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._selected_prop_id = None

        self._build_layout()
        self._bind_keys()
        self.load_current_unit()

    # ---- layout --------------------------------------------------------------
    def _build_layout(self):
        self.header = ttk.Label(self.root, text="Review: AI Annotation Proposals",
                                font=("Helvetica", 16, "bold"))
        self.header.pack(pady=6)

        # Info line: image number / total, priority, category, counts.
        self.info_text = tk.StringVar()
        self.info = ttk.Label(self.root, textvariable=self.info_text,
                              font=("Helvetica", 11), wraplength=1140, justify="left")
        self.info.pack(fill="x", padx=20, pady=2)

        # Stats line: GT/AI counts, classes, confidence, IoU.
        self.stat_text = tk.StringVar()
        self.stats = ttk.Label(self.root, textvariable=self.stat_text,
                               font=("Helvetica", 10), wraplength=1140, justify="left")
        self.stats.pack(fill="x", padx=20, pady=2)

        self.detail_text = tk.StringVar()
        self.detail = ttk.Label(self.root, textvariable=self.detail_text,
                                font=("Consolas", 9), wraplength=1140, justify="left")
        self.detail.pack(fill="x", padx=20, pady=2)

        self.btn_image = ttk.Frame(self.root)
        self.btn_image.pack(fill="x", padx=20, pady=(8, 2))
        self._make_image_buttons()

        # Canvas to display the image + overlaid proposals.
        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True, padx=12, pady=6)
        self.canvas.bind("<Configure>", lambda e: self.render_image())
        self._init_drag()

        self.action_frame = ttk.Frame(self.root)
        self.action_frame.pack(fill="x", padx=20, pady=8)
        self._make_action_buttons()

        self.status = tk.StringVar(value="Ready. Select a proposal then use the action buttons.")
        ttk.Label(self.root, textvariable=self.status, foreground="#555").pack(fill="x", padx=20, pady=(2, 6))

    def _make_image_buttons(self):
        ttk.Button(self.btn_image, text="LOAD IMAGE", command=self.load_image_file).pack(side="left", padx=3)
        ttk.Button(self.btn_image, text="SELECT PROPOSAL", command=self.select_proposal_prompt).pack(side="left", padx=3)
        ttk.Button(self.btn_image, text="ADD BOX", command=self.add_box).pack(side="left", padx=3)

    def _make_action_buttons(self):
        btns = [
            ("ACCEPT ALL (a)", self.accept_all),
            ("REJECT ALL (r)", self.reject_all),
            ("KEEP ORIGINAL (k)", self.keep_original),
            ("EDIT (e)", self.edit_proposal),
            ("DELETE PROPOSAL (d)", self.delete_proposal),
            ("CHANGE CLASS (c)", self.change_class),
            ("MOVE BOX (m)", self.move_box),
            ("RESIZE BOX (z)", self.resize_box),
            ("MARK UNCERTAIN (u)", self.mark_uncertain),
            ("SKIP / NEXT (Right)", self.next_item),
        ]
        for text, cmd in btns:
            ttk.Button(self.action_frame, text=text, command=cmd).pack(side="left", padx=3)

    def _bind_keys(self):
        self.root.bind("<a>", lambda e: self.accept_all())
        self.root.bind("A", lambda e: self.accept_all())
        self.root.bind("<r>", lambda e: self.reject_all())
        self.root.bind("<k>", lambda e: self.keep_original())
        self.root.bind("<e>", lambda e: self.edit_proposal())
        self.root.bind("<d>", lambda e: self.delete_proposal())
        self.root.bind("<c>", lambda e: self.change_class())
        self.root.bind("<m>", lambda e: self.move_box())
        self.root.bind("<z>", lambda e: self.resize_box())
        self.root.bind("<u>", lambda e: self.mark_uncertain())
        self.root.bind("<Right>", lambda e: self.next_item())
        self.root.bind("<Left>", lambda e: self.prev_item())
# ---- unit navigation -----------------------------------------------------
    def load_current_unit(self):
        if self.index >= len(self.units):
            generate_review_report(self.units, self.decisions_data)
            messagebox.showinfo("Done", "All images reviewed!")
            self.root.quit()
            return
        unit = self.units[self.index]
        self._selected_prop_id = None
        self._update_info(unit)
        self.load_image_file()

    def _unit(self) -> dict:
        return self.units[self.index]

    def _proposals(self) -> list:
        return self._unit().get("proposals", [])

    def _update_info(self, unit):
        prio = unit.get("priority_level", "MEDIUM")
        score = unit.get("priority_score")
        cat = unit.get("priority_category") or unit.get("review_category") or "?"
        gt_count = self._estimate_gt_count(unit)
        ai_count = len(unit.get("proposals", []))

        # Gather class + confidence + IoU info per proposal.
        gt_classes = sorted({p.get("gt_class_name") for p in unit["proposals"] if p.get("gt_class_name")})
        ai_classes = sorted({p.get("class_name") for p in unit["proposals"]})
        confs = [p.get("confidence", 0.0) for p in unit["proposals"]]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        ious = [p.get("gt_iou") for p in unit["proposals"] if p.get("gt_iou") is not None]
        avg_iou = sum(ious) / len(ious) if ious else None

        color = self._LEVEL_COLOR.get(prio, "#555")
        self.info_text.set(
            f"[{self.index + 1}/{len(self.units)}]  Image: {unit['image_filename']}   "
            f"Priority: {prio}  (score={score})   Category: {cat}   Split: {unit.get('split')}"
        )
        self.stat_text.set(
            f"GT count: {gt_count}   AI proposals: {ai_count}   "
            f"GT classes: {gt_classes or '-'}   AI classes: {ai_classes or '-'}   "
            f"Avg confidence: {avg_conf:.2f}   Avg IoU: {avg_iou:.2f}" if avg_iou is not None
            else f"GT count: {gt_count}   AI proposals: {ai_count}   "
            f"GT classes: {gt_classes or '-'}   AI classes: {ai_classes or '-'}   Avg confidence: {avg_conf:.2f}"
        )
        # Detailed proposal table.
        lines = []
        for i, p in enumerate(unit["proposals"]):
            iou_s = f"{p['gt_iou']:.2f}" if p.get("gt_iou") is not None else "-"
            mark = ">>" if p.get("proposal_id") == self._selected_prop_id else "  "
            lines.append(
                f"{mark} {i + 1:>2}. {p.get('class_name', '?'):<10} conf={p.get('confidence', 0):.2f} "
                f"iou={iou_s}  box=({p.get('x1', 0):.0f},{p.get('y1', 0):.0f},{p.get('x2', 0):.0f},{p.get('y2', 0):.0f})"
            )
        self.detail_text.set("\n".join(lines) if lines else "  (no AI proposals for this image)")
        self.status.set(f"Image {self.index + 1} of {len(self.units)}. Use keyboard/buttons to review.")

    def _estimate_gt_count(self, unit) -> int:
        """Ground-truth object count (enriched by enrich_units_with_gt).

        Falls back to a per-proposal match count when enrichment was not run.
        """
        if "_gt_count" in unit:
            return unit["_gt_count"]
        return sum(1 for p in unit.get("proposals", []) if p.get("gt_matched"))

    # ---- image ----------------------------------------------------------------
    def load_image_file(self):
        unit = self._unit()
        path = Path(_REPO_ROOT) / unit["image"]
        if not path.exists():
            self.status.set(f"Image missing: {path}")
            self.canvas.delete("all")
            return
        try:
            img = Image.open(path)
        except Exception as exc:  # noqa: BLE001
            self.status.set(f"Could not load image: {exc}")
            return
        self._img_w, self._img_h = img.size
        self._pil_image = img
        self.render_image()

    def render_image(self):
        self.canvas.delete("all")
        if self._pil_image is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        scale = min(cw / self._img_w, ch / self._img_h)
        scale = min(scale, 1.0)
        self._scale = scale
        disp_w, disp_h = int(self._img_w * scale), int(self._img_h * scale)
        img = self._pil_image
        if scale < 0.999:
            try:
                img = self._pil_image.resize((disp_w, disp_h), Image.LANCZOS)
            except Exception:  # noqa: BLE001
                img = self._pil_image
        self._scaled_photo = ImageTk.PhotoImage(img)
        ox = (self.canvas.winfo_width() - disp_w) // 2 if self.canvas.winfo_width() > disp_w else 0
        oy = (self.canvas.winfo_height() - disp_h) // 2 if self.canvas.winfo_height() > disp_h else 0
        # Persist the image placement so every overlay and mouse interaction uses
        # the SAME offset+scale geometry used to position the displayed image.
        self._offset_x = ox
        self._offset_y = oy
        self.canvas.create_image(ox, oy, anchor="nw", image=self._scaled_photo)
        self._draw_proposals()

    def _draw_proposals(self):
        self.canvas.delete("box")
        ox = getattr(self, "_offset_x", 0)
        oy = getattr(self, "_offset_y", 0)
        for p in self._proposals():
            x1, y1, x2, y2 = img_rect_to_canvas(
                p["x1"], p["y1"], p["x2"], p["y2"], self._scale, ox, oy)
            if p.get("proposal_id") == self._selected_prop_id:
                color = "#00e5ff"
                width = 3
            else:
                color = "#00ff7f"
                width = 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags="box")
            label = f"{p.get('class_name', '?')} {p.get('confidence', 0):.2f}"
            self.canvas.create_text(x1, max(y1 - 4, 2), anchor="sw", text=label,
                                    fill=color, font=("Consolas", 8), tags="box")
# ---- decision recording --------------------------------------------------
    def _record(self, human_decision: str, final_boxes=None, notes: str = ""):
        """Write one per-image human-decision record to the manifest."""
        unit = self._unit()
        proposals = unit.get("proposals", [])
        record = {
            "image": unit["image"],
            "image_filename": unit["image_filename"],
            "split": unit.get("split"),
            "review_category": unit.get("priority_category") or unit.get("review_category"),
            "priority": unit.get("priority_level"),
            "priority_score": unit.get("priority_score"),
            "original_annotation": None,
            "ai_proposals": proposals,
            "human_decision": human_decision,
            "final_class": proposals[0].get("class_name") if (proposals and human_decision in ("accepted", "corrected")) else None,
            "final_boxes": final_boxes if final_boxes is not None else (
                [[p["x1"], p["y1"], p["x2"], p["y2"]] for p in proposals]
                if human_decision in ("accepted", "corrected") else []
            ),
            "reviewer": "human_reviewer",
            "timestamp": datetime.now().isoformat(),
            "notes": notes or f"Decision: {human_decision} via Review GUI.",
        }
        self.decisions_data.setdefault("records", []).append(record)
        save_human_decisions(self.decisions_data)

    def _record_and_advance(self, decision, boxes, notes):
        self._record(decision, final_boxes=boxes, notes=notes)
        self.next_item()

    def accept_all(self):
        unit = self._unit()
        if not unit["proposals"]:
            self.status.set("No proposals to accept; use ADD BOX or SKIP.")
            return
        boxes = [[p["x1"], p["y1"], p["x2"], p["y2"]] for p in unit["proposals"]]
        self._record_and_advance("accepted", boxes,
                                 f"ACCEPT ALL: {len(boxes)} proposals approved as annotation.")

    def reject_all(self):
        self._record_and_advance("rejected", [],
                                 "REJECT ALL: entire AI proposal set rejected.")

    def keep_original(self):
        self._record_and_advance("kept", [],
                                 "KEEP ORIGINAL: ground-truth annotation kept; AI proposals ignored.")

    def mark_uncertain(self):
        self._record_and_advance("uncertain", [],
                                 "MARK UNCERTAIN: image remains unresolved.")

    def skip(self):
        # SKIP writes no decision; the image stays pending.
        self.status.set(f"Skipped {self._unit()['image_filename']} (remains pending).")
        self.next_item()

    def next_item(self):
        self.index += 1
        self.load_current_unit()

    def prev_item(self):
        if self.index > 0:
            self.index -= 1
            self.load_current_unit()
# ---- selection & drag ----------------------------------------------------
    def _init_drag(self):
        self._drag_start = None
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def select_proposal_prompt(self):
        unit = self._unit()
        if not unit["proposals"]:
            self.status.set("No proposals to select.")
            return
        labels = [f"{i + 1}: {p.get('class_name')} ({p.get('confidence', 0):.2f})" for i, p in enumerate(unit["proposals"])]
        sel = simpledialog.askinteger("Select Proposal", "Proposal number:\n" + "\n".join(labels),
                                      parent=self.root, minvalue=1, maxvalue=len(labels))
        if sel:
            self._selected_prop_id = unit["proposals"][sel - 1]["proposal_id"]
            self._update_info(unit)
            self._draw_proposals()

    def _find_proposal_at(self, x, y):
        ox = getattr(self, "_offset_x", 0)
        oy = getattr(self, "_offset_y", 0)
        for p in reversed(self._proposals()):
            cx1, cy1, cx2, cy2 = img_rect_to_canvas(
                p["x1"], p["y1"], p["x2"], p["y2"], self._scale, ox, oy)
            if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                return p
        return None

    def _on_click(self, event):
        p = self._find_proposal_at(event.x, event.y)
        if p is not None:
            self._selected_prop_id = p["proposal_id"]
            self._drag_start = (event.x, event.y, "move")
            self._update_info(self._unit())
            self._draw_proposals()
        else:
            self._selected_prop_id = None
            self._drag_start = None
            self._draw_proposals()

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        _, _, mode = self._drag_start
        p = self._selected_proposal()
        if p is None:
            return
        ox = getattr(self, "_offset_x", 0)
        oy = getattr(self, "_offset_y", 0)
        # Canvas mouse coords -> source-image coords (same offset+scale geometry
        # used to position the displayed image).
        ix, iy = canvas_point_to_img(event.x, event.y, self._scale, ox, oy)
        sx, sy = canvas_point_to_img(self._drag_start[0], self._drag_start[1],
                                     self._scale, ox, oy)
        dx, dy = ix - sx, iy - sy
        if mode == "move":
            p["x1"] += dx
            p["y1"] += dy
            p["x2"] += dx
            p["y2"] += dy
            self._drag_start = (event.x, event.y, "move")
        elif mode == "resize":
            p["x2"] = ix
            p["y2"] = iy
            self._drag_start = (event.x, event.y, "resize")
        self._draw_proposals()

    def _on_release(self, event):
        self._drag_start = None

    def _selected_proposal(self):
        for p in self._proposals():
            if p.get("proposal_id") == self._selected_prop_id:
                return p
        return None

    # ---- editing -------------------------------------------------------------
    def edit_proposal(self):
        p = self._selected_proposal()
        if p is None:
            self.status.set("Select a proposal first (click it or SELECT PROPOSAL).")
            return
        new_xy = simpledialog.askstring(
            "Edit Box", "Enter x1,y1,x2,y2 (pixels):",
            initialvalue=f"{p['x1']:.0f},{p['y1']:.0f},{p['x2']:.0f},{p['y2']:.0f}",
            parent=self.root)
        if new_xy:
            try:
                parts = [float(v) for v in new_xy.replace(" ", "").split(",")]
                if len(parts) != 4:
                    raise ValueError("need 4 values")
                p["x1"], p["y1"], p["x2"], p["y2"] = parts
                self._update_info(self._unit())
                self._draw_proposals()
            except ValueError:
                messagebox.showerror("Bad input", "Enter 4 comma-separated numbers.")

    def delete_proposal(self):
        p = self._selected_proposal()
        if p is None:
            self.status.set("Select a proposal first.")
            return
        self._proposals().remove(p)
        self._selected_prop_id = None
        self._update_info(self._unit())
        self._draw_proposals()

    def add_box(self):
        unit = self._unit()
        w = self._img_w * 0.1
        h = self._img_h * 0.1
        unit["proposals"].append({
            "proposal_id": f"added-{datetime.now().isoformat()}",
            "image": unit["image"],
            "split": unit.get("split"),
            "review_category": unit.get("priority_category") or unit.get("review_category"),
            "class_id": 0,
            "class_name": "new_box",
            "confidence": 0.0,
            "x1": 10, "y1": 10, "x2": 10 + w, "y2": 10 + h,
            "proposal_status": "human_added",
            "created_at": datetime.now().isoformat(),
        })
        self._selected_prop_id = unit["proposals"][-1]["proposal_id"]
        self._update_info(unit)
        self._draw_proposals()

    def change_class(self):
        p = self._selected_proposal()
        if p is None:
            self.status.set("Select a proposal first.")
            return
        cls = simpledialog.askstring("Change Class", "New class name:",
                                     initialvalue=p.get("class_name", ""), parent=self.root)
        if cls:
            p["class_name"] = cls.strip()
            self._update_info(self._unit())
            self._draw_proposals()

    def move_box(self):
        if self._selected_proposal() is None:
            self.status.set("Select a proposal first.")
            return
        self.status.set("Drag the box on the image to move it.")
        self._drag_start = (0, 0, "move")

    def resize_box(self):
        p = self._selected_proposal()
        if p is None:
            self.status.set("Select a proposal first.")
            return
        self.status.set("Drag to resize the selected box (bottom-right corner).")
        ox = getattr(self, "_offset_x", 0)
        oy = getattr(self, "_offset_y", 0)
        cx1, cy1, cx2, cy2 = img_rect_to_canvas(
            p["x1"], p["y1"], p["x2"], p["y2"], self._scale, ox, oy)
        # Start the drag at the box's bottom-right corner in CANVAS coordinates
        # (matching how _on_drag converts canvas -> image coords).
        self._drag_start = (cx2, cy2, "resize")
# --- GT enrichment (read-only) -------------------------------------------------

def enrich_units_with_gt(units: list, data_root: Path, class_names: list) -> None:
    """Tag each proposal with its matched ground-truth info (gt_iou, gt_class_name).

    Read-only: never writes to data/detection. Enables the GUI to show GT count,
    GT classes and per-proposal IoU against ground truth.
    """
    from scripts.audit_detection_dataset import _find_dataset_root, _read_boxes  # noqa: F401

    data_root = _find_dataset_root(data_root) or data_root
    for unit in units:
        fname = unit["image_filename"]
        split = unit.get("split", "train")
        label_path = data_root / split / "labels" / (Path(fname).stem + ".txt")
        gt_boxes = read_gt_boxes_xy(label_path, class_names)
        gt_boxes_px = []
        import cv2  # noqa: PLC0415
        img = cv2.imread(str(Path(_REPO_ROOT) / unit["image"]))
        if img is not None:
            ih, iw = img.shape[:2]
            for g in gt_boxes:
                x1 = (g["cx"] - g["w"] / 2.0) * iw
                y1 = (g["cy"] - g["h"] / 2.0) * ih
                x2 = (g["cx"] + g["w"] / 2.0) * iw
                y2 = (g["cy"] + g["h"] / 2.0) * ih
                gt_boxes_px.append((x1, y1, x2, y2, g["class_name"]))

        unit["_gt_count"] = len(gt_boxes_px)
        # Tag each proposal with best GT match IoU + class.
        for p in unit["proposals"]:
            best_iou = 0.0
            best_cls = None
            for g in gt_boxes_px:
                ov = iou((p["x1"], p["y1"], p["x2"], p["y2"]), (g[0], g[1], g[2], g[3]))
                if ov > best_iou:
                    best_iou = ov
                    best_cls = g[4]
            p["gt_iou"] = round(best_iou, 4) if best_iou >= 0.3 else None
            p["gt_class_name"] = best_cls if best_iou >= 0.3 else None
            p["gt_matched"] = best_iou >= 0.5


def read_gt_boxes_xy(label_path: Path, class_names: list) -> list:
    """Read normalized GT boxes (dicts with cx/cy/w/h/class_name)."""
    if not label_path.exists():
        return []
    boxes, _ = _read_boxes(label_path, len(class_names))
    return [{
        "class_id": b[0],
        "class_name": class_names[b[0]] if b[0] < len(class_names) else str(b[0]),
        "cx": b[1], "cy": b[2], "w": b[3], "h": b[4],
    } for b in boxes if b[3] > 0 and b[4] > 0]


def _resolve_class_names(data_root: Path) -> list:
    from scripts.audit_detection_dataset import load_data_config  # noqa: F401
    root = _find_dataset_root(data_root)
    if root is None:
        return []
    _raw, _nc, names = load_data_config(root)
    return names


def _find_data_root(data_root) -> Path:
    from scripts.audit_detection_dataset import _find_dataset_root as fdr
    return fdr(data_root) or Path(data_root)


def main() -> int:
    """Launch the interactive annotation-proposal review GUI.

    Wires together the existing proposal loader, prioritiser/grouping and the
    AnnotationReviewGUI. No proposals, labels, datasets or model weights are
    created or modified -- this is a human review tool only.
    """
    # --- AI proposals (read-only; proposals.json is never regenerated) -----------
    try:
        proposals_by_image = load_proposals_by_image()
    except (OSError, ValueError) as exc:
        print(
            f"ERROR: could not read proposals from {PROPOSAL_FILE}: {exc}. "
            "Run scripts/generate_annotation_proposals.py first.",
            file=sys.stderr,
        )
        return 1

    if not proposals_by_image:
        print(
            f"ERROR: no proposals found at {PROPOSAL_FILE}. "
            "Run scripts/generate_annotation_proposals.py first. "
            "No dataset (V2 or V3) was read or modified.",
            file=sys.stderr,
        )
        return 1

    # group_proposals_by_image expects a flat proposal list; it re-groups the
    # proposals by image and applies the review-priority ordering.
    proposals = [p for group in proposals_by_image.values() for p in group]

    priority_map = load_priority_map()
    units = group_proposals_by_image(proposals, priority_map)
    decisions_data = load_human_decisions()

    # --- tkinter availability ----------------------------------------------------
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(
            f"ERROR: could not initialize a Tk display ({exc}). "
            "This tool requires a graphical environment.",
            file=sys.stderr,
        )
        return 1

    AnnotationReviewGUI(root, units, decisions_data)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())