#!/usr/bin/env python3
"""Manual annotation tool for SmartFreshAI Dataset V3 (Windows GUI).

Human-in-the-loop bounding-box annotation. Draws boxes with the mouse,
saves YOLO labels, and records decisions. It is read-only with respect to
the frozen V2 dataset (data/detection/) and best.pt.

Requires a GUI-capable OpenCV build (WIN32UI/GTK/Qt). It does NOT fall back
to headless mode, because human annotation requires an interactive window.

Usage:
    python scripts/manual_annotate_detection.py
    python scripts/manual_annotate_detection.py --out reports/audit_review/manual_annotations
"""
from __future__ import annotations

import argparse
import json
import logging
def validate_box(x1, y1, x2, y2, img_w, img_h):
    """Validate a pixel-space box. Returns (ok, error_msg).
    x1<x2, y1<y2, positive dims, min area, inside image."""
    if x1 >= x2 or y1 >= y2:
        return False, "x2 must be > x1 and y2 > y1"
    w = x2 - x1
    h = y2 - y1
    if w < 1 or h < 1:
        return False, "positive width/height required"
    # inside image bounds (allow a small clamp tolerance, but require >=0)
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False, "box must be inside image bounds"
    if w * h < MIN_AREA_PX:
        return False, f"box too small (area {w*h} < {MIN_AREA_PX}px)"
    return True, None


def validate_yolo(cls_id, cx, cy, w, h):
    """Validate a normalized YOLO row. Returns (ok, error_msg)."""
    if cls_id not in CID.values():
        return False, f"invalid class id {cls_id}"
    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        return False, f"coords out of [0,1]: cx={cx:.4f}, cy={cy:.4f}"
    if w <= 0 or h <= 0:
        return False, f"width/height must be positive: w={w:.4f}, h={h:.4f}"
    if cx + w / 2 > 1 or cy + h / 2 > 1 or cx - w / 2 < 0 or cy - h / 2 < 0:
        return False, "box extends beyond image"
    if w >= MAX_AREA_RATIO or h >= MAX_AREA_RATIO:
        return False, "box is implausibly large (covers whole image)"
    return True, None


def pixels_to_yolo(x1, y1, x2, y2, img_w, img_h, cls_id):
    """Convert pixel box to normalized YOLO [cls_id, cx, cy, w, h].
    Raises ValueError if the box is invalid."""
    ok, err = validate_box(x1, y1, x2, y2, img_w, img_h)
    if not ok:
        raise ValueError(err)
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    vok, verr = validate_yolo(cls_id, cx, cy, w, h)
    if not vok:
        raise ValueError(verr)
    return [cls_id, cx, cy, w, h]


def load_or_init_queue():
    """Load the annotation queue from decisions.json, or build a fresh pending
    queue if the file does not exist. Returns list of decision records."""
    if DECISIONS_FILE.exists():
        try:
            data = json.loads(DECISIONS_FILE.read_text(encoding="utf-8"))
            if data.get("queue"):
                return data["queue"]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read decisions.json (%s); rebuilding "
                           "queue but will not overwrite existing decisions.", e)
    return [{
        "image": imgrel, "split": split, "original_label": _orig_label(imgrel),
        "status": "pending", "class_names": [cls],
        "boxes": [], "reviewer": "human",
        "notes": "Empty-label image requiring manual annotation.",
    } for imgrel, split, cls in EMPTY_IMGS]


def _orig_label(imgrel):
    """Return the original V2 label path for an image path."""
    p = Path(imgrel)
    return str(p).replace("images", "labels").replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")


def atomic_write_json(path, payload):
    """Write JSON atomically: temp file -> os.replace. Never corrupts target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.write("\n")
            f.flush()
            import os
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


def save_decision(queue):
    """Persist the whole queue (header + records) to decisions.json atomically."""
    statuses = [r["status"] for r in queue]
    payload = {
        "schema_version": 1,
        "source": "human-annotation",
        "total_images": len(queue),
        "annotated": statuses.count("annotated"),
        "skipped": statuses.count("skipped"),
        "manual_review": statuses.count("manual_review"),
        "pending": statuses.count("pending"),
        "queue": queue,
    }
    atomic_write_json(DECISIONS_FILE, payload)
    return payload
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("annotate")

REPO = str(Path(__file__).resolve().parents[1])
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Class mapping identical to data/detection/data.yaml (0..9). Do NOT change.
CLS = ["Apple", "Grape", "Kiwi", "Mango", "Orange",
               "Strawberry", "banana", "cherry", "chickoo", "guava"]
CID = {n: i for i, n in enumerate(CLS)}

# The 8 V2 empty-label images that require manual annotation.
# (image_path_relative_to_repo, split, expected_class)
EMPTY_IMGS = [
    ("data/detection/train/images/apple_18_jpg.rf.01613e8cb77944008b763ecaa2632cdb.jpg", "train", "Apple"),
    ("data/detection/train/images/apple_20_jpg.rf.0100411a2fc97f1ba184cc88650f93ad.jpg", "train", "Apple"),
    ("data/detection/train/images/apple_49_jpg.rf.6e0b48e6a57c8aed5310cde6e6c47d43.jpg", "train", "Apple"),
    ("data/detection/train/images/Grape-23-_jpeg.rf.ddfca0378f72719738dc55d1b956bcc0.jpg", "train", "Grape"),
    ("data/detection/train/images/Grape-34-_jpeg.rf.b74e4c3485b4346336805639dc6b4ddc.jpg", "train", "Grape"),
    ("data/detection/valid/images/Grape-41-_jpeg.rf.beb4a82287752e3700f59aea4907b232.jpg", "valid", "Grape"),
    ("data/detection/valid/images/Grape-50-_jpeg.rf.10c06cde45cb968297fc38c27bfa5b6f.jpg", "valid", "Grape"),
    ("data/detection/valid/images/Grape-64-_jpeg.rf.b1950af05d9c17b60b6062f6eee273a3.jpg", "valid", "Grape"),
]

OUTDIR = Path("reports/audit_review/manual_annotations")
OUTDIR_IMG = OUTDIR / "images"
OUTDIR_LBL = OUTDIR / "labels"
DECISIONS_FILE = OUTDIR / "decisions.json"

MIN_AREA_PX = 10  # minimum bounding-box pixel area
MIN_AREA_RATIO = 0.0001  # minimum normalized width/height
MAX_AREA_RATIO = 0.95    # must not cover almost the whole image

WINDOW_TITLE = "Annotator"
WINDOW_INIT_W = 900
WINDOW_INIT_H = 700


# --------------------------------------------------------------------------- #
# Pure core logic (GUI-independent, unit-testable)
# --------------------------------------------------------------------------- #

def check_gui_capability() -> None:
    """Fail fast with a clear message if OpenCV HighGUI is unavailable."""
    backend = ""
    try:
        info = cv2.getBuildInformation()
        for line in info.split("\n"):
            if "GUI:" in line:
                backend = line.split(":")[-1].strip()
                break
    except Exception:
        backend = "unknown"
    missing = [f for f in ("namedWindow", "imshow", "waitKey",
                           "setMouseCallback") if not hasattr(cv2, f)]
    if missing or "HEADLESS" in backend.upper():
        raise RuntimeError(
            "OpenCV HighGUI is unavailable (backend=%r, missing=%s). "
            "A GUI-capable OpenCV build is required for manual annotation. "
            "Fix: install a GUI build, e.g.  pip install opencv-python "
            "and remove opencv-python-headless." % (backend, missing))

# --------------------------------------------------------------------------- #
# GUI layer (requires the window-capable OpenCV build)
# --------------------------------------------------------------------------- #

class Annotator:
    """Interactive bounding-box annotator window."""

    def __init__(self, img_bgr: np.ndarray, expected_cls: str):
        check_gui_capability()
        self.orig = img_bgr.copy()
        self.disp = img_bgr.copy()
        self.h, self.w = self.orig.shape[:2]
        self.win = WINDOW_TITLE
        self.drawing = False
        self.sx = self.sy = -1
        self.boxes: list = []  # list of x1,y1,x2,y2 in pixels
        self.expected_cls = expected_cls
        self.clsname = expected_cls  # current selection = expected default
        self.clsid = CID.get(self.clsname, 0)
        # explicit, resizable window
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, WINDOW_INIT_W, WINDOW_INIT_H)
        cv2.setMouseCallback(self.win, self._on_mouse)

    # ---- mouse ----
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.sx, self.sy = int(x), int(y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.disp = self.orig.copy()
            cv2.rectangle(self.disp, (self.sx, self.sy), (int(x), int(y)),
                          (0, 255, 0), 2)
            self._overlay()
            cv2.imshow(self.win, self.disp)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self._add_box(self.sx, self.sy, int(x), int(y))
            self.redraw()

    def _add_box(self, x1, y1, x2, y2):
        ok, err = validate_box(x1, y1, x2, y2, self.w, self.h)
        if not ok:
            logger.warning("Rejected box: %s", err)
            self._flash_message(err)
            return
        # store in normalized (x1<=x2, y1<=y2) order
        self.boxes.append((min(x1, x2), min(y1, y2),
                           max(x1, x2), max(y1, y2)))

    # ---- rendering ----
    def _overlay(self):
        """Draw boxes + instruction panel on self.disp (no imshow)."""
        d = self.disp
        h, w = d.shape[:2]
        for (a, b, c, e) in self.boxes:
            color = (0, 0, 255) if self.clsid == 1 else (255, 0, 0)
            cv2.rectangle(d, (a, b), (c, e), color, 2)
            lbl = f"{self.clsname} ({self.clsid})"
            cv2.putText(d, lbl, (a + 2, b - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # top-left panel
        cv2.rectangle(d, (6, 6), (int(w * 0.5), 140), (0, 0, 0), -1)
        cv2.putText(d, f"Expected class: {self.expected_cls}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(d, f"Current class: {self.clsname} ({self.clsid})",
                    (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(d, "Keys: 0-9=class ENTER=confirm R=reset U=undo",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(d, "S=skip Q/ESC=quit   Mouse: Left-drag = box",
                    (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(d, f"Boxes: {len(self.boxes)}",
                    (12, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # bottom progress
        cv2.rectangle(d, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.putText(d, self._progress_line(), (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_top_class(self):
        """Draw the selected class prominently near the top."""
        pass  # handled by _overlay()

    def _flash_message(self, msg):
        """Show a transient message on the window."""
        try:
            d = self.orig.copy()
            cv2.putText(d, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.imshow(self.win, d)
            cv2.waitKey(300)
        except cv2.error:
            pass
        self.redraw()

    def redraw(self):
        self.disp = self.orig.copy()
        self._overlay()
        cv2.imshow(self.win, self.disp)

    def _progress_line(self):
        if not hasattr(self, "_progress"):
            return f"Image ?/?   Boxes: {len(self.boxes)}"
        idxp, tot, ann2, skip2, pen2 = self._progress
        return (f"Image {idxp + 1}/{tot}   Annotated:{ann2} Skipped:{skip2} "
                f"Pending:{pen2}   Boxes:{len(self.boxes)}")

    # ---- key handling ----
    def handle_key(self, k):
        """Handle a key; returns an action string or None."""
        if k in (ord('q'), 27):
            return 'quit'
        if k == 13:  # ENTER
            return 'confirm'
        if k == ord('r'):
            return 'reset'
        if k == ord('u'):
            return 'undo'
        if k == ord('s'):
            return 'skip'
        idx = k - ord('0')
        if 0 <= idx < len(CLS):
            self.clsname = CLS[idx]
            self.clsid = idx
            return None
        return None

    def finish(self):
        """Return validated YOLO rows for all drawn boxes, or [] if none valid."""
        ann = []
        for (a, b, c, e) in self.boxes:
            try:
                ann.append(pixels_to_yolo(a, b, c, e, self.w, self.h, self.clsid))
            except ValueError as ex:
                logger.warning("finish: skipping invalid box: %s", ex)
        return ann

# --------------------------------------------------------------------------- #
# Event loop + entry point
# --------------------------------------------------------------------------- #

def run_image(annotator):
    """Blocking event loop for one image window. Returns an action:
    'confirm', 'skip', or 'quit'."""
    confirm_attempts = 0
    while True:
        annotator.redraw()
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue  # no key; keep pumping GUI events
        action = annotator.handle_key(key)
        if action == 'quit':
            return 'quit'
        if action == 'reset':
            annotator.boxes = []
            annotator.redraw()
        elif action == 'undo':
            if annotator.boxes:
                annotator.boxes.pop()
            annotator.redraw()
        elif action == 'skip':
            return 'skip'
        elif action == 'confirm':
            ann = annotator.finish()
            if not ann:
                confirm_attempts += 1
                annotator._flash_message(
                    "No boxes drawn. Press ENTER again to confirm empty, "
                    "or draw a box.")
                if confirm_attempts < 2:
                    continue
            return 'confirm'
        else:
            annotator.redraw()


def main() -> int:
    global OUTDIR, OUTDIR_IMG, OUTDIR_LBL, DECISIONS_FILE

    parser = argparse.ArgumentParser(
        description="Manual annotation interface for SmartFreshAI Dataset V3")
    parser.add_argument("--out", type=Path, default=OUTDIR,
                        help="output dir (default: reports/audit_review/manual_annotations)")
    args = parser.parse_args()

    OUTDIR = args.out
    OUTDIR_IMG = OUTDIR / "images"
    OUTDIR_LBL = OUTDIR / "labels"
    DECISIONS_FILE = OUTDIR / "decisions.json"
    OUTDIR_IMG.mkdir(parents=True, exist_ok=True)
    OUTDIR_LBL.mkdir(parents=True, exist_ok=True)

    check_gui_capability()
    queue = load_or_init_queue()
    if not queue:
        logger.info("Queue is empty; nothing to annotate.")
        return 0

    pending = [r for r in queue if r["status"] == "pending"]
    logger.info("Queue: %d total, %d pending (annotated/skipped skipped).",
                len(queue), len(pending))

    try:
        for idx, (imgrel, split, cls) in enumerate(EMPTY_IMGS):
            rec = next((r for r in queue if r["image"] == imgrel), None)
            if rec is None:
                continue
            if rec["status"] in ("annotated", "skipped", "manual_review"):
                logger.info("[%d/%d] skip %s (status=%s)",
                            idx + 1, len(EMPTY_IMGS), imgrel, rec["status"])
                continue  # resume: already handled
            ip = Path(imgrel)
            if not ip.exists():
                logger.warning("Image not found: %s", ip)
                rec["status"] = "skipped"
                rec["notes"] = "Images file missing on disk; skipped."
                save_decision(queue)
                continue
            img = cv2.imread(str(ip))
            if img is None:
                logger.error("Could not read image: %s", ip)
                rec["status"] = "manual_review"
                save_decision(queue)
                continue

            logger.info("[%d/%d] Annotate: %s (expected %s)",
                        idx + 1, len(EMPTY_IMGS), ip.name, cls)
            ann_ = Annotator(img, cls)
            ann_._progress = (idx, len(EMPTY_IMGS),
                              sum(1 for r in queue if r["status"] == "annotated"),
                              sum(1 for r in queue if r["status"] == "skipped"),
                              sum(1 for r in queue if r["status"] == "pending"))
            ann_.redraw()

            try:
                action = run_image(ann_)
            except cv2.error as e:
                logger.error("OpenCV GUI error: %s", e)
                raise

            if action == 'quit':
                logger.info("Quit requested. Saving progress without "
                            "marking current image.")
                save_decision(queue)
                return 0
            if action == 'skip':
                rec["status"] = "skipped"
                rec["notes"] = "Skipped by human; no fake label created."
                logger.info("  skipped: %s", ip.name)
            else:  # confirm
                rows = ann_.finish()
                if rows:
                    lbl_name = Path(imgrel).with_suffix(".txt").name
                    out_lbl = OUTDIR_LBL / lbl_name
                    content = "".join(
                        f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n"
                        for r in rows)
                    out_lbl.write_text(content, encoding="utf-8")
                    out_img = OUTDIR_IMG / Path(imgrel).name
                    cv2.imwrite(str(out_img), img.copy())
                    rec["boxes"] = [
                        {"class_id": r[0],
                         "class_name": CLS[r[0]] if r[0] < len(CLS) else "?",
                         "yolo": r} for r in rows]
                    rec["status"] = "annotated"
                    logger.info("  annotated: %s (%d box/es)", ip.name, len(rows))
                else:
                    rec["status"] = "manual_review"
                    rec["notes"] = "Confirmed with no valid box; manual review."
                    logger.info("  manual_review (no box): %s", ip.name)
                save_decision(queue)
    finally:
        cv2.destroyAllWindows()

    save_decision(queue)
    summary = save_decision(queue)
    logger.info("Done. Annotated=%d Skipped=%d Pending=%d ManualReview=%d",
                summary["annotated"], summary["skipped"], summary["pending"],
                summary.get("manual_review", 0))
    logger.info("Output: %s", OUTDIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())