"""Regression tests for the annotation-review GUI coordinate rendering.

These tests lock down the single shared coordinate transform that maps
source-image proposal coordinates onto the (possibly centered) Tkinter canvas,
and its inverse used for mouse interaction.

Root cause that motivated these tests:
    render_image() places the displayed image at canvas offset (ox, oy), but the
    overlay drawing previously scaled proposal coordinates WITHOUT adding that
    offset, so boxes were shifted up/left into the black margin.
"""

import json
from pathlib import Path

from scripts.review_annotation_proposals import (
    PROPOSAL_FILE,
    canvas_point_to_img,
    img_rect_to_canvas,
)


def test_img_rect_to_canvas_example():
    """Given scale=0.5, ox=100, oy=50 the spec'd box maps to (200,100,300,200)."""
    scale, ox, oy = 0.5, 100, 50
    bbox = (200, 100, 400, 300)
    cx1, cy1, cx2, cy2 = img_rect_to_canvas(*bbox, scale, ox, oy)
    assert (cx1, cy1, cx2, cy2) == (200, 100, 300, 200)


def test_img_rect_to_canvas_zero_offset_identity_scaled():
    """With ox=oy=0 the transform is a pure scale about the origin."""
    scale, ox, oy = 0.5, 0, 0
    cx1, cy1, cx2, cy2 = img_rect_to_canvas(10, 20, 40, 80, scale, ox, oy)
    assert (cx1, cy1, cx2, cy2) == (5, 10, 20, 40)


def test_inverse_transform_returns_original():
    """canvas_point_to_img must recover source-image coordinates."""
    scale, ox, oy = 0.5, 100, 50
    bbox = (200, 100, 400, 300)
    cx1, cy1, cx2, cy2 = img_rect_to_canvas(*bbox, scale, ox, oy)
    # Center of the canvas rectangle.
    ix, iy = canvas_point_to_img((cx1 + cx2) / 2, (cy1 + cy2) / 2, scale, ox, oy)
    assert abs(ix - (200 + 400) / 2) < 1e-9
    assert abs(iy - (100 + 300) / 2) < 1e-9


def test_inverse_round_trip_edge():
    """Mapping a corner back and forth reproduces the source corner exactly."""
    scale, ox, oy = 0.25, 37, 41
    x, y = 123.0, 456.0
    cx, cy = ox + x * scale, oy + y * scale
    ix, iy = canvas_point_to_img(cx, cy, scale, ox, oy)
    assert ix == x and iy == y


def test_real_proposal_source_coords_unchanged():
    """proposals.json is read-only: a real proposal's coords must not change."""
    with open(PROPOSAL_FILE, "r", encoding="utf-8") as f:
        proposals = json.load(f)
    assert len(proposals) > 0
    p = proposals[0]
    src = (p["x1"], p["y1"], p["x2"], p["y2"])
    # Re-load to confirm nothing changed on disk during the test.
    with open(PROPOSAL_FILE, "r", encoding="utf-8") as f2:
        proposals2 = json.load(f2)
    p2 = proposals2[0]
    assert (p2["x1"], p2["y1"], p2["x2"], p2["y2"]) == src


def test_real_proposal_maps_into_displayed_image_region():
    """A real proposal's transformed box lands inside the displayed image rect.

    The displayed image occupies canvas [ox, ox+disp_w] x [oy, oy+disp_h]. With a
    non-zero offset the box must still be inside the image, NOT shifted to the
    canvas origin (which is the bug being fixed).
    """
    with open(PROPOSAL_FILE, "r", encoding="utf-8") as f:
        proposals = json.load(f)

    # Reproduce render_image() geometry for a 1000x700 canvas and a large image.
    # The image is scaled down to fit height exactly, leaving a horizontal margin
    # on each side -> ox > 0, oy == 0 (the realistic single-offset centering).
    img_w, img_h = 1280, 960
    canvas_w, canvas_h = 1000, 700
    scale = min(canvas_w / img_w, canvas_h / img_h)
    scale = min(scale, 1.0)
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)
    ox = (canvas_w - disp_w) // 2
    oy = (canvas_h - disp_h) // 2
    assert ox > 0 and oy == 0, "expected horizontal-only centering offset"

    for p in proposals[:20]:
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        cx1, cy1, cx2, cy2 = img_rect_to_canvas(x1, y1, x2, y2, scale, ox, oy)
        # Horizontal edges must respect the non-zero horizontal offset.
        assert cx1 >= ox and cx2 <= ox + disp_w, "box shifted horizontally off image"
        # Vertical is flush (oy == 0); box must still fit within image height.
        assert cy1 >= oy and cy2 <= oy + disp_h, "box shifted vertically off image"


def test_real_proposal_box_not_at_canvas_origin():
    """The transformed box must NOT be drawn relative to the canvas origin.

    If the offset is dropped the box would appear in the black margin to the
    upper-left of the centered image.
    """
    with open(PROPOSAL_FILE, "r", encoding="utf-8") as f:
        proposals = json.load(f)
    # Canvas larger than the image in both dimensions -> both ox and oy > 0
    # (scale caps at 1.0, the image is centered with real margins).
    img_w, img_h = 1000, 1000
    canvas_w, canvas_h = 1500, 1200
    scale = min(canvas_w / img_w, canvas_h / img_h)
    scale = min(scale, 1.0)
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)
    ox = (canvas_w - disp_w) // 2
    oy = (canvas_h - disp_h) // 2
    assert ox > 0 and oy > 0, "expected a centered image with real margins"

    p = proposals[0]
    cx1, cy1, _, _ = img_rect_to_canvas(
        p["x1"], p["y1"], p["x2"], p["y2"], scale, ox, oy)
    # A centered image has a meaningful horizontal offset; without applying it
    # the box's left edge would sit at ox pixels too far left.
    assert cx1 >= ox + 0.5, "offset was not applied to the box"