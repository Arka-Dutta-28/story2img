"""
utils/sanity_check_compositor.py
--------------------------------
Geometry checks for compositor/page.py, using synthetic panels so it runs on a
clean clone with no generated images, no GPU and no network:

    python utils/sanity_check_compositor.py [out_dir]

Writes real PNGs so the result can also be looked at, which is the lesson from
PLAN.md Phase 0: summary numbers looked correct while the boxes were visibly
wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from compositor.page import (  # noqa: E402
    BORDER,
    GUTTER,
    MARGIN,
    PAGE_H,
    PAGE_W,
    Panel,
    compose_page,
)

FABLE = [
    ("left",   [{"speaker": "Mouse", "line": "Please let me go, I meant no harm."}]),
    ("right",  [{"speaker": "Lion", "line": "You are too small to be worth my anger."}]),
    ("center", []),
    ("left",   [{"speaker": "Mouse", "line": "I will gnaw the rope net and free you."},
                {"speaker": "narrator", "line": "And so the Lion was freed."}]),
]


def fake_panel(i: int) -> Image.Image:
    """A distinguishable placeholder, so panel identity is visible in the output."""
    img = Image.new("RGB", (600, 600), (235 - i * 18, 228, 214 + i * 6))
    d = ImageDraw.Draw(img)
    d.ellipse((180, 240, 420, 480), fill=(120 + i * 25, 110, 100))
    # no text label: Magi detects it as a balloon and it corrupts the
    # very text-region count this pipeline is meant to measure.
    return img


def overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/compositor_check")
    panels = [Panel(fake_panel(i), d, pos) for i, (pos, d) in enumerate(FABLE)]

    r = compose_page(panels, out / "page_ltr.png", reading="ltr")

    # one box per panel, all inside the page, respecting the margin
    assert len(r.panel_boxes) == len(panels)
    for x0, y0, x1, y1 in r.panel_boxes:
        assert 0 <= x0 < x1 <= PAGE_W and 0 <= y0 < y1 <= PAGE_H
        assert x0 >= MARGIN and y0 >= MARGIN
        assert x1 <= PAGE_W - MARGIN and y1 <= PAGE_H - MARGIN

    # panels must not overlap, or the decoder cannot separate them
    for i in range(len(r.panel_boxes)):
        for j in range(i + 1, len(r.panel_boxes)):
            assert not overlaps(r.panel_boxes[i], r.panel_boxes[j]), (i, j)

    # a real gutter must exist between neighbours in a row, else panels merge
    a, b = r.panel_boxes[0], r.panel_boxes[1]
    assert b[0] - a[2] >= GUTTER - 1, f"gutter too small: {b[0] - a[2]}"

    # every balloon sits inside the panel it belongs to
    for bbox, slot in zip(r.balloon_boxes, r.balloon_owner):
        px0, py0, px1, py1 = r.panel_boxes[slot]
        assert px0 <= bbox[0] and bbox[2] <= px1, (bbox, r.panel_boxes[slot])
        assert py0 <= bbox[1] and bbox[3] <= py1, (bbox, r.panel_boxes[slot])
        # and in the upper band, which is how faces stay unoccluded
        assert bbox[1] < py0 + (py1 - py0) * 0.62 + 1

    # every balloon must actually contain ink. Counting boxes is not enough:
    # an early return once skipped the text loop, so balloons rendered as blank
    # white boxes while this check still passed and Magi lost a whole line.
    page = Image.open(out / "page_ltr.png").convert("L")
    for bbox in r.balloon_boxes:
        crop = page.crop(bbox)
        dark = sum(crop.histogram()[:100])   # histogram, getdata is deprecated
        assert dark > 40, f"balloon looks empty, only {dark} dark pixels: {bbox}"

    # scene 3 has no dialogue, so no balloon may be attributed to its slot
    empty_slot = r.panel_order.index(2)
    assert empty_slot not in r.balloon_owner, "balloon drawn on a silent panel"

    # four spoken lines across four panels: 1 + 1 + 0 + 2
    expected = sum(len(d) for _, d in FABLE)
    assert len(r.balloon_boxes) == expected == 4, (len(r.balloon_boxes), expected)

    # reading order: ltr is identity, rtl reverses within each row
    r2 = compose_page(panels, out / "page_rtl.png", reading="rtl")
    assert r.panel_order == [0, 1, 2, 3], r.panel_order
    assert r2.panel_order == [1, 0, 3, 2], r2.panel_order

    # odd panel counts must not crash or leave a box outside the page
    r3 = compose_page(panels[:3], out / "page_odd.png")
    assert len(r3.panel_boxes) == 3
    assert r3.panel_boxes[-1][3] <= PAGE_H - MARGIN

    assert Path(out / "page_ltr.png").exists()
    print(f"compositor OK: {len(r.panel_boxes)} panels, {len(r.balloon_boxes)} balloons, "
          f"16 assertions passed")
    print(f"  wrote {out}/page_ltr.png, page_rtl.png, page_odd.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
