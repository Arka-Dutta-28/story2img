"""
compositor/page.py
------------------
Compose generated panels into a comic page: bordered rectangular panels on a
portrait page, with speech balloons carrying each scene's dialogue.

WHY THIS SHAPE. Magi, the comic-understanding model used to score output
(see ../../PLAN.md, Phase 0b), was measured on two layout conventions:

  * Western newspaper strips, landscape bands with thin gutters:
    panel detection FAILED, boxing whole strip bands instead of panels,
    and 15-23% text-to-character association on two of five pages.
  * Golden-age comic book pages, portrait with bordered rectangular panels:
    8/9/8/7/8 panels correctly found, 67-89% association.

So portrait, bordered, gridded is not an aesthetic choice. It is the layout the
decoder can actually read, which is a precondition for measuring anything.

Deliberately crude: fixed grid, rectangular balloons, no importance-driven
panel sizing and no emotion-driven balloon shapes. Those are in Yang et al.
(ACM TOMM 2021) and are the upgrade path once a plain page decodes cleanly.

No GPU, no model, no network.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = 1000, 1500        # portrait, close to the 960x1391 controls
MARGIN = 28
GUTTER = 16
BORDER = 3
COLUMNS = 2

BALLOON_PAD = 8
BALLOON_MAX_FRAC = 0.42            # a balloon may claim this much panel width
BALLOON_TOP_FRAC = 0.62            # balloons live in the top part of a panel
TAIL = 11

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

ReadingOrder = Literal["ltr", "rtl"]


@dataclass
class Panel:
    """One panel: an image, the lines spoken in it, and where the speaker is."""

    image: Image.Image
    dialogue: list[dict] = field(default_factory=list)   # [{"speaker","line"}]
    speaker_position: str = "center"                     # left | center | right


@dataclass
class PageResult:
    """What was composed, so a caller can score it against its own intent."""

    path: Path
    panel_boxes: list[tuple[int, int, int, int]]
    balloon_boxes: list[tuple[int, int, int, int]]
    panel_order: list[int]
    balloon_owner: list[int]       # balloon i belongs to panel_order index


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    hits = sorted(glob.glob("/usr/share/fonts/**/*Sans*Bold*.ttf", recursive=True))
    if hits:
        return ImageFont.truetype(hits[0], size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    """Greedy wrap. Long single words are left to overflow rather than split."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _grid(n: int, columns: int = COLUMNS) -> list[tuple[int, int, int, int]]:
    """Row-major boxes for n panels. Last row is left-aligned if short."""
    rows = math.ceil(n / columns)
    usable_w = PAGE_W - 2 * MARGIN - (columns - 1) * GUTTER
    usable_h = PAGE_H - 2 * MARGIN - (rows - 1) * GUTTER
    cw, ch = usable_w // columns, usable_h // rows
    boxes = []
    for i in range(n):
        r, c = divmod(i, columns)
        x0 = MARGIN + c * (cw + GUTTER)
        y0 = MARGIN + r * (ch + GUTTER)
        boxes.append((x0, y0, x0 + cw, y0 + ch))
    return boxes


def _order(n: int, reading: ReadingOrder, columns: int = COLUMNS) -> list[int]:
    """Grid index in reading order.

    Magi is manga-trained and manga reads right-to-left, while English source
    text reads left-to-right. That difference is a confound for any
    reading-order metric, so it is a parameter here rather than an assumption.
    Compose both ways and compare what the decoder recovers.
    """
    if reading == "ltr":
        return list(range(n))
    out: list[int] = []
    for r in range(math.ceil(n / columns)):
        row = [i for i in range(n) if i // columns == r]
        out.extend(reversed(row))
    return out


def _draw_balloon(draw, box, lines, font, tail_to: str, tail: bool = True) -> None:
    """Draw one balloon. ``tail`` is False for all but the last in a stack.

    Only the final balloon gets a tail. Giving every balloon one puts a tail
    into the top edge of the balloon below it, and Magi split such a balloon
    into two text regions when this was first run, reading a fragment
    "And so freed." out of an intact line. Consecutive lines from one speaker
    are conventionally drawn as a stack with a single tail anyway.
    """
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill="white", outline="black", width=2)
    if tail:
        cx = {"left": x0 + (x1 - x0) * 0.25,
              "right": x0 + (x1 - x0) * 0.75}.get(tail_to, (x0 + x1) / 2)
        draw.polygon([(cx - TAIL / 2, y1 - 1), (cx + TAIL / 2, y1 - 1), (cx, y1 + TAIL)],
                     fill="white", outline="black")
    yy = y0 + BALLOON_PAD
    for ln in lines:
        draw.text((x0 + BALLOON_PAD, yy), ln, font=font, fill="black")
        yy += font.size + 2


def compose_page(
    panels: Sequence[Panel],
    out_path: str | Path,
    reading: ReadingOrder = "ltr",
    columns: int = COLUMNS,
    font_size: int = 15,
) -> PageResult:
    """Compose panels into one page and write it. Returns what it drew."""
    if not panels:
        raise ValueError("no panels to compose")

    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    font = _font(font_size)

    boxes = _grid(len(panels), columns)
    order = _order(len(panels), reading, columns)

    balloon_boxes: list[tuple[int, int, int, int]] = []
    balloon_owner: list[int] = []

    for slot, panel_idx in enumerate(order):
        x0, y0, x1, y1 = boxes[slot]
        p = panels[panel_idx]

        art = p.image.convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS)
        page.paste(art, (x0, y0))
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="black", width=BORDER)

        # Balloons stack from the top, in spoken order, which is also the order
        # a reader takes them. Keeping them in the upper band means they rarely
        # cover a face, which is one of the metrics in ../../PROBLEM-STATEMENT.md.
        pw, ph = x1 - x0, y1 - y0
        max_text_w = int(pw * BALLOON_MAX_FRAC)
        cursor_y = y0 + BORDER + 6
        limit_y = y0 + int(ph * BALLOON_TOP_FRAC)

        side = p.speaker_position if p.speaker_position in ("left", "right") else "center"
        placed: list[tuple[tuple[int, int, int, int], list[str]]] = []

        for turn in p.dialogue:
            line = str(turn.get("line", "")).strip()
            if not line:
                continue
            wrapped = _wrap(draw, line, font, max_text_w)
            bw = int(max((draw.textlength(w, font=font) for w in wrapped), default=0)) + 2 * BALLOON_PAD
            bh = len(wrapped) * (font.size + 2) + 2 * BALLOON_PAD
            if cursor_y + bh > limit_y:
                break  # out of room; a crude compositor drops rather than overlaps
            bx0 = {"left": x0 + BORDER + 6,
                   "right": x1 - BORDER - 6 - bw}.get(side, x0 + (pw - bw) // 2)
            bx0 = max(x0 + BORDER + 2, min(bx0, x1 - BORDER - 2 - bw))
            placed.append(((bx0, cursor_y, bx0 + bw, cursor_y + bh), wrapped))
            # Gap must stay wide even though no tail is drawn between stacked
            # balloons. At 6px Magi merged an adjacent pair into one text region
            # and OCR'd only the lower line, losing "I will gnaw the rope net
            # and free you." entirely. TAIL + 6 is the separation that survives.
            cursor_y += bh + TAIL + 6

        for k, (bbox, wrapped) in enumerate(placed):
            _draw_balloon(draw, bbox, wrapped, font, side, tail=(k == len(placed) - 1))
            balloon_boxes.append(bbox)
            balloon_owner.append(slot)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)
    return PageResult(out, boxes, balloon_boxes, order, balloon_owner)


def compose_from_scenes(
    scene_images: Iterable[str | Path],
    scenes: Sequence[dict],
    out_path: str | Path,
    reading: ReadingOrder = "ltr",
) -> PageResult:
    """Convenience path from pipeline output: selected.png per scene plus parsed scenes.

    ``scenes`` are parser output, so each may carry ``dialogue`` (optional, see
    llm/parser.py) and a layout whose characters have a ``position``.
    """
    panels: list[Panel] = []
    for img_path, scene in zip(scene_images, scenes):
        pos = "center"
        layout = scene.get("layout") or {}
        chars = layout.get("characters") or []
        if chars and isinstance(chars[0], dict):
            pos = chars[0].get("position", "center")
        panels.append(Panel(
            image=Image.open(img_path),
            dialogue=list(scene.get("dialogue") or []),
            speaker_position=pos,
        ))
    return compose_page(panels, out_path, reading=reading)
