"""Reusable building blocks: phantom images, pixel grids, legends, formula text and pipeline nodes."""
from __future__ import annotations

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Circle, Dot, ImageMobject, Line, MarkupText,
                   RESAMPLING_ALGORITHMS, Rectangle, RoundedRectangle, Sector, Square, Text, VGroup)

from . import colors as C
from .phantom import grey

# ---------------------------------------------------------------- images from the phantom arrays


def rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = alpha
    return out


def mri_rgba(mri: np.ndarray, lbl: np.ndarray, dim: float = 1.0) -> np.ndarray:
    """Greyscale 'MRI' of the phantom (grey() mapping from phantom.ts), transparent outside the head."""
    v = grey(mri) * dim
    return rgba(np.repeat(v[..., None], 3, axis=2), np.where(lbl > 0, 255, 0))


def labels_rgba(lbl: np.ndarray, only: list[int] | None = None) -> np.ndarray:
    """Flat palette colors per label (the 'guide' look), optionally keeping only some labels."""
    rgb = np.zeros(lbl.shape + (3,), np.float64)
    alpha = np.zeros(lbl.shape, np.uint8)
    for L in C.LAYERS:
        if only is not None and L["label"] not in only:
            continue
        m = lbl == L["label"]
        rgb[m] = L["rgb"]
        alpha[m] = 255
    return rgba(rgb, alpha)


def illus_rgba(lbl: np.ndarray, mri: np.ndarray, D: np.ndarray) -> np.ndarray:
    """The frontend prototype's shaded illustration (illus() in phantom.ts): palette color, shaded by
    distance to the edge and by the image. This is code, not model output; scene 4 says so on screen."""
    n = lbl.shape[0]
    h = n / 2
    j, i = np.indices(lbl.shape, dtype=np.float64)
    x, y = (i - h) / h, (j - h) / h
    s = (0.74 + 0.26 * np.minimum(1, D / 7)) * (1 + 0.12 * (-x - y) * 0.6)
    s = np.where(lbl == 1, s * (0.72 + 0.42 * mri),
                 np.where(lbl == 10, s * (0.85 + 0.35 * mri), s * (1 + (mri - 0.4) * 0.25)))
    s = np.where(D < 1.1, s * 0.6, s)
    base = labels_rgba(lbl)
    rgb = np.minimum(255, base[..., :3].astype(np.float64) * s[..., None])
    return rgba(rgb, base[..., 3])


def image(arr: np.ndarray, height: float, nearest: bool = False) -> ImageMobject:
    im = ImageMobject(arr)
    if nearest:
        im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


# ---------------------------------------------------------------- pixel grids


def pixel_grid(rgb: np.ndarray, side: float, stroke: float = 0.6) -> VGroup:
    """One Square per pixel (row-major), filled with rgb[j, i]. For small grids only (<= ~50 x 50)."""
    n_rows, n_cols = rgb.shape[:2]
    cells = VGroup()
    for jj in range(n_rows):
        for ii in range(n_cols):
            sq = Square(side, stroke_width=stroke, stroke_color=C.FAINT)
            sq.set_fill(_hex(rgb[jj, ii]), 1)
            sq.move_to(np.array([(ii - (n_cols - 1) / 2) * side, ((n_rows - 1) / 2 - jj) * side, 0]))
            cells.add(sq)
    return cells


def grid_lines(n: int, side: float, color=C.FAINT, width: float = 0.8) -> VGroup:
    """Lines only, for a large pixel grid drawn over an image with nearest-neighbour resampling."""
    half = n * side / 2
    g = VGroup()
    for k in range(n + 1):
        t = -half + k * side
        g.add(Line([t, -half, 0], [t, half, 0], stroke_width=width, color=color))
        g.add(Line([-half, t, 0], [half, t, 0], stroke_width=width, color=color))
    return g


def run_rects(mask: np.ndarray, side: float, color) -> tuple[VGroup, list[int]]:
    """The mask's horizontal runs of pixels as rectangles, in raster order (top row first), plus the
    pixel count of each run. Centered like grid_lines(mask.shape[0], side)."""
    n = mask.shape[0]
    half = n * side / 2
    rects, counts = VGroup(), []
    for jj in range(n):
        row = mask[jj]
        ii = 0
        while ii < n:
            if row[ii]:
                start = ii
                while ii < n and row[ii]:
                    ii += 1
                w = (ii - start) * side
                r = Rectangle(width=w, height=side, stroke_width=0).set_fill(color, 1)
                r.move_to([-half + start * side + w / 2, half - (jj + 0.5) * side, 0])
                rects.add(r)
                counts.append(ii - start)
            else:
                ii += 1
    return rects, counts


def pixel_center(i: float, j: float, n: int, side: float) -> np.ndarray:
    """Scene position of pixel (i, j) in a grid of n x n cells of `side`, centered at the origin."""
    half = n * side / 2
    return np.array([-half + (i + 0.5) * side, half - (j + 0.5) * side, 0])


def _hex(c) -> str:
    r, g, b = (int(round(float(v))) for v in c[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------- text


def text(s: str, size: float = 30, color=C.INK, weight: str = "NORMAL", font: str = C.SANS, **kw) -> Text:
    return Text(s, font=font, font_size=size, color=color, weight=weight, **kw)


def math(markup: str, size: float = 44, color=C.INK) -> MarkupText:
    """Formula text in Cambria; variables in <i>…</i>. Avoids a LaTeX dependency."""
    return MarkupText(markup, font=C.SERIF, font_size=size, color=color)


def headline(s: str) -> Text:
    return text(s, 40, C.INK).to_edge(UP, buff=0.55)


def phantom_tag() -> VGroup:
    """Persistent corner note: what's on screen is synthetic."""
    dot = Dot(radius=0.045, color=C.MATH)
    t = text("Synthetic phantom · generated by code · not a patient", 17, C.MUTED)
    return VGroup(dot, t).arrange(RIGHT, buff=0.15).to_corner(DOWN + LEFT, buff=0.35)


def legend_row(layer_id: str, size: float = 24) -> VGroup:
    L = C.LAYER[layer_id]
    sw = RoundedRectangle(corner_radius=0.05, width=0.32, height=0.32, stroke_width=0)
    sw.set_fill(C.layer_color(layer_id), 1)
    num = text(str(L["label"]), size * 0.8, C.MUTED, font=C.MONO)
    name = text(C.SHORT_NAME[layer_id], size, C.INK)
    return VGroup(sw, num, name).arrange(RIGHT, buff=0.2)


# ---------------------------------------------------------------- pipeline diagram


def status_dot(status: str, radius: float = 0.09) -> VGroup:
    col = C.STATUS[status]
    if status == "working":
        return VGroup(Circle(radius, color=col, fill_opacity=1, stroke_width=0))
    if status == "progress":           # ring with the left half filled
        ring = Circle(radius, color=col, stroke_width=2.5)
        half = Sector(radius=radius, start_angle=np.pi / 2, angle=np.pi, color=col, fill_opacity=1,
                      stroke_width=0)
        return VGroup(ring, half)
    return VGroup(Circle(radius, color=col, stroke_width=2.5))


def node(title: str, sub: str, status: str, width: float = 2.55, accent=C.INK) -> VGroup:
    """A rounded box with a title, a small status line and a status dot."""
    box = RoundedRectangle(corner_radius=0.14, width=width, height=1.05, stroke_width=1.6,
                           stroke_color=accent, stroke_opacity=0.55)
    box.set_fill(C.PANEL, 1)
    t = text(title, 21, C.INK)
    if t.width > width - 0.3:
        t.scale_to_fit_width(width - 0.3)
    dot = status_dot(status, 0.07)
    s = text(sub, 15, C.MUTED)
    if s.width > width - 0.55:
        s.scale_to_fit_width(width - 0.55)
    sub_line = VGroup(dot, s).arrange(RIGHT, buff=0.12)
    body = VGroup(t, sub_line).arrange(DOWN, buff=0.16).move_to(box)
    return VGroup(box, body)


def status_legend() -> VGroup:
    items = VGroup()
    for st, label in (("working", "works today"), ("progress", "in progress"), ("planned", "planned")):
        items.add(VGroup(status_dot(st, 0.08), text(label, 18, C.MUTED)).arrange(RIGHT, buff=0.14))
    return items.arrange(RIGHT, buff=0.45)
