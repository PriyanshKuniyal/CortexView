"""Contract C2: write one slice folder for offline use (FT training/evaluation, demo cache).

slices/<slice_id>/
  meta.json  mri.png  label.png  guide.png  masks/<id>.png  [reference/<id>.png]

Images are padded to a square, never resized, so one pixel keeps the same size in mm. The slice id must
be anonymous (e.g. s0042): this module never writes patient or dataset identifiers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .layers import label_of, load_layers


def pad_square(a: np.ndarray, value=0):
    """Pad a 2D (or HxWxC) array to a square, centred. Returns (padded, [top, bottom, left, right])."""
    h, w = a.shape[:2]
    n = max(h, w)
    top, left = (n - h) // 2, (n - w) // 2
    pad = [top, n - h - top, left, n - w - left]
    widths = [(pad[0], pad[1]), (pad[2], pad[3])] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, widths, constant_values=value), pad


def window01(slice_img: np.ndarray, region: np.ndarray | None = None, lo_pct=1.0, hi_pct=99.0):
    """Percentile window (inside `region`, e.g. the head) to 0..1."""
    vals = slice_img[region] if region is not None and region.any() else slice_img.ravel()
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    return np.clip((slice_img - lo) / max(hi - lo, 1e-6), 0, 1)


def window_8bit(slice_img: np.ndarray, region: np.ndarray | None = None, lo_pct=1.0, hi_pct=99.0):
    """Percentile window (inside `region`, e.g. the head) to 0..255."""
    return (window01(slice_img, region, lo_pct, hi_pct) * 255).round().astype(np.uint8)


def label_map(masks: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros(next(iter(masks.values())).shape, np.uint8)
    for lid, m in masks.items():
        out[m] = label_of(lid)
    return out


def guide_rgb(labels: np.ndarray) -> np.ndarray:
    """Label map painted with the C1 layer colours (background black)."""
    rgb = np.zeros(labels.shape + (3,), np.uint8)
    for layer in load_layers():
        rgb[labels == layer.label] = layer.rgb
    return rgb


def write_slice_folder(root, slice_id: str, mri: np.ndarray, masks: dict[str, np.ndarray], meta: dict,
                       head: np.ndarray | None = None, reference: dict[str, np.ndarray] | None = None) -> Path:
    """Write one C2 folder. `mri` is the raw 2D slice; `masks` are mutually exclusive bool masks keyed by
    layer id (C1); `meta` carries the C2 fields this function cannot know (source, split, sequence, ...).
    Returns the folder path."""
    if not re.fullmatch(r"s\d{4,}", slice_id):
        raise ValueError("slice_id must be anonymous, like s0042")
    ids = [l.id for l in load_layers()]
    masks = {lid: np.asarray(masks.get(lid, np.zeros(mri.shape, bool)), bool) for lid in ids}
    stacked = np.stack([masks[lid] for lid in ids])
    if (stacked.sum(0) > 1).any():
        raise ValueError("layer masks overlap; C1 requires exactly one label per pixel")

    folder = Path(root) / slice_id
    (folder / "masks").mkdir(parents=True, exist_ok=True)
    mri8, pad = pad_square(window_8bit(mri, head))
    labels, _ = pad_square(label_map(masks))
    Image.fromarray(mri8).save(folder / "mri.png")
    Image.fromarray(labels).save(folder / "label.png")
    Image.fromarray(guide_rgb(labels)).save(folder / "guide.png")
    for lid in ids:
        m, _ = pad_square(masks[lid].astype(np.uint8) * 255)
        Image.fromarray(m).save(folder / "masks" / f"{lid}.png")
    ref_ids = []
    for lid, m in (reference or {}).items():
        (folder / "reference").mkdir(exist_ok=True)
        r, _ = pad_square(np.asarray(m, bool).astype(np.uint8) * 255)
        Image.fromarray(r).save(folder / "reference" / f"{lid}.png")
        ref_ids.append(lid)

    full = {
        "slice_id": slice_id,
        "source": meta.get("source", "upload"),
        "split": meta.get("split", "demo"),
        "sequence": meta.get("sequence", "unknown"),
        "sequence_source": meta.get("sequence_source", "default"),
        "plane": meta.get("plane", "axial"),
        "slice_index": meta.get("slice_index"),
        "n_slices": meta.get("n_slices"),
        "pixel_spacing_mm": meta.get("pixel_spacing_mm"),
        "slice_thickness_mm": meta.get("slice_thickness_mm"),
        "pad": pad,
        "patient_left": meta.get("patient_left"),
        "reference_layers": sorted(ref_ids),
    }
    # extra, non-contract fields that help consumers; prefixed so they never clash with C2 names
    for k in ("layer_source", "layer_confidence", "warnings"):
        if k in meta:
            full["x_" + k] = meta[k]
    (folder / "meta.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    return folder
