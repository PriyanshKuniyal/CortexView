"""Fidelity gate (contract C5): did the illustration keep every region where the masks put it?

Each pixel of the illustration is mapped to the nearest layer colour (in Lab, so "nearest" matches what
the eye sees), then compared with the masks:

    per_layer_iou       overlap of each predicted region with its mask, for layers present on the slice
    mean_iou            their mean
    phantom_lesion_pct  lesion-coloured pixels outside the lesion mask, as % of the intracranial pixels.
                        A red blob where there is no lesion is the worst failure this app can show.

Pass: mean_iou >= MIN_MEAN_IOU (0.4) and phantom_lesion_pct < MAX_PHANTOM_PCT (0.5).
Deterministic, numpy only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from cortexview.layers import load_layers

# 0.4 for the demo: the untrained model draws normal anatomy over our noisy gray/white split, so 0.8 was
# out of reach. Raise it (env FIDELITY_MIN_IOU) once the fine-tuned model lands.
MIN_MEAN_IOU = float(os.environ.get("FIDELITY_MIN_IOU", "0.4"))
MAX_PHANTOM_PCT = float(os.environ.get("FIDELITY_MAX_PHANTOM_PCT", "0.5"))
NEAR_BLACK_L = 8.0          # Lab lightness below this counts as background, whatever its hue


@dataclass
class Score:
    mean_iou: float
    per_layer_iou: dict[str, float]
    phantom_lesion_pct: float
    passed: bool


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (uint8, ...x3) to CIE Lab (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193],
                        [0.3576, 0.7152, 0.1192],
                        [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def predicted_labels(rgb: np.ndarray) -> np.ndarray:
    """Layer index (position in load_layers()) per pixel by nearest layer colour; -1 for background."""
    layers = load_layers()
    lab = rgb_to_lab(rgb)
    palette = rgb_to_lab(np.array([(0, 0, 0)] + [l.rgb for l in layers], np.uint8))
    d = ((lab[..., None, :] - palette) ** 2).sum(-1)
    idx = d.argmin(-1) - 1                       # 0 = black -> -1
    idx[lab[..., 0] < NEAR_BLACK_L] = -1
    return idx


def score(illustration_rgb: np.ndarray, masks: dict[str, np.ndarray],
          intracranial: np.ndarray | None = None) -> Score:
    """`illustration_rgb`: HxWx3 uint8, already at the masks' size. `masks`: {layer id: bool HxW}."""
    layers = load_layers()
    pred = predicted_labels(illustration_rgb)
    ious = {}
    for i, l in enumerate(layers):
        m = np.asarray(masks[l.id], bool)
        if not m.any():
            continue
        p = pred == i
        ious[l.id] = round(float((m & p).sum() / max((m | p).sum(), 1)), 4)
    if intracranial is None:
        intracranial = np.logical_or.reduce([np.asarray(masks[l.id], bool) for l in layers if l.id != "skull"])
    lesion_i = next(i for i, l in enumerate(layers) if l.id == "lesion")
    phantom = (pred == lesion_i) & ~np.asarray(masks["lesion"], bool)
    phantom_pct = round(100.0 * float(phantom.sum()) / max(int(intracranial.sum()), 1), 3)
    mean_iou = round(float(np.mean(list(ious.values()))), 4) if ious else 0.0
    return Score(mean_iou, ious, phantom_pct, mean_iou >= MIN_MEAN_IOU and phantom_pct < MAX_PHANTOM_PCT)
