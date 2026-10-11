"""Contract C4 metrics: numbers for each layer of the shown slice, from the masks and pixel spacing.

No AI and no randomness: the same input always gives the same report (byte-identical JSON).

    pipeline.Result -> report(result) -> MetricsReport     (what POST /api/analyze returns as `metrics`)
    masks + slice   -> compute(...)   -> MetricsReport     (one slice, e.g. from a C2 folder)

Coordinates are [x, y] = [column, row] in slice pixels, before C2 square padding.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from cortexview.export import window01
from cortexview.layers import load_layers
from cortexview.segment import Params, _smooth, midline_plane

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))      # shared/contracts.py sits at the repo root, next to backend/
from shared.contracts import Intensity, LayerMetrics, MetricsReport, TextItem  # noqa: E402

TEMPLATES_JSON = ROOT / "shared" / "text_templates.json"

SMALL_REGION_PX = 50        # fewer pixels: warning `small_region`, and an auto layer drops to `low`
MANY_COMPONENTS = 10        # more lesion blobs: warning `many_components` (GM/CSF are always fragmented)
SIDE_SHARE = 0.85           # this share of a layer on one side of the midline -> that hemisphere ...
MIDLINE_MM = 5.0            # ... otherwise centroid this close to the midline -> "midline", else "both"
EXTRACRANIAL = "skull"                              # outside the intracranial space (no % or brain ratio)
BRAIN_LAYERS = ("gray_matter", "white_matter")      # reference for intensity.ratio_to_brain
# The segmenter splits CSF/GM/WM only on the slices it was asked for (tissue_slices); head, brain
# boundary and lesion are always 3D. So these layers have a whole-volume size only after a full run.
TISSUE_LAYERS = ("csf", "gray_matter", "white_matter")


def report(result) -> MetricsReport:
    """C4 report for a pipeline.Result."""
    vol, sl = result.volume, result.slice
    spacing = tuple(float(s) for s in vol.spacing)
    return compute(sl.masks, result.image, spacing[1:], vol.patient_left, sl.confidence, sl.source,
                   head=sl.head, intracranial=sl.intracranial,
                   volumes_ml=volumes_ml(result.segmentation, spacing))


def volumes_ml(seg, spacing) -> dict[str, float | None]:
    """Whole-volume size of each layer whose 3D mask is complete; None otherwise (2D, or tissue split
    only on some slices)."""
    voxel_ml = float(np.prod(spacing)) / 1000.0
    out = {}
    for lid, m in seg.masks.items():
        partial = seg.tissue_range is not None and lid in TISSUE_LAYERS
        out[lid] = None if m.ndim != 3 or partial else _r(m.sum() * voxel_ml)
    return out


def compute(masks, image, spacing_mm, patient_left, confidence, source,
            head=None, intracranial=None, volumes_ml=None) -> MetricsReport:
    """One slice.

    masks:        {layer id: 2D bool}, mutually exclusive (C1)
    image:        the slice as loaded; windowed here to 0..1 inside the head, like mri.png
    spacing_mm:   [row, col] in mm, or None if unknown
    patient_left: "image_left" | "image_right" | None
    confidence, source: {layer id: ...} from the segmenter
    head, intracranial: 2D bool; default to all layers / all layers but skull
    volumes_ml:   {layer id: mL or None}, see volumes_ml()
    """
    layers = load_layers()
    volumes_ml = volumes_ml or {}
    union = lambda ids: np.logical_or.reduce([np.asarray(masks[i], bool) for i in ids])
    if head is None:
        head = union([l.id for l in layers])
    if intracranial is None:
        intracranial = union([l.id for l in layers if l.id != EXTRACRANIAL])
    img = window01(np.asarray(image, np.float64), head)
    sp = None if spacing_mm is None else (float(spacing_mm[0]), float(spacing_mm[1]))
    px_mm2 = None if sp is None else sp[0] * sp[1]
    ic_px = int(intracranial.sum())
    brain = union(BRAIN_LAYERS)
    brain_mean = float(img[brain].mean()) if brain.any() else 0.0
    dist, side_warnings = (_midline_distance(img, intracranial, np.asarray(masks["lesion"], bool), sp)
                           if patient_left else (None, []))

    out = {}
    for l in layers:
        m = np.asarray(masks[l.id], bool)
        area = int(m.sum())
        area_mm2 = None if px_mm2 is None else _r(area * px_mm2)
        pct = None if l.id == EXTRACRANIAL or ic_px == 0 else _r(100.0 * area / ic_px)
        if area == 0:
            out[l.id] = LayerMetrics(
                present=False, area_px=0, area_mm2=area_mm2, pct_intracranial=pct, n_components=0,
                centroid_px=None, bbox_px=None, max_diameter_mm=None, volume_ml=volumes_ml.get(l.id),
                intensity=None, hemisphere=None, anchor_px=None, confidence="empty", warnings=[])
            continue

        warnings, conf = [], confidence[l.id]
        lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
        if l.id == "lesion" and n > MANY_COMPONENTS:
            warnings.append("many_components")
        largest = lab == int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
        rows, cols = np.nonzero(m)

        # anchor: the point deepest inside the largest blob (first in row-major order on ties)
        depth = ndi.distance_transform_edt(largest, sampling=sp or (1.0, 1.0))
        ar, ac = np.unravel_index(int(np.argmax(depth)), depth.shape)

        hemisphere = None
        if dist is not None:
            d = dist[m]
            image_side = ("image_left" if (d < 0).mean() >= SIDE_SHARE else
                          "image_right" if (d > 0).mean() >= SIDE_SHARE else None)
            if image_side:
                hemisphere = "left" if image_side == patient_left else "right"
            else:
                hemisphere = "midline" if abs(float(d.mean())) <= MIDLINE_MM else "both"
            warnings += side_warnings

        vals = img[m]
        ratio = None if l.id == EXTRACRANIAL or brain_mean <= 0 else _r(vals.mean() / brain_mean, 2)

        if area < SMALL_REGION_PX:
            warnings.append("small_region")
            if source.get(l.id) == "auto" and conf in ("high", "medium"):
                conf = "low"                    # doctor labels keep their confidence

        out[l.id] = LayerMetrics(
            present=True, area_px=area, area_mm2=area_mm2, pct_intracranial=pct, n_components=int(n),
            centroid_px=(_r(cols.mean()), _r(rows.mean())),
            bbox_px=(int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())),
            max_diameter_mm=_diameter_mm(largest, sp) if l.id == "lesion" and sp else None,
            volume_ml=volumes_ml.get(l.id),
            intensity=Intensity(mean=_r(vals.mean(), 2), std=_r(vals.std(), 2), ratio_to_brain=ratio),
            hemisphere=hemisphere, anchor_px=(int(ac), int(ar)), confidence=conf, warnings=warnings)

    return MetricsReport(intracranial_area_mm2=None if px_mm2 is None else _r(ic_px * px_mm2),
                         layers=out, text=_text(out, layers))


def _midline_distance(img, intracranial, lesion, sp):
    """Signed distance (mm) of every pixel from the brain's left/right mirror line, positive towards
    image right. Fitted the same way as the segmenter's lesion midline (on its ~2 mm scale, and without
    the lesion, which is what breaks the symmetry); if that fit is not trusted, the middle of the
    brain's width is used instead (warning `midline_estimated`)."""
    sp, p = sp or (1.0, 1.0), Params()
    rows, cols = np.indices(img.shape)
    with np.errstate(invalid="ignore", divide="ignore"):    # flat images: correlation is undefined
        fit = midline_plane(_smooth(img, p.work_mm, sp), intracranial & ~lesion, sp)
    if fit is not None and fit[2] > p.midline_min_corr:
        normal, offset, _ = fit                              # normal points towards image right
        return rows * sp[0] * normal[0] + cols * sp[1] * normal[1] - offset, []
    c = np.nonzero(intracranial.any(axis=0))[0]
    mid = (c.min() + c.max()) / 2.0 if len(c) else img.shape[1] / 2.0
    return (cols - mid) * sp[1], ["midline_estimated"]


def _diameter_mm(mask, sp) -> float:
    """Longest distance between two pixel centres of `mask`, in mm."""
    pts = np.argwhere(mask) * np.asarray(sp)
    if len(pts) == 1:
        return _r(max(sp))
    try:
        pts = pts[ConvexHull(pts).vertices]
    except QhullError:
        pass                                    # collinear or too few points: use them all
    return _r(pdist(pts).max())


@lru_cache(maxsize=1)
def load_templates() -> dict[str, dict]:
    """Fixed sentence templates (C4 `text`), from shared/text_templates.json."""
    data = json.loads(TEMPLATES_JSON.read_text(encoding="utf-8"))
    return {t["template_id"]: t for t in data}


def _text(metrics, layers) -> list[TextItem]:
    """Fill the fixed templates; no generated prose."""
    templates, out = load_templates(), []

    def add(layer, template_id, **values):
        out.append(TextItem(layer=layer, template_id=template_id,
                            text=templates[template_id]["text"].format(**values)))

    les = metrics.get("lesion")
    if les is not None and les.present:
        if les.area_mm2 is not None and les.pct_intracranial is not None:
            add("lesion", "lesion_area", area_cm2=_say(les.area_mm2 / 100.0),
                pct_intracranial=_say(les.pct_intracranial))
        if les.max_diameter_mm is not None:
            add("lesion", "lesion_diameter", max_diameter_mm=_say(les.max_diameter_mm))
        if les.hemisphere in ("left", "right"):
            add("lesion", "lesion_side", hemisphere=les.hemisphere)
    for l in layers:
        v = metrics[l.id].volume_ml
        if v:
            add(l.id, "layer_volume", layer_name=l.name, volume_ml=_say(v))
    return out


def _r(x, nd: int = 1) -> float:
    return round(float(x), nd)


def _say(x: float) -> str:
    """A number for a sentence: one decimal below 10, whole numbers above."""
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"
