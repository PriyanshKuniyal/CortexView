"""Classical (non-AI) brain MRI segmentation into the C1 layers.

Pure math on voxel intensities: thresholds (Otsu / multi-Otsu), distance transforms, connected
components and robust statistics. No randomness, no learned weights: the same input always gives the
same masks. Works on a 3D volume (preferred) or a single 2D slice; every size is in mm, so voxels that
are not cubes (e.g. 2.5 mm thick slices) are handled correctly.

Pipeline:
  1. head mask      Otsu threshold -> close -> largest blob -> fill holes
  2. window         1st..99th percentile inside the head -> 0..1
  3. intracranial   "not dark" tissue -> erode (cuts brain loose from scalp) -> largest blob
                    -> grow back -> close + fill (adds sulci and ventricles). skull = head - intracranial
  4. lesion         doctor label if given; else bright outliers (median + k*MAD) on FLAIR/T2
  5. CSF / GM / WM  multi-Otsu (3 classes) inside the intracranial mask; order depends on the sequence
  6. exclusivity    lesion > csf > white_matter > gray_matter > skull

Contract C6: segment_volume(volume, meta) and segment(slice, meta) return {layer_id: bool mask}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu, threshold_otsu

from .layers import PRIORITY, layer_ids


@dataclass(frozen=True)
class Params:
    work_mm: float = 2.0            # head/brain masks are found on a coarse grid of about this voxel size
    smooth_mm: float = 0.5          # Gaussian denoise before thresholding
    head_close_mm: float = 3.0
    brain_erode_mm: float = 4.0     # breaks thin bridges between brain and scalp/eyes
    brain_depth_mm: float = 15.0    # the brain's core lies at least this deep under the head surface;
                                    # scalp, fat and bone marrow (bright on T1) never do
    brain_close_mm: float = 6.0     # pulls sulci + thin outer CSF into the intracranial mask
    fine_mm: float = 0.9            # finest grid worked on; finer scans (e.g. 0.47 mm) are block-averaged
    bias_mm: float = 30.0           # smoothness of the brightness-drift (bias field) estimate
    csf_k: float = 2.0             # CSF = this many robust SDs darker (T1/FLAIR) or brighter (T2) than tissue
    local_mm: float = 6.0           # neighbourhood size for local GM/WM contrast
    local_gain: float = 1.5         # how sharply local contrast (in local SDs) votes GM vs WM
    cortex_mm: float = 4.0          # gray-matter depth scale for the cortex prior
    cortex_prior: float = 0.35      # 0 = local contrast only, 1 = geometry decides GM vs WM
    posterior_smooth_mm: float = 0.7
    lesion_k: float = 3.0           # outlier threshold in robust SDs (MAD * 1.4826)
    lesion_rim_mm: float = 5.0      # bright blobs living mostly this close to the skull are rejected
    lesion_min_ml: float = 0.3      # 3D minimum lesion size
    lesion_min_mm2: float = 25.0    # 2D minimum lesion size


# Darkest -> brightest tissue class per sequence (TASKS §3).
CLASS_ORDER = {
    "T1_pre": ("csf", "gray_matter", "white_matter"),
    "T1_post": ("csf", "gray_matter", "white_matter"),
    "T2": ("white_matter", "gray_matter", "csf"),
    "FLAIR": ("csf", "white_matter", "gray_matter"),
    "unknown": ("csf", "gray_matter", "white_matter"),
}

# Base confidence per sequence: how well that contrast separates the layer.
CONFIDENCE = {
    "T1_pre":  {"skull": "medium", "gray_matter": "medium", "white_matter": "medium", "csf": "medium", "lesion": "low"},
    "T1_post": {"skull": "medium", "gray_matter": "medium", "white_matter": "medium", "csf": "medium", "lesion": "low"},
    "T2":      {"skull": "medium", "gray_matter": "medium", "white_matter": "medium", "csf": "medium", "lesion": "low"},
    "FLAIR":   {"skull": "medium", "gray_matter": "low", "white_matter": "low", "csf": "medium", "lesion": "medium"},
    "unknown": {"skull": "low", "gray_matter": "low", "white_matter": "low", "csf": "low", "lesion": "low"},
}


@dataclass
class Segmentation:
    masks: dict[str, np.ndarray]                  # bool, mutually exclusive, input shape
    confidence: dict[str, str]                    # high | medium | low | empty
    source: dict[str, str]                        # auto | doctor
    warnings: list[str] = field(default_factory=list)
    head: np.ndarray | None = None
    intracranial: np.ndarray | None = None        # brain + csf + lesion (C4 denominator)
    stats: dict = field(default_factory=dict)     # tissue intensities + class separation
    slice_index: int | None = None                # C3 auto slice (largest lesion, else mid-brain)
    tissue_range: tuple[int, int] | None = None   # axial slices that got CSF/GM/WM (None = all)

    def label_map(self) -> np.ndarray:
        from .layers import label_of
        out = np.zeros(next(iter(self.masks.values())).shape, dtype=np.uint8)
        for lid, m in self.masks.items():
            out[m] = label_of(lid)
        return out

    def cut(self, index: int) -> "Segmentation":
        """One axial slice (axis 0) of a 3D result. Confidence becomes `empty` where the slice is empty."""
        masks = {k: v[index] for k, v in self.masks.items()}
        conf = {k: ("empty" if not masks[k].any() else c) for k, c in self.confidence.items()}
        if self.tissue_range and not self.tissue_range[0] <= index < self.tissue_range[1]:
            raise ValueError("slice was not tissue-classified; pass tissue_slices covering it")
        return Segmentation(masks, conf, dict(self.source), list(self.warnings),
                            None if self.head is None else self.head[index],
                            None if self.intracranial is None else self.intracranial[index],
                            dict(self.stats), index, None)


# ---------------------------------------------------------------- morphology in mm

def _erode(mask, r_mm, spacing):
    return ndi.distance_transform_edt(mask, sampling=spacing) > r_mm


def _dilate(mask, r_mm, spacing):
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= r_mm


def _close(mask, r_mm, spacing):
    # pad so structures touching the array edge do not get eroded away by the border
    pad = [int(np.ceil(r_mm / s)) + 1 for s in spacing]
    m = np.pad(mask, [(p, p) for p in pad])
    m = _erode(_dilate(m, r_mm, spacing), r_mm, spacing)
    return m[tuple(slice(p, -p) for p in pad)]


def _largest(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _fill(mask):
    """3D hole fill plus a per-slice fill, which also closes holes that leak out through the
    top/bottom of the volume (sinuses, ear canals)."""
    out = ndi.binary_fill_holes(mask)
    if out.ndim == 3:
        for k in range(out.shape[0]):
            out[k] = ndi.binary_fill_holes(out[k])
    return out


def _smooth(img, sigma_mm, spacing):
    return ndi.gaussian_filter(img, sigma=[sigma_mm / s for s in spacing], mode="nearest")


def _factors(spacing, work_mm):
    return tuple(max(1, int(round(work_mm / s))) for s in spacing)


def _down(img, f):
    """Block-average by integer factors (edges that do not fill a block are dropped)."""
    sh = [s // ff * ff for s, ff in zip(img.shape, f)]
    img = img[tuple(slice(0, s) for s in sh)]
    shape = [x for s, ff in zip(sh, f) for x in (s // ff, ff)]
    return img.reshape(shape).mean(axis=tuple(range(1, 2 * img.ndim, 2)))


def _up(field, f, shape):
    """Back to the full grid: linear upsampling of a float field (a mask's 0/1 edges get smooth
    sub-block boundaries when thresholded at 0.5), edge-padded to the original shape."""
    if all(ff == 1 for ff in f):
        return field
    # block copy, then a blur about half a block wide: same smooth result as linear zoom, ~10x faster
    z = _repeat(field.astype(np.float32), f, shape)
    return ndi.gaussian_filter(z, [0.5 * ff if ff > 1 else 0 for ff in f], mode="nearest", truncate=2.0)


# ---------------------------------------------------------------- pipeline steps

def head_mask(img, spacing, p: Params = Params()):
    s = _smooth(img, p.smooth_mm, spacing)
    t = threshold_otsu(s)
    # Otsu on a mostly-air image lands between air and *average* tissue; half of it keeps dark scalp edges
    head = s > 0.5 * t
    head = _largest(_close(head, p.head_close_mm, spacing))
    return _fill(head)


def intracranial_mask(n, head, spacing, p: Params = Params()):
    """Brain + CSF inside the skull (run on the coarse grid).

    Relies on two facts that hold on T1, T2 and FLAIR: the skull bone is dark, and the brain is a big
    blob sitting deep inside the head. Scalp fat and bone marrow can be bright and touch the brain
    through thin bridges, so the brain is first found as a deep, eroded core and then grown back."""
    s = _smooth(n, p.smooth_mm, spacing)
    t_dark = threshold_multiotsu(s[head], classes=3)[0]
    tissue = head & (s > t_dark)
    depth = ndi.distance_transform_edt(head, sampling=spacing)
    core = _largest(_erode(tissue, p.brain_erode_mm, spacing) & (depth > p.brain_depth_mm))
    brain = _largest(_dilate(core, p.brain_erode_mm + 1.5, spacing) & tissue)
    return _fill(_close(brain, p.brain_close_mm, spacing)) & head


def auto_lesion(n, ic, inner, spacing, sequence, p: Params = Params()):
    """Bright outliers on FLAIR/T2. `inner` = intracranial voxels away from the skull rim.
    Returns (mask, warning or None)."""
    if sequence not in ("FLAIR", "T2") or inner is None:
        return np.zeros_like(ic), "lesion_auto_needs_flair_or_t2"
    s = _smooth(n, p.smooth_mm, spacing)
    vals = s[inner]
    if vals.size == 0:
        return np.zeros_like(ic), None
    if sequence == "T2":
        # CSF is the brightest thing on T2; measure outliers against tissue only, and cap below CSF
        t_csf = threshold_multiotsu(vals, classes=3)[1]
        tissue_vals = vals[vals < t_csf]
    else:
        tissue_vals = vals
    med = float(np.median(tissue_vals))
    mad = float(np.median(np.abs(tissue_vals - med))) * 1.4826
    cand = ic & (s > med + p.lesion_k * max(mad, 1e-3))
    if sequence == "T2":
        cand &= s < t_csf
    lab, nlab = ndi.label(cand)
    if nlab == 0:
        return cand, None
    voxel = float(np.prod(spacing))
    sizes = ndi.sum_labels(np.ones_like(cand), lab, index=np.arange(1, nlab + 1)) * voxel
    inside = ndi.sum_labels(inner, lab, index=np.arange(1, nlab + 1)) * voxel
    min_size = p.lesion_min_ml * 1000 if n.ndim == 3 else p.lesion_min_mm2
    keep = np.where((sizes >= min_size) & (inside >= 0.5 * sizes))[0] + 1
    lesion = np.isin(lab, keep)
    return _fill(lesion) & ic, None


def bias_field(n_c, ic_c, cspacing, p: Params = Params()):
    """Slow brightness drift across the head (MRI coil shading), estimated on the coarse grid as a
    heavily smoothed average of mid-brightness brain voxels. Divide the image by it to correct."""
    vals = n_c[ic_c]
    if vals.size < 50:
        return np.ones_like(n_c)
    lo, hi = np.quantile(vals, [0.15, 0.85])          # skip CSF and bright outliers
    w = (ic_c & (n_c > lo) & (n_c < hi)).astype(np.float32)
    sig = [p.bias_mm / s for s in cspacing]
    field = ndi.gaussian_filter(n_c * w, sig) / np.maximum(ndi.gaussian_filter(w, sig), 1e-3)
    field = np.where(ic_c, field, np.median(field[ic_c]))
    return field / np.median(field[ic_c])


def _local_z(s, mask, sigma_mm, spacing):
    """How much brighter (+) or darker (-) each voxel is than its neighbourhood, in local SDs.
    Neighbourhood = Gaussian window of `sigma_mm`, using only voxels inside `mask`."""
    w = mask.astype(np.float32)
    g = [sigma_mm / sp for sp in spacing]
    norm = np.maximum(ndi.gaussian_filter(w, g), 1e-3)
    r = s - ndi.gaussian_filter(s * w, g) / norm
    sd = np.sqrt(ndi.gaussian_filter(r * r * w, g) / norm)
    return r / np.maximum(sd, 1e-6)


def tissue_level(n, ic, exclude, spacing, p: Params = Params()):
    """Whole-brain tissue median and robust SD (MAD * 1.4826) that the CSF threshold is based on."""
    vals = _smooth(n, p.smooth_mm, spacing)[ic & ~exclude]
    med = float(np.median(vals))
    return med, float(np.median(np.abs(vals - med))) * 1.4826


def tissue_classes(n, ic, exclude, spacing, sequence, p: Params = Params(), level=None):
    """Split the intracranial mask into CSF / GM / WM.

    1. CSF = robust brightness outliers on the fluid side: below median - k*MAD on T1/FLAIR
       (fluid is dark), above median + k*MAD on T2 (fluid is bright).
    2. GM vs WM by *local* contrast: cortex is darker than the white matter right next to it on T1
       (brighter on T2/FLAIR), even on scans where global brightness of GM and WM overlaps.
    3. Cortex prior: gray matter lies along the brain surface and CSF-filled folds, so voxels near
       CSF lean GM and deep voxels lean WM.
    4. Lightly smooth the combined score (neighbours tend to agree) and threshold at 0.5.
    `level` = (median, robust SD) of the whole brain, when only a slab of slices is passed in.
    Returns ({class: mask}, stats)."""
    region = ic & ~exclude
    out = {k: np.zeros_like(ic) for k in ("csf", "gray_matter", "white_matter")}
    if region.sum() < 50:
        return out, {}
    order = CLASS_ORDER.get(sequence, CLASS_ORDER["unknown"])
    s = _smooth(n, p.smooth_mm, spacing)
    med, mad = level or tissue_level(n, ic, exclude, spacing, p)
    csf_dark = order[0] == "csf"
    t_csf = med - p.csf_k * mad if csf_dark else med + p.csf_k * mad
    csf = region & ((s < t_csf) if csf_dark else (s > t_csf))
    tissue = region & ~csf

    gm_darker = order.index("gray_matter") < order.index("white_matter")
    z = _local_z(s, tissue, p.local_mm, spacing)
    z_gm = -z if gm_darker else z                       # + means "looks like gray matter"
    intensity = 1 / (1 + np.exp(np.clip(-p.local_gain * z_gm, -50, 50)))

    d_csf = ndi.distance_transform_edt(~(csf | ~ic), sampling=spacing)
    geometry = np.exp(-d_csf / p.cortex_mm)

    a = p.cortex_prior
    score = (1 - a) * intensity + a * geometry
    sig = [p.posterior_smooth_mm / sp for sp in spacing]
    gm = ndi.gaussian_filter(score.astype(np.float32), sig) > 0.5

    # self-check of this scan: does local contrast actually differ between cortex and deep WM?
    band = tissue & (d_csf > 1.0) & (d_csf < 3.0)
    deep = tissue & (d_csf > 12.0)
    sep = 0.0
    if band.sum() > 50 and deep.sum() > 50:
        zb, zd = z_gm[band], z_gm[deep]
        sep = float((np.median(zb) - np.median(zd)) /
                    max(np.sqrt((np.var(zb) + np.var(zd)) / 2), 1e-6))

    out["csf"] = csf
    out["gray_matter"] = tissue & gm
    out["white_matter"] = tissue & ~gm
    stats = {
        "tissue_median": med, "tissue_mad": mad, "csf_threshold": float(t_csf),
        "gm_wm_separation": sep,     # d' of local contrast, cortex band vs deep WM
        "csf_separation": float(abs(np.median(s[csf]) - med) / max(mad, 1e-6)) if csf.any() else 0.0,
    }
    return out, stats


def _grade(d, cuts):
    return "low" if d < cuts[0] else ("medium" if d < cuts[1] else "high")


# ---------------------------------------------------------------- top level

def analyze(img: np.ndarray, spacing, sequence: str = "unknown",
            doctor: dict[str, np.ndarray] | None = None, p: Params = Params(),
            tissue_slices=None) -> Segmentation:
    """Segment a 2D slice or 3D volume. `doctor` may hold bool masks on the same grid:
    'lesion', 'ventricles', 'brain' (cerebrum). Masks come back on the input grid.

    Head, brain boundary and lesion are always 3D over the whole volume. The CSF/GM/WM split is the
    slow part, so `tissue_slices` can limit it to some axial slices (3D only):
      None        every slice (offline export, volumes)
      "auto"      the C3 auto slice (largest lesion area, else mid-brain), as for the API
      int         that slice;   (start, stop)  that range
    Other slices then have no CSF/GM/WM labels (see Segmentation.tissue_range)."""
    img = np.asarray(img, dtype=np.float32)
    spacing = tuple(float(s) for s in (spacing if spacing is not None else [1.0] * img.ndim))
    doctor = {k: np.asarray(v, dtype=bool) for k, v in (doctor or {}).items() if v is not None}
    if sequence not in CLASS_ORDER:
        sequence = "unknown"
    ff = _factors(spacing, p.fine_mm)
    if all(f == 1 for f in ff):
        return _analyze(img, spacing, sequence, doctor, p, tissue_slices)

    # very fine scans: work on a ~1 mm grid, then copy labels back to every original pixel
    fine = _down(img, ff)
    fsp = tuple(s * f for s, f in zip(spacing, ff))
    ts = tissue_slices
    if img.ndim == 3 and ff[0] > 1 and ts is not None and ts != "auto":
        ts = (ts // ff[0]) if isinstance(ts, (int, np.integer)) else (ts[0] // ff[0], -(-ts[1] // ff[0]))
    seg = _analyze(fine, fsp, sequence, {k: _down(v.astype(np.float32), ff) > 0.5 for k, v in doctor.items()},
                   p, ts)
    rep = lambda m: _repeat(m, ff, img.shape)
    head, ic = rep(seg.head), rep(seg.intracranial)
    raw = {k: rep(m) for k, m in seg.masks.items()}
    if "lesion" in doctor:                                    # doctor labels stay pixel-exact
        raw["lesion"] = doctor["lesion"] & head
    if "ventricles" in doctor:
        raw["csf"] |= doctor["ventricles"] & ic
    masks = _exclusive(raw, img.shape)
    conf = {k: ("empty" if not masks[k].any() else c) for k, c in seg.confidence.items()}
    z = ff[0] if img.ndim == 3 else 1
    tr = seg.tissue_range and (seg.tissue_range[0] * z, min(img.shape[0], seg.tissue_range[1] * z))
    k = None if seg.slice_index is None else seg.slice_index * z
    return Segmentation(masks, conf, seg.source, seg.warnings, head, ic | masks["lesion"], seg.stats, k, tr)


def _repeat(mask, f, shape):
    """Nearest-neighbour upsample by integer factors, edge-padded to `shape`."""
    for ax, k in enumerate(f):
        mask = np.repeat(mask, k, axis=ax)
    pad = [(0, max(0, s - m)) for s, m in zip(shape, mask.shape)]
    return np.pad(mask, pad, mode="edge")[tuple(slice(0, s) for s in shape)]


def _exclusive(raw, shape):
    masks, taken = {}, np.zeros(shape, bool)
    for lid in PRIORITY:
        masks[lid] = raw[lid] & ~taken
        taken |= masks[lid]
    return {lid: masks[lid] for lid in layer_ids()}


def _auto_slice(lesion, ic) -> int:
    """C3 auto slice: largest lesion area, else the middle of the brain-present slices."""
    area = lesion.reshape(lesion.shape[0], -1).sum(1)
    if area.max() > 0:
        return int(np.argmax(area))
    ic_area = ic.reshape(ic.shape[0], -1).sum(1)
    ok = np.where(ic_area >= 0.25 * max(ic_area.max(), 1))[0]
    return int(ok[len(ok) // 2]) if len(ok) else lesion.shape[0] // 2


def _analyze(img, spacing, sequence, doctor, p: Params, tissue_slices=None) -> Segmentation:
    warnings = []
    source = {lid: "auto" for lid in layer_ids()}

    # head + brain boundary on a coarse (~2 mm) grid: same answer, ~10x less work
    f = _factors(spacing, p.work_mm)
    coarse = _down(img, f)
    cspacing = tuple(s * ff for s, ff in zip(spacing, f))
    head_c = head_mask(coarse, cspacing, p)
    if head_c.sum() < 25:
        empty = {lid: np.zeros(img.shape, bool) for lid in layer_ids()}
        return Segmentation(empty, {lid: "empty" for lid in empty}, source, ["no_head_found"], empty["skull"], empty["skull"])
    head = _up(head_c, f, img.shape) > 0.5
    lo, hi = np.percentile(img[head][::7], [1, 99])          # fixed stride: deterministic subsample
    scale = lambda x: np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    n_c = scale(coarse)

    # brain boundary: the doctor's cerebrum (+ tumor + ventricles, which BrainLab cuts out of it)
    if "brain" in doctor and doctor["brain"].any():
        ic = _fill(doctor["brain"] | doctor.get("lesion", False) | doctor.get("ventricles", False)) & head
        ic_c = _down(ic.astype(np.float32), f) > 0.5
        source["skull"] = "doctor"
    else:
        ic_c = intracranial_mask(n_c, head_c, cspacing, p)
        ic = (_up(ic_c, f, img.shape) > 0.5) & head
    # remove the coil's slow brightness drift so one threshold fits the whole brain
    n = scale(img) / _up(bias_field(n_c, ic_c, cspacing, p), f, img.shape)

    if "lesion" in doctor:
        lesion = doctor["lesion"] & head
        ic |= lesion
        source["lesion"] = "doctor"
    else:
        inner = None
        if sequence in ("FLAIR", "T2"):
            rim = ndi.distance_transform_edt(ic_c, sampling=cspacing)
            inner = ic & (_up(rim, f, img.shape) > p.lesion_rim_mm)
        lesion, w = auto_lesion(n, ic, inner, spacing, sequence, p)
        if w:
            warnings.append(w)

    # whole-brain tissue level for the CSF threshold, from the coarse grid (cheap, same in every mode)
    n_cc = _down(n, f)
    level = tissue_level(n_cc, _down((ic & ~lesion).astype(np.float32), f) > 0.5,
                         np.zeros(n_cc.shape, bool), cspacing, p)
    k_auto = _auto_slice(lesion, ic) if img.ndim == 3 else None
    ts = k_auto if isinstance(tissue_slices, str) and tissue_slices == "auto" else tissue_slices
    if img.ndim == 3 and ts is not None:
        z0, z1 = (int(ts), int(ts) + 1) if isinstance(ts, (int, np.integer)) else (int(ts[0]), int(ts[1]))
        z0, z1 = max(0, z0), min(img.shape[0], z1)
        # neighbours the local-contrast blur reaches (4 sigma) plus the smaller smoothing steps
        m = int(np.ceil((4 * p.local_mm + 4) / spacing[0]))
        a, b = max(0, z0 - m), min(img.shape[0], z1 + m)
        sub, stats = tissue_classes(n[a:b], ic[a:b], lesion[a:b], spacing, sequence, p, level)
        classes = {}
        for name, msk in sub.items():
            full = np.zeros(img.shape, bool)
            full[z0:z1] = msk[z0 - a:z1 - a]
            classes[name] = full
        tissue_range = (z0, z1)
    else:
        classes, stats = tissue_classes(n, ic, lesion, spacing, sequence, p, level)
        tissue_range = None
    if "ventricles" in doctor:
        classes["csf"] |= doctor["ventricles"] & ic
        warnings.append("csf_includes_doctor_ventricles")
    masks = _exclusive(dict(classes, lesion=lesion, skull=head & ~ic), img.shape)

    # confidence: the sequence's ceiling, lowered further when the fitted classes overlap
    rank = {"low": 0, "medium": 1, "high": 2}
    conf = dict(CONFIDENCE[sequence])
    if stats:
        fit_gw = _grade(stats["gm_wm_separation"], (1.0, 2.0))
        fit_csf = _grade(stats["csf_separation"], (3.5, 5.0))   # robust SDs between CSF and tissue
        for lid, fit in (("gray_matter", fit_gw), ("white_matter", fit_gw), ("csf", fit_csf)):
            conf[lid] = min(conf[lid], fit, key=rank.get)
        if fit_gw == "low":
            warnings.append("gm_wm_low_contrast")
    if source["lesion"] == "doctor":
        conf["lesion"] = "high"
    if source["skull"] == "doctor":
        conf["skull"] = "high"
    conf = {lid: ("empty" if not masks[lid].any() else conf[lid]) for lid in masks}
    if tissue_range:
        warnings.append("tissue_partial")
    return Segmentation(masks, conf, source, warnings, head, ic, stats, k_auto, tissue_range)


def segment_volume(volume, meta: dict | None = None) -> dict[str, np.ndarray]:
    """C6 interface. `volume` is a cortexview.volume.Volume or a 3D array (then meta must give
    'spacing_mm' [k, j, i] and 'sequence'). meta may carry 'doctor' masks."""
    return analyze_volume(volume, meta).masks


def analyze_volume(volume, meta: dict | None = None) -> Segmentation:
    meta = meta or {}
    if hasattr(volume, "data"):
        img, spacing, seq = volume.data, volume.spacing, volume.sequence
    else:
        img, spacing, seq = volume, meta.get("spacing_mm"), meta.get("sequence", "unknown")
    return analyze(img, spacing, meta.get("sequence", seq), meta.get("doctor"), meta.get("params", Params()),
                   meta.get("tissue_slices"))


def segment(slice_img, meta: dict | None = None) -> dict[str, np.ndarray]:
    """C6 interface. If meta carries the whole series ('volume'), segment in 3D and cut out
    meta['slice_index']; otherwise segment the 2D slice on its own (lower quality)."""
    meta = meta or {}
    if meta.get("volume") is not None:
        k = int(meta["slice_index"])
        return analyze_volume(meta["volume"], dict(meta, tissue_slices=k)).cut(k).masks
    spacing = meta.get("pixel_spacing_mm") or [1.0, 1.0]
    return analyze(slice_img, spacing, meta.get("sequence", "unknown"), meta.get("doctor"),
                   meta.get("params", Params())).masks


def brain_slices(seg: Segmentation, min_frac: float = 0.25) -> np.ndarray:
    """'Brain present' gate: axial slices whose intracranial area is at least `min_frac` of the
    largest one (rejects near-blank and skull-base slices)."""
    area = seg.intracranial.reshape(seg.intracranial.shape[0], -1).sum(1)
    return np.where(area >= min_frac * max(area.max(), 1))[0]


def pick_slice(seg: Segmentation) -> int:
    """C3 auto slice: largest lesion area, else the middle of the brain-present slices."""
    if seg.slice_index is not None:
        return seg.slice_index
    return _auto_slice(seg.masks["lesion"], seg.intracranial)
