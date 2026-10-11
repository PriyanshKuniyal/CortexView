"""Classical (non-AI) brain MRI segmentation into the C1 layers.

Pure math on voxel intensities: thresholds (Otsu / multi-Otsu), distance transforms, connected
components and robust statistics. No randomness, no learned weights: the same input always gives the
same masks. Works on a 3D volume (preferred) or a single 2D slice; every size is in mm, so voxels that
are not cubes (e.g. 2.5 mm thick slices) are handled correctly.

Pipeline (head and brain boundary on a ~2 mm grid, the rest at <= ~1 mm):
  1. head mask      Otsu threshold -> close -> largest blob -> fill holes; flag speckle-noise
                    backgrounds (e.g. MP2RAGE) that defeat brightness-based head finding
  2. window         1st..99th percentile inside the head -> 0..1, then divide out the coil's slow
                    brightness drift (bias field)
  3. intracranial   deep core of "not dark" tissue (relative to brain measured where brain surely is)
                    -> grow back -> extend into the rest of the skull cavity (vault geometry, see
                    extend_to_vault). skull layer = head - intracranial
  4. lesion         doctor label if given; else, on FLAIR only, bright AND left/right-asymmetric
                    regions (hysteresis thresholding against a fitted mid-line mirror)
  5. CSF / GM / WM  CSF = past the halfway point between tissue and fluid brightness; GM vs WM =
                    local contrast + cortex geometry; confidence from the contrast measured in this scan
  6. exclusivity    lesion > csf > white_matter > gray_matter > skull

Contract C6: segment_volume(volume, meta) and segment(slice, meta) return {layer_id: bool mask}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image

from .layers import PRIORITY, layer_ids


@dataclass(frozen=True)
class Params:
    work_mm: float = 2.0            # head/brain masks are found on a coarse grid of about this voxel size
    smooth_mm: float = 0.5          # Gaussian denoise before thresholding
    head_close_mm: float = 3.0
    brain_erode_mm: float = 6.0     # breaks bridges between brain and scalp/eyes/neck (the cord is ~10 mm)
    brain_depth_mm: float = 15.0    # the brain's core lies at least this deep under the head surface;
                                    # scalp, fat and bone marrow (bright on T1) never do
    brain_close_mm: float = 6.0     # pulls sulci + thin outer CSF into the intracranial mask
    noise_jitter: float = 0.045     # median fine-scale jitter deep in the head above this = noisy background
    sure_brain_depth_mm: float = 25.0  # upper head, this deep under the skin: certainly brain
    tissue_ratio: float = 0.7       # brain tissue is brighter than this x typical brain (CSF ~0.3-0.55 on T1)
    bone_ratio: float = 0.2         # darker than this x deep-brain brightness = bone/air, never intracranial
    shell_quantile: float = 0.2     # scalp+skull thickness = this quantile of the brain's depth under the skin
    shell_margin_mm: float = 1.0
    vault_offset_mm: float = 25.0   # "vault" = slices above (brain centre - this); below it is the skull base
    dark_open_mm: float = 3.0       # added CSF/tumour blobs must be thicker than 2x this (cuts thin leaks)
    csf_rim_mm: float = 3.0         # outer CSF claimed just outside the brain
    fine_mm: float = 0.9            # finest grid worked on; finer scans (e.g. 0.47 mm) are block-averaged
    bias_mm: float = 30.0           # smoothness of the brightness-drift (bias field) estimate
    csf_k: float = 2.0              # the CSF cut is never closer to typical tissue than this many robust SDs
    local_mm: float = 6.0           # neighbourhood size for local GM/WM contrast
    local_gain: float = 1.5         # how sharply local contrast (in local SDs) votes GM vs WM
    cortex_mm: float = 4.0          # gray-matter depth scale for the cortex prior
    cortex_prior: float = 0.35      # geometry's share of the GM/WM decision on a high-contrast scan ...
    cortex_prior_max: float = 0.85  #   ... rising to this when the scan shows no GM/WM contrast
    posterior_smooth_mm: float = 1.2   # smoothing of the GM/WM score (removes speckle)
    lesion_seed_k: float = 3.0      # FLAIR lesion seeds: this many robust SDs brighter than typical brain
    lesion_asym_k: float = 2.0      #   ... and this much brighter than the mirror-image point
    lesion_grow_k: float = 2.0      # lesion grows from seeds through voxels this bright (hysteresis)
    lesion_grow_asym_k: float = 1.0 #   ... that are also at least this much brighter than their mirror
    lesion_smooth_mm: float = 1.0
    midline_min_corr: float = 0.3   # below this the left/right mirror fit is not trusted
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
    """Back to the full grid: block copy plus a half-block blur, i.e. smooth (near-linear) upsampling of
    a float field (a mask's 0/1 edges get smooth sub-block boundaries when thresholded at 0.5)."""
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


def intracranial_mask(n, head, spacing, p: Params = Params(), return_core: bool = False):
    """Brain + CSF (+ any tumour) inside the skull (run on the coarse grid).

    Step 1, conservative brain: the brain is a big blob deep inside the head. Scalp fat and bone marrow
    can be bright and touch it through thin bridges, so it is found as a deep, eroded core of "not dark"
    tissue and then grown back. This misses dark tumours/cysts and fluid at the brain's edge, and the
    top of the brain (which lies less than the depth limit under the skin).
    Step 2, extend_to_vault: add what lies inside the skull but outside step 1.
    With return_core, also returns the step-1 brain (used to keep lesion seeds inside real brain)."""
    s = _smooth(n, p.smooth_mm, spacing)
    depth = ndi.distance_transform_edt(head, sampling=spacing)
    # Where brain certainly is: deep under the skin in the upper half of the head (the neck and face are
    # below; the scan may include a lot of them). Its median brightness is typical brain tissue.
    sure = head & (depth > p.sure_brain_depth_mm)
    if head.ndim == 3:
        zs = np.where(head.any(axis=(1, 2)))[0]
        sure[: (zs[0] + zs[-1]) // 2] = False
    if sure.sum() < 50:
        sure = head & (depth >= np.quantile(depth[head], 0.9))
    # "Not dark" = brighter than tissue_ratio x typical brain: keeps grey and white matter, drops fluid,
    # bone and air. (A histogram split of the whole head is unreliable: bright scalp fat and neck can
    # push it up to brain level.)
    t_dark = p.tissue_ratio * float(np.median(s[sure]))
    tissue = head & (s > t_dark)
    lab, nlab = ndi.label(_erode(tissue, p.brain_erode_mm, spacing) & (depth > p.brain_depth_mm))
    if nlab == 0:
        out = tissue & sure
        return (out, out) if return_core else out
    # the brain core is the eroded blob that overlaps the sure-brain region most (not simply the
    # largest: with a long neck in the field of view, neck muscle can be bigger)
    overlap = ndi.sum_labels(sure, lab, index=np.arange(1, nlab + 1))
    core = lab == int(np.argmax(overlap)) + 1
    brain = _largest(_dilate(core, p.brain_erode_mm + 1.5, spacing) & tissue)
    brain = _fill(_close(brain, p.brain_close_mm, spacing)) & head
    if brain.sum() < 50:
        return (brain, brain) if return_core else brain
    ic = extend_to_vault(brain, s, head, depth, t_dark, spacing, p)
    return (ic, brain) if return_core else ic


def _hull_per_slice(mask):
    out = np.zeros_like(mask)
    for k in range(mask.shape[0]):
        if mask[k].sum() > 10:
            out[k] = convex_hull_image(mask[k])
    return out


def extend_to_vault(brain, s, head, depth, t_dark, spacing, p: Params = Params()):
    """Grow the conservative brain into the rest of the cranial cavity.

    The skull is too thin and too varied in brightness to trace directly (on T1 it is a 1-3 mm dark
    line followed by bright marrow and fat, sometimes no dark line at all). What is stable is geometry:
    over the top of the head the inner skull lies at a fairly constant depth under the skin, measured
    here from where the brain itself starts. So:
      vault (upper head): add any voxel deeper than that shell that is not bone/air dark
                          -> recovers the top of the brain, outer CSF and tumours touching the skull
      skull base (lower): only CSF/tumour-dark voxels inside the brain's own outline on that slice;
                          sinuses, muscles and orbits below the brain stay out
    Added regions must be thicker than 2 x dark_open_mm and touch the brain."""
    inside = ndi.distance_transform_edt(brain, sampling=spacing)
    deep = brain & (inside > 10)
    med = float(np.median(s[deep if deep.sum() > 50 else brain]))
    nonbone = s > p.bone_ratio * med
    if brain.ndim == 3:
        edge = brain & ~ndi.binary_erosion(brain)
        zs = np.where(brain.any(axis=(1, 2)))[0]
        upper_half = np.zeros_like(brain)
        upper_half[zs[len(zs) // 3]:] = True
        ref = edge & upper_half
        shell = float(np.quantile(depth[ref if ref.any() else edge], p.shell_quantile))
        vault = depth > shell - p.shell_margin_mm
        zc = int(round(np.nonzero(brain)[0].mean() - p.vault_offset_mm / spacing[0]))
        upper = np.zeros_like(brain)
        upper[max(zc, 0):] = True
        cand = vault & ~brain & nonbone & (upper | (_hull_per_slice(brain) & (s < t_dark)))
    else:
        # a lone 2D slice (small uploads: usually a few slices around the area of interest) gets the
        # vault rule; its shell depth comes from the slice's own brain edge
        shell = float(np.quantile(depth[brain & ~ndi.binary_erosion(brain)], p.shell_quantile))
        cand = (depth > shell - p.shell_margin_mm) & ~brain & nonbone
    cand = _dilate(_erode(cand, p.dark_open_mm, spacing), p.dark_open_mm, spacing) & cand
    lab, nlab = ndi.label(cand)
    if nlab:
        touching = np.unique(lab[_dilate(brain, 1.01 * max(spacing), spacing) & (lab > 0)])
        brain = brain | np.isin(lab, touching)
    ic = _fill(_close(brain, p.brain_close_mm, spacing)) & head
    # outer CSF: a thin fluid layer separates brain and skull. Claim fluid-dark (not bone-dark) voxels
    # just outside; on FLAIR fluid is as dark as bone, so this adds little there.
    rim = _dilate(ic, p.csf_rim_mm, spacing) & ~ic & head & nonbone & (s < t_dark)
    return ic | rim


def _mirror_coords(shape, spacing, normal, offset):
    """Voxel index coordinates of every voxel's reflection across the plane normal.x = offset (x in mm)."""
    sp = np.asarray(spacing, float)
    idx = np.indices(shape, dtype=np.float32).reshape(len(shape), -1)
    x = idx * sp[:, None]
    dist = normal @ x - offset
    return ((x - 2 * normal[:, None] * dist) / sp[:, None]).reshape((len(shape),) + tuple(shape))


def _rotate(v, axis, angle):
    """Rodrigues rotation of vector v about unit `axis` (3D) or a plain rotation (2D)."""
    if len(v) == 2:
        c, s_ = np.cos(angle), np.sin(angle)
        return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])
    k = axis / np.linalg.norm(axis)
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * (k @ v) * (1 - np.cos(angle))


def midline_plane(s, ic, spacing):
    """Fit the brain's mid-sagittal (left/right mirror) plane: start from the brain's principal axis
    closest to the image's left-right axis (last array axis), then refine tilt and position by
    maximising the correlation between the image and its own reflection (coordinate search, fixed steps:
    deterministic). Returns (unit normal, offset in mm, correlation)."""
    sp = np.asarray(spacing, float)
    pts = np.argwhere(ic) * sp
    if len(pts) < 100:
        return None
    centre = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    k = int(np.argmax(np.abs(vecs[-1, :])))
    normal = vecs[:, k] * np.sign(vecs[-1, k] or 1.0)
    others = [vecs[:, j] for j in range(len(sp)) if j != k]
    sub = ic.astype(np.float32)
    idx = np.argwhere(ic)[::3].astype(np.float64)             # every 3rd brain voxel (fixed: deterministic)
    x = idx * sp
    v = s[tuple(idx.astype(int).T)]

    def score(nrm, off):
        coords = ((x - 2 * np.outer(x @ nrm - off, nrm)) / sp).T
        both = ndi.map_coordinates(sub, coords, order=0, mode="constant") > 0.5
        if both.sum() < 100:
            return -1.0
        m = ndi.map_coordinates(s, coords[:, both], order=1, mode="nearest")
        return float(np.corrcoef(v[both], m)[0, 1])

    offset = float(normal @ centre)
    best = score(normal, offset)
    for step_deg, step_mm in ((4.0, 4.0), (2.0, 2.0), (1.0, 1.0)):
        improved = True
        while improved:
            improved = False
            moves = [(normal, offset + d) for d in (-step_mm, step_mm)]
            for ax in (others if len(sp) == 3 else [None]):
                for a in (-step_deg, step_deg):
                    nrm = _rotate(normal, ax, np.deg2rad(a))
                    moves.append((nrm / np.linalg.norm(nrm), offset))
            for nrm, off in moves:
                sc = score(nrm, off)
                if sc > best + 1e-4:
                    best, normal, offset, improved = sc, nrm, off, True
    return normal, offset, best


def auto_lesion(n, ic, inner, spacing, sequence, p: Params = Params(), seed_region=None):
    """Abnormal bright tissue on FLAIR (fluid is dark there, swelling and most gliomas bright).

    z = brightness above typical brain, in robust SDs. Asymmetry = brighter than the mirror-image point
    across the brain's midline, in the same units (normal tissue that is FLAIR-bright, like the
    hippocampi or the cerebellum on some scanners, is bright on both sides; most tumours are one-sided).
      seeds:   z > lesion_seed_k  and  asymmetry > lesion_asym_k
      lesion:  connected voxels with z > lesion_grow_k that contain a seed  (hysteresis thresholding)
    then minimum-size and skull-rim filters. `inner` = intracranial voxels away from the skull rim.
    `seed_region`: where seeds may start (the conservative brain: keeps orbit fat and skull-base tissue,
    which can sit inside a generous brain boundary, from starting a lesion).

    Not done on T1 (lesions can be dark, bright or both) or T2 (fluid is the brightest thing on T2 and
    swelling overlaps it, so brightness outliers are mostly CSF edges: tested on ReMIND and the phantom).
    Returns (mask, warning or None)."""
    if sequence != "FLAIR" or inner is None:
        return np.zeros_like(ic), "lesion_auto_needs_flair"
    s = _smooth(n, p.lesion_smooth_mm, spacing)
    vals = s[inner]
    if vals.size < 50:
        return np.zeros_like(ic), None
    med = float(np.median(vals))
    sd = _spread(vals, med)
    z = (s - med) / sd

    # asymmetry, on the ~2 mm grid (lesions are much bigger than that)
    f = _factors(spacing, p.work_mm)
    cs = _down(s, f)
    cic = _down(ic.astype(np.float32), f) > 0.5
    csp = tuple(a * b for a, b in zip(spacing, f))
    plane = midline_plane(cs, cic, csp)
    if plane is not None and plane[2] > p.midline_min_corr:
        mirrored = ndi.map_coordinates(cs, _mirror_coords(cs.shape, csp, plane[0], plane[1]),
                                       order=1, mode="nearest")
        asym = _up((cs - mirrored) / sd, f, s.shape)
        seeds = ic & (z > p.lesion_seed_k) & (asym > p.lesion_asym_k)
        # growth must stay one-sided too, or it runs along symmetric bright tissue (e.g. FLAIR cortex)
        grow = ic & (z > p.lesion_grow_k) & (asym > p.lesion_grow_asym_k)
        warning = None
    else:                                    # no trustworthy midline (odd field of view): stricter seeds
        seeds = ic & (z > p.lesion_seed_k + 1.0)
        grow = ic & (z > p.lesion_grow_k + 1.0)
        warning = "lesion_no_midline"
    if seed_region is not None:
        seeds &= seed_region
    lab, nlab = ndi.label(grow)
    if nlab == 0 or not seeds.any():
        return np.zeros_like(ic), warning
    seeded = np.unique(lab[seeds])
    seeded = seeded[seeded > 0]
    voxel = float(np.prod(spacing))
    sizes = ndi.sum_labels(np.ones_like(grow), lab, index=seeded) * voxel
    inside = ndi.sum_labels(inner, lab, index=seeded) * voxel
    min_size = p.lesion_min_ml * 1000 if n.ndim == 3 else p.lesion_min_mm2
    keep = seeded[(sizes >= min_size) & (inside >= 0.5 * sizes)]
    lesion = np.isin(lab, keep)
    return _fill(lesion) & ic, warning


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


def _spread(vals, med):
    """Robust SD: the larger of MAD*1.4826 and IQR/1.349. MAD alone only sees the dominant tissue
    (it would put the CSF cut right at gray matter when GM and WM separate cleanly); the IQR spans both."""
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    return max(float(np.median(np.abs(vals - med))) * 1.4826, float(q3 - q1) / 1.349, 1e-6)


def tissue_level(n, ic, exclude, sequence, spacing, p: Params = Params()):
    """Whole-brain numbers for the tissue split: (tissue median, robust SD, CSF threshold, GM/WM contrast).

    CSF cut: halfway between typical tissue and typical fluid brightness (fluid = median of the most
    fluid-like 5% of brain voxels: darkest on T1/FLAIR, brightest on T2), but never closer to tissue than
    csf_k robust SDs, so scans with little visible fluid do not lose tissue to it.

    GM/WM contrast (d'): is the cortex (a band 1.5-4 mm from CSF) actually darker (T1) / brighter (T2,
    FLAIR) than deep white matter, in robust SDs? Wrong direction counts as none. Measured on slices
    ~4 mm apart over the whole brain, with in-plane distances, so it is cheap and identical whether the
    caller classifies the whole volume or one slab."""
    sm = _smooth(n, p.smooth_mm, spacing)
    region = ic & ~exclude
    vals = sm[region]
    med = float(np.median(vals))
    sd = _spread(vals, med)
    order = CLASS_ORDER.get(sequence, CLASS_ORDER["unknown"])
    dark = order[0] == "csf"
    if dark:
        fluid = float(np.median(vals[vals <= np.quantile(vals, 0.05)]))
        t = min((med + fluid) / 2, med - p.csf_k * sd)
    else:
        fluid = float(np.median(vals[vals >= np.quantile(vals, 0.95)]))
        t = max((med + fluid) / 2, med + p.csf_k * sd)

    gm_darker = order.index("gray_matter") < order.index("white_matter")
    if sm.ndim == 3:
        ks = np.where(region.any(axis=(1, 2)))[0][:: max(1, int(round(4.0 / spacing[0])))]
    band_v, deep_v = [], []
    for k in ([None] if sm.ndim == 2 else ks):
        s2, r2, ic2 = (sm, region, ic) if k is None else (sm[k], region[k], ic[k])
        csf2 = r2 & ((s2 < t) if dark else (s2 > t))
        tissue2 = r2 & ~csf2
        d2 = ndi.distance_transform_edt(~(csf2 | ~ic2), sampling=spacing[-2:])
        band_v.append(s2[tissue2 & (d2 > 1.5) & (d2 < 4.0)])
        deep_v.append(s2[tissue2 & (d2 > 12.0)])
    vb, vd = np.concatenate(band_v), np.concatenate(deep_v)
    sep = 0.0
    if vb.size > 50 and vd.size > 50:
        mb, md = float(np.median(vb)), float(np.median(vd))
        pooled = np.sqrt((_spread(vb, mb) ** 2 + _spread(vd, md) ** 2) / 2)
        sep = float((md - mb if gm_darker else mb - md) / max(pooled, 1e-6))
    return med, sd, t, sep


def tissue_classes(n, ic, exclude, spacing, sequence, p: Params = Params(), level=None):
    """Split the intracranial mask into CSF / GM / WM.

    1. CSF = voxels past the fluid cut from tissue_level: darker on T1/FLAIR, brighter on T2.
    2. GM vs WM by *local* contrast: cortex is darker than the white matter right next to it on T1
       (brighter on T2/FLAIR), even on scans where global brightness of GM and WM overlaps.
    3. Cortex prior: gray matter lies along the brain surface and CSF-filled folds, so voxels near
       CSF lean GM and deep voxels lean WM. Its weight grows as the scan's measured contrast shrinks.
    4. Lightly smooth the combined score (neighbours tend to agree) and threshold at 0.5.
    `level` = tissue_level() of the whole brain, when only a slab of slices is passed in.
    Returns ({class: mask}, stats)."""
    region = ic & ~exclude
    out = {k: np.zeros_like(ic) for k in ("csf", "gray_matter", "white_matter")}
    if region.sum() < 50:
        return out, {}
    order = CLASS_ORDER.get(sequence, CLASS_ORDER["unknown"])
    s = _smooth(n, p.smooth_mm, spacing)
    med, mad, t_csf, sep = level or tissue_level(n, ic, exclude, sequence, spacing, p)
    csf_dark = order[0] == "csf"
    csf = region & ((s < t_csf) if csf_dark else (s > t_csf))
    tissue = region & ~csf

    gm_darker = order.index("gray_matter") < order.index("white_matter")
    z = _local_z(s, tissue, p.local_mm, spacing)
    z_gm = -z if gm_darker else z                       # + means "looks like gray matter"
    intensity = 1 / (1 + np.exp(np.clip(-p.local_gain * z_gm, -50, 50)))

    d_csf = ndi.distance_transform_edt(~(csf | ~ic), sampling=spacing)
    geometry = np.exp(-d_csf / p.cortex_mm)

    # the weaker this scan's measured GM/WM contrast, the more the cortex geometry decides
    # (noisy local contrast otherwise scatters grey-matter blotches through the white matter)
    a = p.cortex_prior + (p.cortex_prior_max - p.cortex_prior) * float(np.clip(1 - sep / 2.0, 0, 1))
    score = (1 - a) * intensity + a * geometry
    sig = [p.posterior_smooth_mm / sp for sp in spacing]
    gm = ndi.gaussian_filter(score.astype(np.float32), sig) > 0.5

    out["csf"] = csf
    out["gray_matter"] = tissue & gm
    out["white_matter"] = tissue & ~gm
    stats = {
        "tissue_median": med, "tissue_mad": mad, "csf_threshold": float(t_csf),
        "gm_wm_separation": sep,     # d': cortex band vs deep WM brightness (whole brain)
        "geometry_weight": a,        # share of the GM/WM decision made by cortex geometry
        "csf_separation": float(abs(np.median(s[csf]) - med) / max(mad, 1e-6)) if csf.any() else 0.0,
    }
    return out, stats


def _grade(d, cuts):
    return "low" if d < cuts[0] else ("medium" if d < cuts[1] else "high")


# ---------------------------------------------------------------- top level

def analyze(img: np.ndarray, spacing, sequence: str = "unknown",
            doctor: dict[str, np.ndarray] | None = None, p: Params = Params(),
            tissue_slices=None, lesion_hint: np.ndarray | None = None) -> Segmentation:
    """Segment a 2D slice or 3D volume. `doctor` may hold bool masks on the same grid:
    'lesion', 'ventricles', 'brain' (cerebrum). Masks come back on the input grid.

    Head, brain boundary and lesion are always 3D over the whole volume. The CSF/GM/WM split is the
    slow part, so `tissue_slices` can limit it to some axial slices (3D only):
      None        every slice (offline export, volumes)
      "auto"      the C3 auto slice (largest lesion area, else mid-brain), as for the API
      int         that slice;   (start, stop)  that range
    Other slices then have no CSF/GM/WM labels (see Segmentation.tissue_range).

    `lesion_hint`: an automatic lesion mask found on another series of the same scan (e.g. FLAIR) and
    mapped onto this grid. Used instead of this series' own lesion step when there is no doctor label;
    the layer stays source "auto"."""
    img = np.asarray(img, dtype=np.float32)
    spacing = tuple(float(s) for s in (spacing if spacing is not None else [1.0] * img.ndim))
    doctor = {k: np.asarray(v, dtype=bool) for k, v in (doctor or {}).items() if v is not None}
    if sequence not in CLASS_ORDER:
        sequence = "unknown"
    ff = _factors(spacing, p.fine_mm)
    if all(f == 1 for f in ff):
        return _analyze(img, spacing, sequence, doctor, p, tissue_slices, lesion_hint)

    # very fine scans: work on a ~1 mm grid, then copy labels back to every original pixel
    fine = _down(img, ff)
    fsp = tuple(s * f for s, f in zip(spacing, ff))
    ts = tissue_slices
    if img.ndim == 3 and ff[0] > 1 and ts is not None and ts != "auto":
        ts = (ts // ff[0]) if isinstance(ts, (int, np.integer)) else (ts[0] // ff[0], -(-ts[1] // ff[0]))
    hint = None if lesion_hint is None else _down(np.asarray(lesion_hint, np.float32), ff) > 0.5
    seg = _analyze(fine, fsp, sequence, {k: _down(v.astype(np.float32), ff) > 0.5 for k, v in doctor.items()},
                   p, ts, hint)
    rep = lambda m: _repeat(m, ff, img.shape)
    head, ic = rep(seg.head), rep(seg.intracranial)
    raw = {k: rep(m) for k, m in seg.masks.items()}
    if "lesion" in doctor:                                    # doctor labels stay pixel-exact
        raw["lesion"] = doctor["lesion"] & head
    elif lesion_hint is not None:
        raw["lesion"] = np.asarray(lesion_hint, bool) & head
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


def _analyze(img, spacing, sequence, doctor, p: Params, tissue_slices=None, lesion_hint=None) -> Segmentation:
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
    # Noisy background check: some sequences (e.g. MP2RAGE "UNI" images) fill air with bright speckle
    # noise, and brightness-based head finding then takes in the whole field of view. Detected from the
    # fine-scale voxel-to-voxel jitter deep inside the "head" (tissue is smooth, noise is not).
    jitter = _down(np.abs(img - _smooth(img, 0.7, spacing)), f) / max(float(np.percentile(img[head][::7], 99)), 1e-6)
    deep_head = ndi.distance_transform_edt(head_c, sampling=cspacing) > 20
    noisy = deep_head.any() and float(np.median(jitter[deep_head])) > p.noise_jitter
    if noisy:
        warnings.append("noisy_background")
    lo, hi = np.percentile(img[head][::7], [1, 99])          # fixed stride: deterministic subsample
    scale = lambda x: np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    n_c = scale(coarse)

    ic_c, core_c = intracranial_mask(n_c, head_c, cspacing, p, return_core=True)
    ic = (_up(ic_c, f, img.shape) > 0.5) & head
    # brain boundary from a doctor label: a "cerebrum" segmentation (e.g. BrainLab's) leaves out the
    # cerebellum, brainstem and outer CSF, and cuts out the tumour and ventricles. Used alone it would
    # paint the cerebellum as skull, so it is combined with the automatic boundary.
    if "brain" in doctor and doctor["brain"].any():
        doc = _fill(doctor["brain"] | doctor.get("lesion", False) | doctor.get("ventricles", False)) & head
        added = (ic & ~doc).sum() / max(doc.sum(), 1)
        ic = _fill(ic | doc) & head
        ic_c = _down(ic.astype(np.float32), f) > 0.5
        source["skull"] = "doctor"
        if added > 0.05:
            warnings.append("brain_label_extended")     # cerebellum/brainstem/outer CSF came from auto
    # remove the coil's slow brightness drift so one threshold fits the whole brain
    n = scale(img) / _up(bias_field(n_c, ic_c, cspacing, p), f, img.shape)

    if "lesion" in doctor:
        lesion = doctor["lesion"] & head
        ic |= lesion
        source["lesion"] = "doctor"
    elif lesion_hint is not None:
        lesion = np.asarray(lesion_hint, bool) & head
        ic |= lesion
        warnings.append("lesion_from_other_series")
    else:
        inner = None
        if sequence == "FLAIR":
            rim = ndi.distance_transform_edt(ic_c, sampling=cspacing)
            inner = ic & (_up(rim, f, img.shape) > p.lesion_rim_mm)
            # seeds only inside the conservative brain (grown 3 mm), never in orbit/skull-base tissue
            core = _up(_dilate(core_c, 3.0, cspacing), f, img.shape) > 0.5
        lesion, w = auto_lesion(n, ic, inner, spacing, sequence, p,
                                seed_region=core if sequence == "FLAIR" else None)
        if w:
            warnings.append(w)

    # whole-brain levels for the CSF cut (computed once, so slab and full runs agree exactly)
    level = tissue_level(n, ic, lesion, sequence, spacing, p)
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
        fit_csf = _grade(stats["csf_separation"], (3.0, 5.0))   # robust SDs between CSF and tissue
        for lid, fit in (("gray_matter", fit_gw), ("white_matter", fit_gw), ("csf", fit_csf)):
            conf[lid] = min(conf[lid], fit, key=rank.get)
        if fit_gw == "low":
            warnings.append("gm_wm_low_contrast")
    if source["lesion"] == "doctor":
        conf["lesion"] = "high"
    elif lesion_hint is not None:
        conf["lesion"] = CONFIDENCE["FLAIR"]["lesion"]
    if source["skull"] == "doctor":
        conf["skull"] = "medium" if "brain_label_extended" in warnings else "high"
    if noisy:                                   # nothing automatic can be trusted on such images
        conf = {lid: ("high" if source[lid] == "doctor" and lid == "lesion" else "low") for lid in conf}
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
