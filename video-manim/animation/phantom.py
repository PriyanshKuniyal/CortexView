"""NumPy port of the frontend's synthetic phantom (frontend/src/cortexview/demo/phantom.ts).

The CortexView film scenes show a synthetic axial slice generated in the browser: `phantom(440, 0,
mkRnd(7))` with 0.56 mm per pixel, and metrics computed from its label map ("math on masks"). This module
reproduces the same label map, the same greyscale "MRI", and the same metrics, so the video shows the
numbers the app shows. It is a phantom: geometry plus noise from a fixed seed. It is never a patient.

Kept deliberately close to the TypeScript, branch for branch, so the two can be compared
(see tests/test_phantom_port.py, which runs the original .ts with Node and diffs the outputs).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAYERS_JSON = REPO / "shared" / "layers.json"

N_MAIN = 440        # buildPhantomSlice(): N
SP_MAIN = 0.56      # buildPhantomSlice(): mm per pixel (square pixels)
SEED_MAIN = 7       # buildPhantomSlice(): mkRnd(7)


def load_layers() -> list[dict]:
    """shared/layers.json, sorted by peel order (outside in), exactly like frontend/src/cortexview/layers.ts."""
    reg = json.loads(LAYERS_JSON.read_text(encoding="utf-8"))
    return sorted(reg["layers"], key=lambda layer: layer["peel_order"])


def lcg(seed: int, count: int) -> np.ndarray:
    """mkRnd(seed) from phantom.ts, `count` draws: s = (s*1664525 + 1013904223) >>> 0; return s / 2^32."""
    out = np.empty(count, np.float64)
    s = seed
    for k in range(count):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[k] = s / 4294967296
    return out


def _ventricle_points():
    pts = []
    for s in (-1, 1):
        t = 0.0
        while t <= 1.0001:            # same float accumulation as the JS loop
            pts.append((s * (0.045 + 0.12 * t * t), -0.3 + 0.6 * t, 0.028 + 0.024 * np.sin(np.pi * t)))
            t += 1 / 36
    return pts


VENT = _ventricle_points()


def phantom(n: int, z: float, seed: int):
    """Label map (uint8, n x n) and 0-1 greyscale image (float32), as phantom(n, z, mkRnd(seed))."""
    h = n / 2
    sc = max(0.12, np.sqrt(max(0.0, 1 - z * z)))
    v_on = abs(z) < 0.42
    les_s = np.sqrt(max(0.0, 1 - (z / 0.22) ** 2))

    j, i = np.indices((n, n), dtype=np.float64)
    x = (i - h + 0.5) / h / sc
    y = (j - h + 0.5) / h / sc
    ax = 0.74 * (1 + 0.05 * y)
    nx, ny = x / ax, y / 0.9
    r = np.hypot(nx, ny)
    th = np.arctan2(ny, nx)

    lab = np.zeros((n, n), np.uint8)
    m = np.zeros((n, n), np.float64)
    inside = r < 1
    rb = 0.86 + 0.006 * np.sin(31 * th) + 0.004 * np.sin(17 * th + 1)

    # skull / scalp: banded brightness (fat, bone, marrow) through the shell
    shell = inside & (r >= rb)
    q = (r - rb) / (1 - rb)
    m_shell = np.where(q < 0.12, 0.14,
                       np.where(q < 0.52, np.where(np.abs(q - 0.32) < 0.06, 0.48, 0.06),
                                0.74 * (1 - (0.5 * np.maximum(0, q - 0.85)) / 0.15)))
    lab[shell] = 1
    m[shell] = m_shell[shell]

    # brain: gray matter, folded white-matter core, outer CSF rim, sulci, midline, ventricles, lesion
    brain = inside & ~shell
    ph = 10 * th + 0.7 * np.sin(3 * th) + 0.4 * np.sin(5 * th + 2)
    sn = np.abs(np.sin(ph))
    wm = 0.5 + 0.22 * np.power(sn, 0.55) - 0.08 * np.exp(-(x * x) / 0.004) * (np.abs(y) > 0.35)
    bl = np.full((n, n), 2, np.uint8)
    bm = np.full((n, n), 0.42)
    is_wm = r < wm
    bl[is_wm], bm[is_wm] = 3, 0.68
    rim = r > rb - 0.024
    sulc = ~rim & (bl == 2) & (sn < 0.075)
    bl[rim], bm[rim] = 4, 0.08
    bl[sulc], bm[sulc] = 4, 0.1
    mid = (np.abs(x) < 0.012) & (np.abs(y) > 0.36)
    bl[mid], bm[mid] = 4, 0.08
    if v_on:
        box = (np.abs(x) < 0.26) & (np.abs(y) < 0.42)
        hit = np.zeros((n, n), bool)
        for px, py, w in VENT:
            hit |= np.hypot(x - px, y - py) < w
        vent = box & hit
        bl[vent], bm[vent] = 4, 0.07
    if les_s > 0:
        lx, ly = x - 0.31, y + 0.3
        lr, la = np.hypot(lx, ly), np.arctan2(ly, lx)
        big_r = 0.12 * les_s * (1 + 0.13 * np.sin(3 * la + 1) + 0.07 * np.sin(5 * la))
        les = lr < big_r
        halo = ~les & (lr < big_r * 1.6) & ((bl == 2) | (bl == 3))
        qq = lr / big_r
        bm = np.where(les, np.where(qq > 0.72, 0.95, 0.24 + 0.1 * qq), bm)
        bl[les] = 10
        bm[halo] *= 0.82
    lab[brain] = bl[brain]
    m[brain] = bm[brain]

    # one random draw per pixel, row-major, as in the JS loop
    rnd = lcg(seed, n * n).reshape(n, n)
    mri = np.where(lab > 0, np.maximum(0, m + (rnd - 0.5) * 0.07), rnd * 0.02).astype(np.float32)
    # 5-point smoothing of the interior, reading from a copy. JS stores float32 but does the arithmetic in
    # float64, so do the same.
    t = mri.astype(np.float64)
    mri[1:-1, 1:-1] = ((t[1:-1, 1:-1] * 4 + t[1:-1, :-2] + t[1:-1, 2:] + t[:-2, 1:-1] + t[2:, 1:-1]) / 8) \
        .astype(np.float32)
    return lab, mri


def chamfer(lbl: np.ndarray) -> np.ndarray:
    """Distance to the nearest label boundary, the same two-pass chamfer as phantom.ts (float32)."""
    n = lbl.shape[0]
    L = lbl
    edge = np.zeros_like(L, bool)
    edge[:, 1:] |= L[:, 1:] != L[:, :-1]
    edge[:, :-1] |= L[:, :-1] != L[:, 1:]
    edge[1:, :] |= L[1:, :] != L[:-1, :]
    edge[:-1, :] |= L[:-1, :] != L[1:, :]
    D = np.where(edge, 0.0, 1e9).astype(np.float32)
    d = D.tolist()
    f = lambda v: float(np.float32(v))   # keep float32 rounding like the JS Float32Array
    for jj in range(1, n):
        row, up = d[jj], d[jj - 1]
        for ii in range(1, n - 1):
            row[ii] = min(row[ii], f(row[ii - 1] + 1), f(up[ii] + 1), f(up[ii - 1] + 1.414), f(up[ii + 1] + 1.414))
    for jj in range(n - 2, -1, -1):
        row, dn = d[jj], d[jj + 1]
        for ii in range(n - 2, 0, -1):
            row[ii] = min(row[ii], f(row[ii + 1] + 1), f(dn[ii] + 1), f(dn[ii + 1] + 1.414), f(dn[ii - 1] + 1.414))
    return np.asarray(d, np.float32)


def grey(v):
    """phantom.ts grey(): display mapping of the 0-1 image."""
    return np.minimum(255, np.power(v, 0.9) * 270)


@dataclass
class LayerMetrics:
    px: int
    mm2: float
    comps: int
    cx: float
    cy: float
    anchor_px: tuple[int, int]
    diam: float | None
    pct: float | None


def _components(mask: np.ndarray) -> int:
    from scipy import ndimage as ndi        # 4-connectivity, as the JS flood fill
    return int(ndi.label(mask)[1])


def measure(lbl: np.ndarray, D: np.ndarray, sp: float, layers: list[dict]) -> dict[str, LayerMetrics]:
    """Metrics as buildPhantomSlice() computes them: area = N * sp^2, % of intracranial area,
    largest boundary-to-boundary distance (lesion only), anchor = deepest point inside the mask."""
    out: dict[str, LayerMetrics] = {}
    intracranial = 0
    for L in layers:
        mask = lbl == L["label"]
        n_px = int(mask.sum())
        jj, ii = np.nonzero(mask)
        k = int(np.argmax(np.where(mask, D, -1).ravel()))       # first maximum in raster order, as JS
        diam = None
        if L["id"] == "lesion":
            pts = np.argwhere(mask & (D == 0))[:, ::-1].astype(np.float64)   # (i, j)
            if len(pts):
                d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
                diam = float(np.sqrt(d2.max()) * sp)
        out[L["id"]] = LayerMetrics(n_px, n_px * sp * sp, _components(mask), float(ii.mean()), float(jj.mean()),
                                    (k % lbl.shape[1], k // lbl.shape[1]), diam, None)
        if L["id"] != "skull":
            intracranial += n_px
    for L in layers:
        if L["id"] != "skull":
            out[L["id"]].pct = out[L["id"]].px / intracranial * 100
    out["_intracranial_px"] = intracranial  # type: ignore[assignment]
    return out


def lesion_boundary_pair(lbl: np.ndarray, D: np.ndarray):
    """The two lesion boundary pixels (i, j) that are farthest apart: the ends of the max diameter."""
    pts = np.argwhere((lbl == 10) & (D == 0))[:, ::-1]
    d2 = ((pts[:, None, :].astype(float) - pts[None, :, :]) ** 2).sum(-1)
    a, b = np.unravel_index(int(np.argmax(d2)), d2.shape)
    return tuple(pts[a]), tuple(pts[b])


@lru_cache(maxsize=None)
def main_slice():
    """The film's demo slice: labels, image, chamfer distance and metrics (cached per process)."""
    layers = load_layers()
    lbl, mri = phantom(N_MAIN, 0.0, SEED_MAIN)
    D = chamfer(lbl)
    return {"lbl": lbl, "mri": mri, "D": D, "sp": SP_MAIN, "layers": layers,
            "metrics": measure(lbl, D, SP_MAIN, layers)}


@lru_cache(maxsize=None)
def coarse_slice(n: int, seed: int = SEED_MAIN):
    """The same phantom function sampled on a coarse n x n grid, for the pixel-grid shots."""
    return phantom(n, 0.0, seed)


if __name__ == "__main__":
    s = main_slice()
    for key, v in s["metrics"].items():
        print(key, v)
