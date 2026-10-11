"""Run the classical segmenter on ReMIND patients and score it against the dataset's labels.

The segmenter runs fully automatically (no doctor labels given). Its output is then compared with:
  brain boundary (intracranial) vs ReMIND cerebrum SEG (+ tumor + ventricles, which BrainLab cuts out)
  csf                           vs ReMIND ventricles SEG (recall only: ventricles are just part of CSF)
  lesion (FLAIR/T2 only)        vs ReMIND tumor SEG
Writes contact sheets to data/out/seg/ (gitignored). Case names are anonymous (case01, ...).

Usage (from repo root):
  backend/.venv/Scripts/python.exe -I scripts/seg/run_on_remind.py --patients 1-5
  backend/.venv/Scripts/python.exe -I scripts/seg/run_on_remind.py --patients 5 --sequence FLAIR
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from cortexview import dicom_io as io  # noqa: E402
from cortexview.layers import load_layers  # noqa: E402
from cortexview.segment import analyze, pick_slice  # noqa: E402

SEQ_PREF = {"T1": ["T1_post", "T1_pre"], "FLAIR": ["FLAIR"], "T2": ["T2"]}


def dice(a, b):
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else 2 * (a & b).sum() / s


def striped_fraction(mask):
    """Share of interior slices whose area is under half of both neighbours' (a real brain label
    changes smoothly from slice to slice; a corrupted one alternates full and empty slices)."""
    area = mask.reshape(mask.shape[0], -1).sum(1).astype(float)
    zs = np.where(area > 0)[0]
    if len(zs) < 10:
        return 0.0
    a = area[zs[0]:zs[-1] + 1]
    inner = a[1:-1]
    return float(np.mean((inner < 0.5 * a[:-2]) & (inner < 0.5 * a[2:])))


def score(seg, ref, vox_ml):
    out = {}
    if "brain" in ref:
        truth = ref["brain"] | ref.get("lesion", False) | ref.get("ventricles", False)
        inside = (truth & seg.head).sum() / truth.sum()
        if inside < 0.97:
            # a real brain label lies inside the head; ReMIND-004's cerebrum SEG is sheared in the file
            out["reference_suspect"] = f"{100 * (1 - inside):.0f}% of the brain label lies outside the head"
            return out
        jumps = striped_fraction(truth)
        if jumps > 0.05:
            # some ReMIND cerebrum SEGs come out as stripes (slices missing in between), e.g. 010, 037
            out["reference_suspect"] = f"{100 * jumps:.0f}% of the brain label's slices are missing/striped"
            return out
        ic = seg.intracranial
        out["brain_dice"] = dice(ic, truth)
        out["brain_recall"] = (ic & truth).sum() / truth.sum()
        out["brain_precision"] = (ic & truth).sum() / max(ic.sum(), 1)
        out["brain_ml"] = (ic.sum() * vox_ml, truth.sum() * vox_ml)
    if "ventricles" in ref:
        v = ref["ventricles"]
        out["ventricle_recall_as_csf"] = (seg.masks["csf"] & v).sum() / max(v.sum(), 1)
    if "lesion" in ref and ref["lesion"].any():
        out["lesion_dice"] = dice(seg.masks["lesion"], ref["lesion"])
        out["lesion_ml"] = (seg.masks["lesion"].sum() * vox_ml, ref["lesion"].sum() * vox_ml)
    return out


# ---------------------------------------------------------------- drawing

COLORS = {l.id: l.rgb for l in load_layers()}


def to_rgb(gray):
    g = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return np.stack([g] * 3, -1)


def overlay(gray, masks, alpha=0.45):
    rgb = to_rgb(gray).astype(np.float32)
    for lid, m in masks.items():
        rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(COLORS[lid])
    return rgb.astype(np.uint8)


def outline(mask):
    from scipy import ndimage as ndi
    return mask & ~ndi.binary_erosion(mask)


def panel(gray, seg_masks, truth, aspect):
    """[MRI | layers + reference outline (yellow)] side by side, stretched to square mm."""
    left = to_rgb(gray)
    right = overlay(gray, seg_masks)
    if truth is not None:
        right[outline(truth)] = (255, 230, 0)
    img = Image.fromarray(np.concatenate([left, right], 1))
    h = max(1, round(img.height * aspect))
    return img.resize((img.width, h), Image.NEAREST)   # display only; data is never resampled


def norm_view(vol, head):
    lo, hi = np.percentile(vol[head], [1, 99])
    return np.clip((vol - lo) / (hi - lo), 0, 1)


def contact_sheet(name, vol, seg, truth, out_dir, metrics):
    g = norm_view(vol.data, seg.head)
    sp = vol.spacing
    z = np.where(seg.intracranial.any((1, 2)))[0]
    picks = [z[int(len(z) * f)] for f in (0.3, 0.55, 0.8)] if len(z) else [vol.shape[0] // 2]
    lesion_k = pick_slice(seg)
    if seg.masks["lesion"].any():
        picks[1] = lesion_k
    tiles = []
    for k in picks:
        tiles.append(panel(g[k], {l: m[k] for l, m in seg.masks.items()}, None if truth is None else truth[k], sp[1] / sp[2]))
    cy, cx = [int(c) for c in np.array(np.nonzero(seg.intracranial)).mean(1)[1:]]
    tiles.append(panel(g[:, cy][::-1], {l: m[:, cy][::-1] for l, m in seg.masks.items()},
                       None if truth is None else truth[:, cy][::-1], sp[0] / sp[2]))
    tiles.append(panel(g[:, :, cx][::-1], {l: m[:, :, cx][::-1] for l, m in seg.masks.items()},
                       None if truth is None else truth[:, :, cx][::-1], sp[0] / sp[1]))
    w = max(t.width for t in tiles)
    labels = [f"axial {k}" for k in picks] + ["coronal", "sagittal"]
    sheet = Image.new("RGB", (w, sum(t.height + 18 for t in tiles) + 40), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), f"{name}  {vol.sequence}  " + "  ".join(
        f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)), fill=(255, 255, 255))
    d.text((6, 20), "left: MRI   right: layers (tan skull, mauve GM, off-white WM, blue CSF, red lesion), "
                    "yellow = reference brain outline", fill=(180, 180, 180))
    y = 40
    for t, lab in zip(tiles, labels):
        d.text((6, y), lab, fill=(200, 200, 200))
        sheet.paste(t, (0, y + 16))
        y += t.height + 18
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(out_dir / f"{name}_{vol.sequence}.png")


# ---------------------------------------------------------------- main

def fmt(k, v):
    if isinstance(v, tuple):
        return f"{k}=({v[0]:.0f} vs {v[1]:.0f} ref)"
    if isinstance(v, str):
        return f"{k}: {v}"
    return f"{k}={float(v):.3f}"


def parse_patients(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load_case(folder, family, cache_dir):
    """(Volume, {role: mask on the volume's grid}) for one patient and sequence family.
    Series choice mirrors C3: the series a tumour label was drawn on (same coordinate frame) first, then
    series that can be shown axially, then more slices / finer voxels. References come only from SEG
    files in the same coordinate frame (patients can move between series)."""
    cache = cache_dir / f"{folder.parent.name}_{family}.npz" if cache_dir else None
    if cache and cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                from cortexview.volume import Volume
                vol = Volume(z["data"], z["affine"], sequence=str(z["sequence"]), sequence_source="header",
                             frame_of_reference=str(z["frame"]), acquired_plane=str(z["plane"]))
                return vol, {k[4:]: z[k] for k in z.files if k.startswith("ref_")}
        except Exception:                                # half-written by an interrupted run
            import gc
            gc.collect()                                 # np.load can leave the handle open on failure
            try:
                cache.unlink()
            except OSError:
                cache = None                             # still locked: load without caching
    series, seg_infos, _ = io.scan(sorted(folder.rglob("*.dcm")))
    fam = [s for s in series.values() if s.sequence in SEQ_PREF[family]]
    if not fam:
        return None, {}
    tumour_frames = {g.frame_of_reference for g in seg_infos if "lesion" in g.roles}
    info = sorted(fam, key=lambda s: (s.frame_of_reference not in tumour_frames, not s.axial_ok,
                                      -len(s.sources), float(np.prod(s.spacing_mm))))[0]
    vol = io.load_info(info)
    masks = [m for g in seg_infos if g.frame_of_reference == vol.frame_of_reference
             for m in io.load_seg(pydicom.dcmread(g.source))]
    ref, _ = io.doctor_labels_on(vol, masks)            # strict: same coordinate frame only
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, data=vol.data, affine=vol.affine, sequence=vol.sequence,
                            frame=vol.frame_of_reference or "", plane=vol.acquired_plane or "",
                            **{f"ref_{k}": v for k, v in ref.items()})
        os.replace(tmp, cache)
    return vol, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remind", default=str(ROOT / "data/archive/remind"))
    ap.add_argument("--out", default=str(ROOT / "data/out/seg"))
    ap.add_argument("--patients", default="1-5")
    ap.add_argument("--sequence", default="T1", choices=sorted(SEQ_PREF))
    ap.add_argument("--no-sheets", action="store_true", help="skip contact sheets")
    ap.add_argument("--check-determinism", action="store_true", help="run twice and compare masks")
    ap.add_argument("--cache", default=str(ROOT / "data/out/cache"),
                    help="where loaded volumes are cached (gitignored); '' to disable")
    ap.add_argument("--cache-only", action="store_true", help="only load + cache, no segmentation")
    args = ap.parse_args()
    cache_dir = Path(args.cache) if args.cache else None
    if args.cache_only:
        for pid in parse_patients(args.patients):
            folder = Path(args.remind) / f"ReMIND-{pid:03d}" / "Preop"
            if folder.exists():
                t = time.perf_counter()
                vol, ref = load_case(folder, args.sequence, cache_dir)
                print(f"case{pid:02d} {args.sequence}: {'-' if vol is None else vol.shape} refs={sorted(ref)} "
                      f"{time.perf_counter() - t:.1f}s", flush=True)
        return

    rows = []
    for pid in parse_patients(args.patients):
        folder = Path(args.remind) / f"ReMIND-{pid:03d}" / "Preop"
        name = f"case{pid:02d}"
        if not folder.exists():
            continue
        vol, ref = load_case(folder, args.sequence, cache_dir)
        if vol is None:
            continue

        t = time.perf_counter()
        seg = analyze(vol.data, vol.spacing, vol.sequence)          # fully automatic
        dt = time.perf_counter() - t
        det = ""
        if args.check_determinism:
            again = analyze(vol.data, vol.spacing, vol.sequence)
            det = f"  deterministic={all(np.array_equal(seg.masks[k], again.masks[k]) for k in seg.masks)}"

        vox_ml = float(np.prod(vol.spacing)) / 1000
        m = score(seg, ref, vox_ml)
        if vol.sequence != "FLAIR":
            m.pop("lesion_dice", None)                  # no automatic lesion on T1, nothing to score
        truth = None
        if "brain" in ref and "reference_suspect" not in m:
            truth = ref["brain"] | ref.get("lesion", False) | ref.get("ventricles", False)
        if not args.no_sheets:
            contact_sheet(name, vol, seg, truth, Path(args.out), m)
        vols_ml = {k: round(float(v.sum() * vox_ml)) for k, v in seg.masks.items()}
        print(f"{name} {vol.sequence} {vol.acquired_plane} {vol.shape} {np.round(vol.spacing, 2).tolist()}  "
              f"{dt:.1f}s{det}  refs={sorted(ref)}")
        print(f"   mL {vols_ml}  conf {seg.confidence}")
        print("   " + "  ".join(fmt(k, v) for k, v in m.items()) + (f"  warnings={seg.warnings}" if seg.warnings else ""))
        rows.append(m)

    print(f"--- {args.sequence}: {len(rows)} cases")
    for key in ("brain_dice", "brain_recall", "brain_precision", "ventricle_recall_as_csf", "lesion_dice"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            print(f"MEAN {key}: {np.mean(vals):.3f}  median {np.median(vals):.3f}  "
                  f"min {np.min(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
