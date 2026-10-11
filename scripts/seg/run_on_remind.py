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
import sys
import time
from pathlib import Path

import numpy as np
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


def score(seg, ref, vox_ml):
    out = {}
    if "brain" in ref:
        truth = ref["brain"] | ref.get("lesion", False) | ref.get("ventricles", False)
        inside = (truth & seg.head).sum() / truth.sum()
        if inside < 0.97:
            # a real brain label lies inside the head; ReMIND-004's cerebrum SEG is sheared in the file
            out["reference_suspect"] = f"{100 * (1 - inside):.0f}% of the brain label lies outside the head"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remind", default=str(ROOT / "data/archive/remind"))
    ap.add_argument("--out", default=str(ROOT / "data/out/seg"))
    ap.add_argument("--patients", default="1-5")
    ap.add_argument("--sequence", default="T1", choices=sorted(SEQ_PREF))
    args = ap.parse_args()

    rows = []
    for case_no, pid in enumerate(parse_patients(args.patients), 1):
        folder = Path(args.remind) / f"ReMIND-{pid:03d}" / "Preop"
        name = f"case{pid:02d}"
        if not folder.exists():
            print(f"{name}: not downloaded")
            continue
        series, segs = io.group_series(io.read_folder(folder))
        vols = [io.load_series(s) for s in series.values()]
        vols = [v for pref in SEQ_PREF[args.sequence] for v in vols if v.sequence == pref]
        if not vols:
            print(f"{name}: no {args.sequence} series")
            continue
        vol = vols[0]
        ref, warn = io.doctor_labels_on(vol, [m for s in segs for m in io.load_seg(s)], allow_frame_mismatch=True)

        t = time.perf_counter()
        seg = analyze(vol.data, vol.spacing, vol.sequence)          # fully automatic
        dt = time.perf_counter() - t
        again = analyze(vol.data, vol.spacing, vol.sequence)
        deterministic = all(np.array_equal(seg.masks[k], again.masks[k]) for k in seg.masks)

        vox_ml = float(np.prod(vol.spacing)) / 1000
        m = score(seg, ref, vox_ml)
        truth = None
        if "brain" in ref:
            truth = ref["brain"] | ref.get("lesion", False) | ref.get("ventricles", False)
        contact_sheet(name, vol, seg, truth, Path(args.out), m)
        vols_ml = {k: round(float(v.sum() * vox_ml), 1) for k, v in seg.masks.items()}
        print(f"{name} {vol.sequence} {vol.shape} {np.round(vol.spacing, 2).tolist()}  {dt:.2f}s  "
              f"deterministic={deterministic}")
        print(f"   layer volumes mL: {vols_ml}   confidence: {seg.confidence}")
        if seg.stats:
            print(f"   class separation d': GM/WM {seg.stats['gm_wm_separation']:.2f}  "
                  f"CSF {seg.stats['csf_separation']:.2f}")
        print("   " + "  ".join(fmt(k, v) for k, v in m.items()) + (f"  warnings={seg.warnings}" if seg.warnings else ""))
        rows.append(m)

    for key in ("brain_dice", "brain_recall", "brain_precision", "ventricle_recall_as_csf", "lesion_dice"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            print(f"MEAN {key}: {np.mean(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
