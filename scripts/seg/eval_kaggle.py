"""FLAIR lesion benchmark on the Kaggle LGG set (Buda/Mazurowski, TCGA; CC BY-NC-SA 4.0).

Evaluation only (contract C6: never train on it). The set has no headers, so voxel size is estimated:
  in-plane  head width ~150 mm  ->  mm per pixel
  slices    the top of the head is roughly a dome of the same radius: the number of slices from the
            widest one to the crown gives the slice gap (clipped to 1.5-8 mm)
Each patient's slices are stacked into a volume (slice 1 = lowest) and segmented as FLAIR.
Scores: per-patient Dice of the lesion (3D) and per-slice detection (lesion present or not).

Usage (from repo root):
  backend/.venv/Scripts/python.exe -I scripts/seg/eval_kaggle.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from cortexview.segment import analyze  # noqa: E402

HEAD_WIDTH_MM = 150.0


def load_patient(folder: Path):
    imgs = [p for p in folder.glob("*.tif") if not p.name.endswith("_mask.tif")]
    imgs.sort(key=lambda p: int(re.search(r"_(\d+)\.tif$", p.name).group(1)))
    flair = np.stack([np.asarray(Image.open(p))[:, :, 1] for p in imgs]).astype(np.float32)
    mask = np.stack([np.asarray(Image.open(p.with_name(p.stem + "_mask.tif"))) > 0 for p in imgs])
    return flair, mask


def estimate_spacing(vol):
    head = vol > 0.1 * np.percentile(vol[vol > 0], 99)
    head = np.stack([ndi.binary_fill_holes(ndi.binary_opening(h, iterations=2)) for h in head])
    widths = np.array([np.ptp(np.nonzero(h.any(0))[0]) if h.any() else 0 for h in head])
    px = HEAD_WIDTH_MM / max(widths.max(), 1)
    k_wide = int(np.argmax(widths))
    above = np.where(widths[k_wide:] > 0.1 * widths.max())[0]
    n_dome = max(int(above.max()) if len(above) else 1, 1)
    dz = float(np.clip((HEAD_WIDTH_MM / 2) / n_dome, 1.5, 8.0))
    return dz, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/archive/kaggle_3m"))
    ap.add_argument("--out", default=str(ROOT / "data/out/kaggle_eval.csv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    folders = sorted(p for p in Path(args.data).iterdir() if p.is_dir())
    if args.limit:
        folders = folders[: args.limit]
    rows = []
    tp = fp = fn = tn = 0
    for i, folder in enumerate(folders, 1):
        vol, truth = load_patient(folder)
        dz, px = estimate_spacing(vol)
        t = time.perf_counter()
        seg = analyze(vol, (dz, px, px), "FLAIR", tissue_slices=0)
        dt = time.perf_counter() - t
        pred = seg.masks["lesion"]
        s = pred.sum() + truth.sum()
        dice = 2 * (pred & truth).sum() / s if s else 1.0
        ps, ts = pred.any((1, 2)), truth.any((1, 2))
        tp += int((ps & ts).sum()); fp += int((ps & ~ts).sum())
        fn += int((~ps & ts).sum()); tn += int((~ps & ~ts).sum())
        rows.append({"case": f"k{i:03d}", "slices": vol.shape[0], "dz_mm": round(dz, 2), "px_mm": round(px, 3),
                     "dice": round(float(dice), 4), "pred_px": int(pred.sum()), "truth_px": int(truth.sum()),
                     "warnings": " ".join(seg.warnings), "seconds": round(dt, 2)})
        print(f"k{i:03d} n={vol.shape[0]:2d} dz={dz:.1f} px={px:.2f}  dice {dice:.3f}  "
              f"pred {pred.sum():6d} truth {truth.sum():6d} px  {dt:.1f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = np.array([r["dice"] for r in rows if r["truth_px"] > 0])
    print(f"--- {len(rows)} patients: lesion Dice mean {d.mean():.3f} median {np.median(d):.3f}  "
          f"(>0.5: {np.mean(d > 0.5):.0%}, 0: {np.mean(d == 0):.0%})")
    print(f"slice detection: sensitivity {tp / max(tp + fn, 1):.3f}  specificity {tn / max(tn + fp, 1):.3f}  "
          f"(slices: {tp + fn} with lesion, {tn + fp} without)")


if __name__ == "__main__":
    main()
