"""
Step 3: score every candidate against its guide and make side-by-side previews.

Each pixel of the guide and of the candidate is mapped to the nearest palette color,
then per-label IoU is computed. This is a simple first version of fidelity.py:
it assumes the illustration keeps each region in its palette's color family
(which the generation prompt asks for).

    python fidelity_filter.py --guides guides --candidates candidates --palette palette.json

palette.json maps label id -> RGB, e.g. {"0": [0,0,0], "1": [230,220,200], "2": [...], "10": [...]}
Use the FROZEN team palette (shared/layers.json), converted to this shape.
"""
import argparse
import csv
import json
import pathlib

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--guides", default="guides")
ap.add_argument("--candidates", default="candidates")
ap.add_argument("--palette", default="palette.json")
ap.add_argument("--out_csv", default="fidelity.csv")
ap.add_argument("--preview_dir", default="previews")
ap.add_argument("--threshold", type=float, default=0.85, help="min mean IoU to pass")
args = ap.parse_args()

pal = json.loads(pathlib.Path(args.palette).read_text())
ids = np.array([int(k) for k in pal.keys()])
colors = np.array([pal[k] for k in pal.keys()], dtype=np.float32)  # (K, 3)


def to_labels(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)  # (H, W, 3)
    d = ((arr[:, :, None, :] - colors[None, None, :, :]) ** 2).sum(-1)  # (H, W, K)
    return ids[d.argmin(-1)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


rows = []
guide_cache = {}
for cp in sorted(pathlib.Path(args.candidates).glob("*__s*.png")):
    stem = cp.stem.split("__s")[0]
    gp = pathlib.Path(args.guides) / f"{stem}.png"
    if not gp.exists():
        print(f"skip {cp.name}: no guide {gp.name}")
        continue
    if stem not in guide_cache:
        g_img = Image.open(gp).convert("RGB")
        guide_cache[stem] = (g_img, to_labels(g_img))
    g_img, g_lab = guide_cache[stem]

    c_img = Image.open(cp).convert("RGB").resize(g_img.size, Image.BILINEAR)
    c_lab = to_labels(c_img)

    present = [int(i) for i in np.unique(g_lab) if i != 0]  # skip background
    per = {i: iou(g_lab == i, c_lab == i) for i in present}
    findings = [v for i, v in per.items() if i >= 10]
    rows.append({
        "candidate": cp.name,
        "guide": gp.name,
        "mean_iou": round(float(np.mean(list(per.values()))) if per else 0.0, 4),
        "min_iou": round(min(per.values()) if per else 0.0, 4),
        "finding_iou": round(min(findings), 4) if findings else "",
        "per_label": json.dumps({k: round(v, 3) for k, v in per.items()}),
    })

rows.sort(key=lambda r: r["mean_iou"], reverse=True)
with open(args.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Side-by-side previews (guide | candidate) for the best passing sample per guide.
prev = pathlib.Path(args.preview_dir)
prev.mkdir(exist_ok=True)
best = {}
for r in rows:
    if r["mean_iou"] >= args.threshold and r["guide"] not in best:
        best[r["guide"]] = r
for r in best.values():
    g = Image.open(pathlib.Path(args.guides) / r["guide"]).convert("RGB")
    c = Image.open(pathlib.Path(args.candidates) / r["candidate"]).convert("RGB").resize(g.size)
    canvas = Image.new("RGB", (g.width * 2, g.height))
    canvas.paste(g, (0, 0))
    canvas.paste(c, (g.width, 0))
    canvas.save(prev / f"{r['mean_iou']:.3f}__{r['candidate']}")

n_pass = sum(r["mean_iou"] >= args.threshold for r in rows)
print(f"{len(rows)} candidates scored, {n_pass} pass (mean IoU >= {args.threshold}), "
      f"{len(best)} guides have a passing sample. Previews in {prev}/")
print(f"Base-model pass rate: {n_pass / len(rows):.1%}  <- keep this number for the pitch")
