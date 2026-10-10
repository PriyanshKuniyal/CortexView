"""Per-case: IoU of head footprint (>thr) between FLAIR and pre/post channels on the middle slice; head width/height in px.
Usage: python -I registration_check.py <kaggle_3m dir>"""
import os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiffmini import read_pixels
W = 256; THR = 25
root = sys.argv[1]; res = []
for case in sorted(os.listdir(root)):
    d = os.path.join(root, case)
    if not os.path.isdir(d): continue
    n = sum(1 for f in os.listdir(d) if f.endswith("_mask.tif"))
    _, px = read_pixels(os.path.join(d, f"{case}_{n // 2}.tif"))
    fp = [bytes(1 if v > THR else 0 for v in px[k::3]) for k in range(3)]
    def iou(a, b):
        i = sum(x & y for x, y in zip(a, b)); u = sum(x | y for x, y in zip(a, b)); return i / u if u else 1
    rows = [y for y in range(W) if sum(fp[1][y * W:(y + 1) * W]) > 3]; cols = [x for x in range(W) if sum(fp[1][x::W]) > 3]
    res.append((case, round(iou(fp[0], fp[1]), 2), round(iou(fp[2], fp[1]), 2), cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1))
lo = [r for r in res if min(r[1], r[2]) < 0.85]
print("cases with a channel footprint IoU<0.85 vs FLAIR (mid slice):", len(lo))
for r in lo: print("  ", r)
w = [r[3] for r in res]; h = [r[4] for r in res]
print("head width px min/med/max", min(w), st.median(w), max(w), "| height", min(h), st.median(h), max(h))
