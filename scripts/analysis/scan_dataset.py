"""Read-only scan of the kaggle_3m LGG dataset. Usage: python -I scan_dataset.py <kaggle_3m dir> <out.json>"""
import json, os, re, sys
from collections import Counter, deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiffmini import read_pixels

W = H = 256


def hist_stats(ch):
    c = Counter(ch)
    n = len(ch)
    nz = n - c.get(0, 0)
    s = sum(v * k for v, k in c.items())
    cum, pct, targets = 0, {}, [1, 50, 99, 99.9]
    for v in range(256):
        cum += c.get(v, 0)
        for t in targets:
            if t not in pct and cum >= n * t / 100:
                pct[t] = v
    return {"min": min(c), "max": max(c), "mean": s / n, "nonzero_frac": nz / n,
            "p1": pct[1], "p50": pct[50], "p99": pct[99], "p999": pct[99.9]}


def components(mask):
    seen = bytearray(W * H)
    sizes = []
    for i in range(W * H):
        if mask[i] and not seen[i]:
            q, n = deque([i]), 0
            seen[i] = 1
            while q:
                p = q.popleft(); n += 1
                y, x = divmod(p, W)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W:
                            j = yy * W + xx
                            if mask[j] and not seen[j]:
                                seen[j] = 1; q.append(j)
            sizes.append(n)
    return sorted(sizes, reverse=True)


def max_feret(mask):
    pts = [divmod(i, W) for i in range(W * H) if mask[i]]
    # boundary points only
    s = set(pts)
    b = [(y, x) for y, x in pts if not all((y + dy, x + dx) in s for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)))]
    best = 0
    for i in range(len(b)):
        y1, x1 = b[i]
        for j in range(i + 1, len(b)):
            d = (y1 - b[j][0]) ** 2 + (x1 - b[j][1]) ** 2
            if d > best: best = d
    return best ** 0.5


def main(root, out):
    rx = re.compile(r"^(TCGA_[A-Z0-9]{2}_[A-Z0-9]{4})_(\d{8})_(\d+)\.tif$")
    rows, problems = [], []
    for case in sorted(os.listdir(root)):
        d = os.path.join(root, case)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        imgs = [f for f in files if not f.endswith("_mask.tif")]
        for f in sorted(imgs, key=lambda s: int(rx.match(s).group(3)) if rx.match(s) else -1):
            m = rx.match(f)
            if not m:
                problems.append(f"bad name {case}/{f}"); continue
            mf = f[:-4] + "_mask.tif"
            if mf not in files:
                problems.append(f"missing mask {case}/{f}"); continue
            t, px = read_pixels(os.path.join(d, f))
            mt, mpx = read_pixels(os.path.join(d, mf))
            if (t["width"], t["height"], t["spp"]) != (W, H, 3) or (mt["width"], mt["height"], mt["spp"]) != (W, H, 1):
                problems.append(f"shape {case}/{f} {t['width']}x{t['height']}x{t['spp']} mask {mt['width']}x{mt['height']}x{mt['spp']}")
                continue
            chans = [px[k::3] for k in range(3)]
            mvals = sorted(set(mpx))
            pos = mpx.count(255) if 255 in mvals else sum(1 for v in mpx if v)
            row = {"case": case, "patient": m.group(1), "date": m.group(2), "slice": int(m.group(3)),
                   "ch": [hist_stats(c) for c in chans],
                   "pre_eq_flair": chans[0] == chans[1], "post_eq_flair": chans[2] == chans[1],
                   "mask_values": mvals, "mask_px": pos}
            # head extent on FLAIR: border rows/cols nonzero?
            fl = chans[1]
            row["border_nonzero"] = sum(1 for v in fl[:W] + fl[-W:] + fl[::W] + fl[W - 1::W] if v > 10)
            if pos:
                comps = components(mpx)
                row["mask_components"] = comps
                row["mask_feret_px"] = round(max_feret(mpx), 2) if pos < 6000 else None
                # lesion mean FLAIR intensity vs non-lesion nonzero FLAIR
                lv = [fl[i] for i in range(W * H) if mpx[i]]
                row["lesion_flair_mean"] = sum(lv) / len(lv)
                bv = [fl[i] for i in range(W * H) if not mpx[i] and fl[i] > 10]
                row["nonlesion_flair_mean"] = sum(bv) / max(1, len(bv))
                ys = [i // W for i in range(W * H) if mpx[i]]; xs = [i % W for i in range(W * H) if mpx[i]]
                row["mask_centroid_xy"] = [sum(xs) / len(xs), sum(ys) / len(ys)]
            rows.append(row)
        print(case, len(imgs), file=sys.stderr)
    json.dump({"rows": rows, "problems": problems}, open(out, "w"))
    print("slices", len(rows), "problems", len(problems))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
