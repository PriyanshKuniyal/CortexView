"""
Pilot check: is the ReMIND tumor SEG aligned with the T1 it was drawn on?

For each patient: an overlay PNG at the largest-tumor axial slice (display orientation, radiological by default),
plus a numeric test. The mean T1 gradient magnitude along the mask boundary should be higher for the mask as
stored than for the same mask flipped left-right or shifted in-plane by +/- N voxels. A flipped or shifted SEG
fails it. z shifts are reported but not used for pass/fail; a coronal panel shows z alignment instead.

    python check_seg_alignment.py --work W --only ID1,ID2,ID3 [--out DIR] [--neurological]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def boundary_edge(grad, mask):
    from scipy.ndimage import binary_erosion
    b = mask & ~binary_erosion(mask)
    return float(grad[b].mean()) if b.any() else 0.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from scipy.ndimage import gaussian_gradient_magnitude

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--neurological", dest="radiological", action="store_false")
    a = ap.parse_args()

    deriv = Path(a.work) / "derivatives" / "synthseg"
    out = Path(a.out) if a.out else deriv / "qc_png"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for pid in a.only.split(","):
        d = deriv / pid
        t1 = np.asanyarray(nib.as_closest_canonical(nib.load(str(d / "image.nii.gz"))).dataobj).astype(np.float32)
        tum = np.asanyarray(nib.as_closest_canonical(nib.load(str(d / "tumor.nii.gz"))).dataobj).astype(bool)
        assert t1.shape == tum.shape, (t1.shape, tum.shape)
        grad = gaussian_gradient_magnitude(t1, 1.0)
        z = int(tum.sum(axis=(0, 1)).argmax())

        score = boundary_edge(grad, tum)
        alts = {"flip_LR": boundary_edge(grad, tum[::-1, :, :])}
        for ax, nm in [(0, "x"), (1, "y"), (2, "z")]:
            for s in (-a.shift, a.shift):
                alts[f"shift_{nm}{s:+d}"] = boundary_edge(grad, np.roll(tum, s, axis=ax))
        brain_vals = t1[t1 > np.percentile(t1, 50)]
        # z shifts are weak evidence for tall, smooth-walled tumors (the in-plane boundary barely changes),
        # so the pass/fail uses flip + in-plane shifts; z results are reported and checked visually (coronal)
        inplane = {k: v for k, v in alts.items() if not k.startswith("shift_z")}
        res = {
            "slice_ras": z,
            "tumor_cm3": round(float(tum.sum()) * float(np.prod(nib.load(str(d / "image.nii.gz")).header.get_zooms()[:3])) / 1000, 1),
            "boundary_edge": round(score, 2),
            "best_alternative": max(alts, key=alts.get),
            "best_alternative_edge": round(max(alts.values()), 2),
            "beats_flip_and_inplane_shifts": bool(score > max(inplane.values())),
            "beats_z_shifts": bool(score > max(v for k, v in alts.items() if k.startswith("shift_z"))),
            "tumor_mean_over_brain_median": round(float(t1[tum].mean() / np.median(brain_vals)), 2),
        }
        results[pid] = res

        # overlay in display orientation: anterior up, radiological = patient right on screen left
        def disp(sl):
            o = sl.T[::-1, :]
            return o[:, ::-1] if a.radiological else o
        lo, hi = np.percentile(t1[t1 > 0], [1, 99.5])
        base = np.clip((disp(t1[:, :, z]) - lo) / (hi - lo + 1e-6), 0, 1)
        m = disp(tum[:, :, z])
        # coronal slice through the tumor's largest-area y, superior up, same left/right convention
        y = int(tum.sum(axis=(0, 2)).argmax())
        def disp_cor(sl):
            o = sl.T[::-1, :]
            return o[:, ::-1] if a.radiological else o
        zx, _, zz = nib.load(str(d / "image.nii.gz")).header.get_zooms()[:3]
        cor = np.clip((disp_cor(t1[:, y, :]) - lo) / (hi - lo + 1e-6), 0, 1)
        mc = disp_cor(tum[:, y, :])
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(base, cmap="gray")
        ax[0].set_title(f"{pid} T1 axial z={z}", fontsize=9)
        ax[1].imshow(base, cmap="gray")
        ax[1].contour(m, levels=[0.5], colors="red", linewidths=1)
        ok = res["beats_flip_and_inplane_shifts"]
        ax[1].set_title(f"axial contour ({'pass' if ok else 'CHECK'}; edge {res['boundary_edge']} "
                        f"vs {res['best_alternative']} {res['best_alternative_edge']})", fontsize=9)
        ax[2].imshow(cor, cmap="gray", aspect=float(zz) / float(zx))
        ax[2].contour(mc, levels=[0.5], colors="red", linewidths=1)
        ax[2].set_title(f"coronal y={y} (checks z alignment)", fontsize=9)
        side = "R" if a.radiological else "L"
        for x in ax:
            x.axis("off")
            x.text(3, 12, side, color="yellow", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / f"{pid}_tumor_overlay.png", dpi=90)
        plt.close(fig)
        print(pid, json.dumps(res), flush=True)
    (out / "alignment_check.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
