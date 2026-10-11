"""Score the AI and the classical segmenter on the held-out test cases (never seen in training).

References: SynthSeg (gray/white matter, CSF; a silver standard, not ground truth) and ReMIND's manual
tumour labels; healthy IXI scans must come out with no lesion. The ignore zone (3 mm around the tumour,
resection cavity) is left out of both sides, as in training.

Prints per-layer Dice / sensitivity / precision, brain-pixel accuracy, tumour detection on tumour scans
and false alarms on healthy scans. Per-case rows go to ml/nnunet/eval_<segmenter>.jsonl (gitignored).

Usage (repo root, AI venv): backend/.venv-ai/Scripts/python.exe -I ml/evaluate.py [--classical]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
DATA = ROOT / "ml/nnunet/raw/Dataset501_CortexView"
NAMES = {1: "skull", 2: "gray_matter", 3: "white_matter", 4: "csf", 5: "lesion"}
IGNORE = 6
MIN_LESION_ML = 0.5        # smaller specks don't count as "a tumour was found"


def load(path):
    im = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(im), tuple(im.GetSpacing()[::-1])


def classical_classes(img, spacing, sequence):
    from cortexview.segment import analyze
    seg = analyze(img, spacing, sequence)
    out = np.zeros(img.shape, np.uint8)
    for c, lid in NAMES.items():
        out[seg.masks[lid]] = c
    return out


def lesion_blobs_ml(mask, vox_ml):
    lab, n = ndi.label(mask)
    if n == 0:
        return []
    return sorted((np.bincount(lab.ravel())[1:] * vox_ml).tolist(), reverse=True)


def score(pred, ref, vox_ml):
    ok = ref != IGNORE
    p, r = pred[ok], ref[ok]
    row = {}
    for c, lid in NAMES.items():
        a, b = p == c, r == c
        tp, fp, fn = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())
        row[lid] = {"tp": tp, "fp": fp, "fn": fn}
    brain_p, brain_r = (p >= 2) & (p <= 5), (r >= 2) & (r <= 5)
    tp = int((brain_p & brain_r).sum())
    row["brain"] = {"tp": tp, "fp": int((brain_p & ~brain_r).sum()), "fn": int((~brain_p & brain_r).sum())}
    inside = (r >= 2) & (r <= 4)                                  # tissue pixels (tumour scored separately)
    row["tissue_acc"] = float((p[inside] == r[inside]).mean())
    row["pred_lesion_blobs_ml"] = lesion_blobs_ml(pred == 5, vox_ml)
    row["ref_lesion_ml"] = float((ref == 5).sum() * vox_ml)
    return row


def dice(k):
    return 2 * k["tp"] / max(2 * k["tp"] + k["fp"] + k["fn"], 1)


def summary(rows, name):
    print(f"\n===== {name} =====")
    for group, sel in (("tumour scans (ReMIND)", [r for r in rows if r["tumour"]]),
                       ("healthy scans (IXI)", [r for r in rows if not r["tumour"]])):
        if not sel:
            continue
        print(f"-- {group}: {len(sel)} scans")
        for lid in ("brain", "skull", "gray_matter", "white_matter", "csf") + (("lesion",) if sel[0]["tumour"] else ()):
            d = np.array([dice(r[lid]) for r in sel])
            sens = np.array([r[lid]["tp"] / max(r[lid]["tp"] + r[lid]["fn"], 1) for r in sel])
            prec = np.array([r[lid]["tp"] / max(r[lid]["tp"] + r[lid]["fp"], 1) for r in sel])
            print(f"   {lid:13s} Dice mean {d.mean():.3f} median {np.median(d):.3f} worst {d.min():.3f}"
                  f" | caught {sens.mean():.3f} | correct {prec.mean():.3f}")
        acc = np.array([r["tissue_acc"] for r in sel])
        print(f"   tissue pixel accuracy (GM/WM/CSF) mean {acc.mean():.3f} median {np.median(acc):.3f} worst {acc.min():.3f}")
        found = [bool(r["pred_lesion_blobs_ml"]) and r["pred_lesion_blobs_ml"][0] >= MIN_LESION_ML for r in sel]
        if sel[0]["tumour"]:
            print(f"   tumour found in {sum(found)}/{len(sel)} tumour scans")
        else:
            fp = [sum(r["pred_lesion_blobs_ml"]) for r in sel]
            print(f"   false tumour (>= {MIN_LESION_ML} mL) in {sum(found)}/{len(sel)} healthy scans; "
                  f"total false-lesion volume median {np.median(fp):.2f} mL, max {max(fp):.2f} mL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", action="store_true", help="score the classical segmenter instead")
    a = ap.parse_args()
    split = json.load(open(ROOT / "ml/splits_v2.json"))
    name = "classical" if a.classical else "ai"
    out = (ROOT / f"ml/nnunet/eval_{name}.jsonl").open("w")
    rows = []
    for case in split["test"]:
        cid = case["id"]
        img, spacing = load(DATA / "imagesTs" / f"{cid}_0000.nii.gz")
        ref, _ = load(DATA / "labelsTs" / f"{cid}.nii.gz")
        t = time.perf_counter()
        if a.classical:
            pred = classical_classes(img.astype(np.float32), spacing, "T1_post" if case["tumour"] else "T1_pre")
        else:
            from cortexview.segment_ai import predict
            pred, _ = predict(img, spacing)
        row = {"id": cid, "tumour": case["tumour"], "seconds": round(time.perf_counter() - t, 1),
               **score(pred, ref, float(np.prod(spacing)) / 1000)}
        rows.append(row)
        out.write(json.dumps(row) + "\n"); out.flush()
        print(f"{cid} {'tumour ' if case['tumour'] else 'healthy'} {row['seconds']:5.1f}s  "
              + "  ".join(f"{k[:5]} {dice(row[k]):.2f}" for k in ("brain", "gray_matter", "white_matter", "csf", "lesion")),
              flush=True)
    summary(rows, name)


if __name__ == "__main__":
    main()
