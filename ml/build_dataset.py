"""Build the nnU-Net dataset for the AI segmenter (contract C6).

Cases
  ReMIND (data/synthseg/ReMIND-*): T1-post with a brain tumour. Gray/white matter and CSF come from
    SynthSeg (data/synthseg, FreeSurfer mri_synthseg --robust); tumour, ventricles and resection cavity
    come from ReMIND's own SEGs. Patients failing SynthSeg's QC (< 0.65) are left out.
  IXI (data/archive/ixi/T1 + synthseg/): healthy volunteers, so the lesion layer is empty. This teaches
    the model that a scan may have no tumour at all. Labels from ml/synthseg_dipy.py.

Labels (nnU-Net value -> C1 label)
  0 outside the head -> 0      1 skull -> 1      2 gray matter -> 2     3 white matter -> 3
  4 CSF -> 4                   5 lesion -> 10
  6 ignore (no loss): a 3 mm ring around the tumour and the resection cavity +3 mm, where SynthSeg's
    labels mean nothing. Precedence where labels meet: tumour > ventricles > SynthSeg (C6).
  Brainstem and ventral diencephalon (mixed tissue) count as white matter, as in the guide images.
  Skull = head minus everything SynthSeg labels; the head outline is the classical head mask.

Every volume is put in the segmenter's frame first (axis 0 inferior->superior, rows anterior->posterior,
columns right->left: the same frame dicom_io's Volume uses), so the app can feed Volume.data straight in.

Split (patient level, seeded): 12 ReMIND test patients (incl. the 3 T1-post ones from ml/splits.json),
the rest train; IXI: as many healthy train scans as ReMIND train patients (balanced), the rest test.
Case IDs are anonymous (cv0001...); the map to real IDs goes to ml/splits_v2_private.json (gitignored).

Usage (repo root, AI venv): backend/.venv-ai/Scripts/python.exe -I ml/build_dataset.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from cortexview.segment import _down, _factors, _fill, _up, head_mask  # noqa: E402

SEED = 20261011
SYN = ROOT / "data/synthseg"
IXI = ROOT / "data/archive/ixi"
OUT = ROOT / "ml/nnunet/raw/Dataset501_CortexView"
GM = [3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58]
WM = [2, 41, 7, 46, 16, 28, 60]
CSF = [4, 5, 14, 15, 24, 43, 44]
KEEP_IN_TEST = ["ReMIND-008", "ReMIND-017", "ReMIND-025"]     # T1-post test patients of ml/splits.json
N_TEST_REMIND = 12
LABELS = {"background": 0, "skull": 1, "gray_matter": 2, "white_matter": 3, "csf": 4, "lesion": 5, "ignore": 6}


def canonical(img: nib.Nifti1Image):
    """(array [k, j, i] in the segmenter's frame, spacing (k, j, i), transform for other arrays)."""
    aff = img.affine
    lps = np.diag([-1, -1, 1]) @ aff[:3, :3]
    world = np.abs(lps).argmax(0)                       # world axis of each array axis (0 L, 1 P, 2 S)
    perm = [int(np.where(world == w)[0][0]) for w in (2, 1, 0)]
    flips = [lps[w, a] < 0 for w, a in zip((2, 1, 0), perm)]
    spacing = tuple(float(np.linalg.norm(aff[:3, a])) for a in perm)

    def fix(x):
        x = np.transpose(np.asarray(x), perm)
        for ax, f in enumerate(flips):
            if f:
                x = np.flip(x, ax)
        return np.ascontiguousarray(x)
    return fix(np.asarray(img.dataobj, np.float32)), spacing, fix


def head_of(img, spacing):
    f = _factors(spacing, 2.0)
    head_c = head_mask(_down(img, f), tuple(s * ff for s, ff in zip(spacing, f)))
    return _up(head_c, f, img.shape) > 0.5


def ring(mask, mm, spacing):
    if not mask.any():
        return mask
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= mm


def label_map(syn, head, spacing, tumor=None, vent=None, cavity=None):
    lab = np.zeros(syn.shape, np.uint8)
    lab[head | (syn > 0)] = LABELS["skull"]
    lab[np.isin(syn, GM)] = LABELS["gray_matter"]
    lab[np.isin(syn, WM)] = LABELS["white_matter"]
    lab[np.isin(syn, CSF)] = LABELS["csf"]
    if vent is not None:
        lab[vent] = LABELS["csf"]
    ignore = np.zeros(syn.shape, bool)
    if cavity is not None:
        ignore |= ring(cavity, 3.0, spacing)
    if tumor is not None:
        ignore |= ring(tumor, 3.0, spacing) & ~tumor
    lab[ignore] = LABELS["ignore"]
    if tumor is not None:
        lab[tumor] = LABELS["lesion"]
    return lab


def write(arr, spacing, path, dtype):
    im = sitk.GetImageFromArray(arr.astype(dtype))       # numpy [k, j, i] -> ITK (i, j, k)
    im.SetSpacing(tuple(float(s) for s in spacing[::-1]))
    sitk.WriteImage(im, str(path), True)


def remind_case(pid):
    d = SYN / pid
    img, spacing, fix = canonical(nib.load(str(d / "image.nii.gz")))
    opt = lambda n: fix(nib.load(str(d / n)).dataobj) > 0 if (d / n).exists() else None
    syn = fix(nib.load(str(d / "synthseg_native.nii.gz")).dataobj).astype(np.int16)
    tumor, vent, cavity = opt("tumor.nii.gz"), opt("ventricles.nii.gz"), opt("cavity.nii.gz")
    head = _fill(head_of(img, spacing) | (syn > 0) | (tumor if tumor is not None else False))
    return img, spacing, label_map(syn, head, spacing, tumor, vent, cavity)


def ixi_case(name):
    img, spacing, fix = canonical(nib.load(str(IXI / "T1" / f"{name}.nii.gz")))
    syn = fix(nib.load(str(IXI / "synthseg" / f"{name}_synthseg.nii.gz")).dataobj).astype(np.int16)
    head = _fill(head_of(img, spacing) | (syn > 0))
    return img, spacing, label_map(syn, head, spacing)


def main() -> None:
    qc = {r["PatientID"]: float(r["synthseg_qc_min"]) for r in csv.DictReader(open(SYN / "manifest.csv"))}
    remind = sorted(p for p, q in qc.items() if q >= 0.65 and (SYN / p / "synthseg_native.nii.gz").exists())
    rng = random.Random(SEED)
    others = [p for p in remind if p not in KEEP_IN_TEST]
    rng.shuffle(others)
    test_r = sorted(KEEP_IN_TEST + others[:N_TEST_REMIND - len(KEEP_IN_TEST)])
    train_r = sorted(others[N_TEST_REMIND - len(KEEP_IN_TEST):])

    ixi = sorted(p.name[:-len("_synthseg.nii.gz")] for p in (IXI / "synthseg").glob("*_synthseg.nii.gz"))
    rng.shuffle(ixi)
    train_i, test_i = sorted(ixi[:len(train_r)]), sorted(ixi[len(train_r):])

    cases = [(s, "remind", "train") for s in train_r] + [(s, "ixi", "train") for s in train_i] + \
            [(s, "remind", "test") for s in test_r] + [(s, "ixi", "test") for s in test_i]
    order = list(range(len(cases)))
    rng.shuffle(order)                                    # anonymous IDs don't reveal source or order
    ids = {cases[i][0]: f"cv{n + 1:04d}" for n, i in enumerate(order)}

    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    public = {"seed": SEED, "note": "Patient-level split, anonymous IDs only.", "train": [], "test": []}
    for src, kind, part in cases:
        cid = ids[src]
        public[part].append({"id": cid, "source": kind, "tumour": kind == "remind"})
        img_dir, lab_dir = ("imagesTr", "labelsTr") if part == "train" else ("imagesTs", "labelsTs")
        if (OUT / lab_dir / f"{cid}.nii.gz").exists():
            continue
        img, spacing, lab = remind_case(src) if kind == "remind" else ixi_case(src)
        brain_ml = float(((lab >= 2) & (lab <= 5)).sum() * np.prod(spacing) / 1000)
        if not 700 < brain_ml < 2300:
            raise SystemExit(f"{cid}: implausible brain volume {brain_ml:.0f} mL; check its labels")
        write(img, spacing, OUT / img_dir / f"{cid}_0000.nii.gz", np.float32)
        write(lab, spacing, OUT / lab_dir / f"{cid}.nii.gz", np.uint8)
        print(f"{cid} {part:5s} {kind:6s} {img.shape} brain {brain_ml:.0f} mL", flush=True)

    json.dump({"channel_names": {"0": "T1"}, "labels": LABELS, "numTraining": len(public["train"]),
               "file_ending": ".nii.gz", "name": "CortexView",
               "description": "ReMIND T1-post (SynthSeg + ReMIND labels) and healthy IXI T1 (SynthSeg)"},
              open(OUT / "dataset.json", "w"), indent=1)
    json.dump(public, open(ROOT / "ml/splits_v2.json", "w"), indent=1)
    json.dump({v: k for k, v in ids.items()}, open(ROOT / "ml/splits_v2_private.json", "w"), indent=1)
    print(f"train {len(public['train'])} ({len(train_r)} ReMIND + {len(train_i)} IXI), "
          f"test {len(public['test'])} ({len(test_r)} ReMIND + {len(test_i)} IXI)")


if __name__ == "__main__":
    main()
