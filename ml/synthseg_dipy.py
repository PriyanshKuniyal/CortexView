"""Run SynthSeg on NIfTI T1 scans with DIPY's PyTorch port (original SynthSeg weights, GPU if present).

Used for the healthy IXI scans, which have no SynthSeg labels yet. The ReMIND labels in data/synthseg
came from FreeSurfer's `mri_synthseg --robust`; `--check` runs this port on ReMIND scans too and
reports how well the two agree, so the two label sources can be trusted to mean the same thing.

Writes <out>/<scan name>_synthseg.nii.gz (FreeSurfer label numbers, on the input grid).

Usage (from the repo root, AI venv):
  backend/.venv-ai/Scripts/python.exe -I ml/synthseg_dipy.py data/archive/ixi/T1 --out data/archive/ixi/synthseg
  backend/.venv-ai/Scripts/python.exe -I ml/synthseg_dipy.py --check ReMIND-002 ReMIND-017
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GM = [3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58]
WM = [2, 41, 7, 46]
CSF = [4, 5, 14, 15, 24, 43, 44]


def model():
    import torch
    from dipy.nn.torch.synthseg import SynthSeg

    def gpu_safe_predict(self, x):
        # DIPY 1.12's version returns a GPU tensor to numpy (fails on CUDA) and keeps autograd state.
        # Half precision on GPU: in float32 a whole head doesn't fit in 8 GB, and the driver then spills
        # into system RAM (~10x slower). Labels are an argmax, so this doesn't change them in practice.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            return self.model(x).float().cpu().numpy()
    SynthSeg._SynthSeg__predict = gpu_safe_predict
    net = SynthSeg(use_cuda=torch.cuda.is_available())
    net.model.eval()
    return net


def run(net, path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # DIPY's notes about mask post-processing
        labels, _, _ = net.predict(data, img.affine)
    return labels.astype(np.int16), img.affine


def check(net, pids):
    dice = lambda a, b: 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    for pid in pids:
        d = ROOT / "data/synthseg" / pid
        ours, _ = run(net, d / "image.nii.gz")
        ref = np.asarray(nib.load(str(d / "synthseg_native.nii.gz")).dataobj).astype(np.int16)
        ok = ~(np.asarray(nib.load(str(d / "ignore.nii.gz")).dataobj) > 0)
        msg = "  ".join(f"{n} {dice(np.isin(ours, L) & ok, np.isin(ref, L) & ok):.3f}"
                        for n, L in (("GM", GM), ("WM", WM), ("CSF", CSF)))
        msg += f"  brain {dice((ours > 0) & ok, (ref > 0) & ok):.3f}"
        print(f"{pid}: DIPY SynthSeg vs FreeSurfer SynthSeg-robust  {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", type=Path, help="folder of *.nii.gz scans")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", nargs="+", metavar="PATIENT")
    a = ap.parse_args()
    net = model()
    if a.check:
        check(net, a.check)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(a.src.glob("*.nii.gz")):
        dst = a.out / path.name.replace(".nii.gz", "_synthseg.nii.gz")
        if dst.exists():
            continue
        t = time.perf_counter()
        labels, affine = run(net, path)
        tmp = dst.with_name(dst.name + ".part.nii.gz")
        nib.save(nib.Nifti1Image(labels, affine), str(tmp))
        tmp.replace(dst)
        print(f"{path.name}: {time.perf_counter() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
