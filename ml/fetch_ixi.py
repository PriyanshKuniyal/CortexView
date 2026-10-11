"""Download healthy-volunteer T1 scans from IXI (no tumour) for the AI segmenter (stdlib only).

IXI-T1.tar is 4.8 GB (581 subjects). It is read as a stream, and only the first `--count` scans are
kept, so only about count x 8 MB is downloaded. Scans land in data/archive/ixi/T1/ (gitignored).

Source: https://brain-development.org/ixi-dataset/  (licence CC BY-SA 3.0; cite as the IXI dataset)

Usage: python -I ml/fetch_ixi.py --count 70
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=70)
    ap.add_argument("--out", type=Path, default=ROOT / "data/archive/ixi/T1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in a.out.glob("*.nii.gz")}
    got = len(have)
    if got >= a.count:
        print(f"already have {got} scans")
        return
    with urllib.request.urlopen(URL, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|") as tar:
        for m in tar:
            name = Path(m.name).name
            if not (m.isfile() and name.endswith(".nii.gz")):
                continue
            if name in have:
                continue                                  # stream past scans from an earlier run
            src = tar.extractfile(m)
            tmp = a.out / (name + ".part")
            tmp.write_bytes(src.read())
            tmp.replace(a.out / name)
            got += 1
            print(f"{got:3d}/{a.count} {name}", flush=True)
            if got >= a.count:
                break


if __name__ == "__main__":
    main()
