"""Download UPENN-GBM patients from TCIA as an outside test set, in the same layout as the ReMIND download.

UPENN-GBM (University of Pennsylvania glioblastoma, CC BY 4.0) is not used anywhere in CortexView's
segmentation tuning or LoRA training, so it is a fair "never seen" test. Only the structural series are
fetched (T1, T1 post-contrast, T2, FLAIR); perfusion and diffusion series are skipped.

    backend/.venv/Scripts/python.exe scripts/analysis/download_upenn.py UPENN-GBM-00001 UPENN-GBM-00002

Writes data/archive/upenn-gbm/<PatientID>/Preop/<n>_MR_<description>/*.dcm and Preop.zip next to it.
Citation: Bakas S, et al. The University of Pennsylvania glioblastoma (UPenn-GBM) cohort. Sci Data 9, 453 (2022).
"""
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "archive" / "upenn-gbm"
STRUCTURAL = re.compile(r"t1|t2|flair", re.I)
SKIP = re.compile(r"perf|dti|diff|ep2d|adc|trace|fa\b", re.I)


def get(url, timeout=600):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main(pids):
    series = json.loads(get(f"{API}/getSeries?Collection=UPENN-GBM"))
    for pid in pids:
        mine = [s for s in series if s["PatientID"] == pid and s.get("Modality") == "MR"]
        keep = [s for s in mine if STRUCTURAL.search(s.get("SeriesDescription") or "")
                and not SKIP.search(s.get("SeriesDescription") or "")]
        if not keep:
            print(f"{pid}: no structural series found")
            continue
        preop = OUT / pid / "Preop"
        for n, s in enumerate(sorted(keep, key=lambda s: s.get("SeriesDescription") or ""), 1):
            name = re.sub(r"[^A-Za-z0-9]+", "_", (s.get("SeriesDescription") or "series").split(":")[0]).strip("_")
            dest = preop / f"{n}_MR_{name}"
            if (dest / ".complete").exists():
                continue
            print(f"{pid}: {s.get('SeriesDescription')} ({s.get('ImageCount')} images)")
            data = get(f"{API}/getImage?SeriesInstanceUID={urllib.parse.quote(s['SeriesInstanceUID'])}")
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for m in zf.infolist():
                    if not m.is_dir() and m.filename.lower().endswith(".dcm"):
                        (dest / Path(m.filename).name).write_bytes(zf.read(m))
            (dest / ".complete").touch()
        with zipfile.ZipFile(OUT / pid / "Preop.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(preop.rglob("*.dcm")):
                zf.write(f, f.relative_to(OUT / pid))
        print(f"{pid}: wrote {(OUT / pid / 'Preop.zip').relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["UPENN-GBM-00001"])
