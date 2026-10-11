"""Sample scans for the "Try it" page, from sources CortexView never trained or tuned on.

    survey  download candidate scans, run the real pipeline on each, write a review sheet
    build   package the curated cases (CASES below) into frontend/public/samples/
    warm    send every bundled sample through the running API once, so its illustration lands in the
            render cache (needs RENDER_CACHE_RECORD=1 and a working AI backend in backend/.env)

Sources (all outside the ReMIND / Kaggle-LGG data used to build CortexView):
  UPENN-GBM   glioblastoma, University of Pennsylvania, via TCIA. CC BY 4.0.
              Bakas S. et al., Sci Data 9, 453 (2022). https://doi.org/10.7937/TCIA.709X-DN49
  ds000221    healthy adults, MPI-Leipzig Mind-Brain-Body (LEMON), via OpenNeuro. CC0.
              Babayan A. et al., Sci Data 6, 180308 (2019). https://openneuro.org/datasets/ds000221
              Distributed as NIfTI; converted here to a DICOM series (geometry kept from the NIfTI affine).

Only the FLAIR series is bundled: automatic lesion detection runs on FLAIR, and one series keeps each zip small.

    backend/.venv/Scripts/python.exe scripts/samples/build_samples.py survey
    backend/.venv/Scripts/python.exe scripts/samples/build_samples.py build
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import struct
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
SRC = ROOT / "data" / "archive" / "samples_src"
OUT = ROOT / "frontend" / "public" / "samples"
NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
OPENNEURO = "https://s3.amazonaws.com/openneuro.org"

CREDITS = {
    "upenn": {"label": "UPenn-GBM · TCIA", "url": "https://www.cancerimagingarchive.net/collection/upenn-gbm/"},
    "lemon": {"label": "LEMON · OpenNeuro", "url": "https://openneuro.org/datasets/ds000221"},
}

# Curated after `survey` (see the review sheet): only cases where the app's automatic lesion outline matches the
# visible tumour. Locations are plain positions read from the scan (patient's left/right, front/back), not lobe
# diagnoses. All UPENN-GBM cases are glioblastoma (the collection's diagnosis); "healthy" = research volunteer.
GBM = "Glioblastoma · FLAIR"
CASES: dict[str, dict] = {
    "gbm-right-front": {"src": "upenn", "source_id": "UPENN-GBM-00020", "kind": "tumor", "title": "Tumor, right front", "detail": GBM},
    "gbm-left-upper-front": {"src": "upenn", "source_id": "UPENN-GBM-00035", "kind": "tumor", "title": "Tumor, left upper front", "detail": GBM},
    "gbm-left-lower-front": {"src": "upenn", "source_id": "UPENN-GBM-00028", "kind": "tumor", "title": "Large tumor, left lower front", "detail": GBM},
    "gbm-right-temple": {"src": "upenn", "source_id": "UPENN-GBM-00006", "kind": "tumor", "title": "Tumor, right side near the temple", "detail": GBM},
    "gbm-right-large": {"src": "upenn", "source_id": "UPENN-GBM-00012", "kind": "tumor", "title": "Large tumor, right side", "detail": GBM},
    "gbm-left-back": {"src": "upenn", "source_id": "UPENN-GBM-00007", "kind": "tumor", "title": "Tumor, left back", "detail": GBM},
    "gbm-right-back": {"src": "upenn", "source_id": "UPENN-GBM-00009", "kind": "tumor", "title": "Tumor, right back", "detail": GBM},
    "gbm-small-left-back": {"src": "upenn", "source_id": "UPENN-GBM-00013", "kind": "tumor", "title": "Small tumor, left back", "detail": GBM},
    "gbm-left-top": {"src": "upenn", "source_id": "UPENN-GBM-00026", "kind": "tumor", "title": "Tumor near the top of the brain, left", "detail": GBM},
    "gbm-right-upper-back": {"src": "upenn", "source_id": "UPENN-GBM-00023", "kind": "tumor", "title": "Tumor, right upper back", "detail": GBM},
    "gbm-right-midline": {"src": "upenn", "source_id": "UPENN-GBM-00003", "kind": "tumor", "title": "Tumor beside the midline, right", "detail": GBM},
    "gbm-small-right": {"src": "upenn", "source_id": "UPENN-GBM-00021", "kind": "tumor", "title": "Small tumor, right side", "detail": GBM},
    "gbm-right-swelling": {"src": "upenn", "source_id": "UPENN-GBM-00032", "kind": "tumor", "title": "Tumor with wide swelling, right", "detail": GBM},
    "gbm-left-lower": {"src": "upenn", "source_id": "UPENN-GBM-00016", "kind": "tumor", "title": "Tumor, left lower brain", "detail": GBM},
    "healthy-01": {"src": "lemon", "source_id": "sub-010001", "kind": "normal", "title": "Healthy adult brain, 1", "detail": "Research volunteer · FLAIR"},
    "healthy-02": {"src": "lemon", "source_id": "sub-010011", "kind": "normal", "title": "Healthy adult brain, 2", "detail": "Research volunteer · FLAIR"},
    "healthy-03": {"src": "lemon", "source_id": "sub-010014", "kind": "normal", "title": "Healthy adult brain, 3", "detail": "Research volunteer · FLAIR"},
    "healthy-04": {"src": "lemon", "source_id": "sub-010018", "kind": "normal", "title": "Healthy adult brain, 4", "detail": "Research volunteer · FLAIR"},
}


def get(url: str, timeout: int = 600) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


# ------------------------------------------------------------------------------------------ UPENN-GBM

def upenn_flairs(limit: int) -> list[dict]:
    series = json.loads(get(f"{NBIA}/getSeries?Collection=UPENN-GBM"))
    flair = [s for s in series if "flair" in (s.get("SeriesDescription") or "").lower()
             and "axial" in (s.get("SeriesDescription") or "").lower()]
    by_pid: dict[str, dict] = {}
    for s in sorted(flair, key=lambda s: s["PatientID"]):
        by_pid.setdefault(s["PatientID"], s)                 # first (baseline) FLAIR per patient
    return list(by_pid.values())[:limit]


def fetch_upenn(s: dict) -> Path:
    dest = SRC / "upenn" / s["PatientID"]
    if (dest / ".complete").exists():
        return dest
    data = get(f"{NBIA}/getImage?SeriesInstanceUID={urllib.parse.quote(s['SeriesInstanceUID'])}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for m in zf.infolist():
            if not m.is_dir() and m.filename.lower().endswith(".dcm"):
                (dest / Path(m.filename).name).write_bytes(zf.read(m))
    (dest / ".complete").touch()
    return dest


# ------------------------------------------------------------------------------------------ LEMON (NIfTI -> DICOM)

NIFTI_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 512: np.uint16}


def read_nifti(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Minimal NIfTI-1 reader: (data[i, j, k], 4x4 voxel->RAS affine from the sform, else the qform)."""
    h = raw[:348]
    assert struct.unpack("<i", h[:4])[0] == 348, "big-endian or not NIfTI-1"
    dim = struct.unpack("<8h", h[40:56])
    dtype = NIFTI_DTYPES[struct.unpack("<h", h[70:72])[0]]
    pix = struct.unpack("<8f", h[76:108])
    off = int(struct.unpack("<f", h[108:112])[0])
    slope, inter = struct.unpack("<2f", h[112:120])
    qcode, scode = struct.unpack("<2h", h[252:256])
    shape = dim[1:1 + dim[0]][:3]
    data = np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape)), offset=off).reshape(shape, order="F").astype(np.float32)
    if slope not in (0.0, 1.0) and np.isfinite(slope):
        data = data * slope + inter
    if scode > 0:
        A = np.eye(4)
        A[:3] = np.array(struct.unpack("<12f", h[280:328])).reshape(3, 4)
    else:
        assert qcode > 0, "no sform or qform"
        b, c, d, qx, qy, qz = struct.unpack("<6f", h[256:280])
        a = np.sqrt(max(0.0, 1 - b * b - c * c - d * d))
        R = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                      [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                      [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-c*c-b*b]])
        R[:, 2] *= -1 if pix[0] < 0 else 1
        A = np.eye(4)
        A[:3, :3] = R * np.array(pix[1:4])
        A[:3, 3] = (qx, qy, qz)
    return data, A


def nifti_to_dicom(data: np.ndarray, A: np.ndarray, dest: Path, subject: str, description: str) -> None:
    """One DICOM file per slice along the NIfTI k axis. NIfTI is RAS, DICOM is LPS: flip x and y."""
    lps = np.diag([-1, -1, 1, 1]) @ A
    row_dir = lps[:3, 0] / np.linalg.norm(lps[:3, 0])        # column index (i) runs along this
    col_dir = lps[:3, 1] / np.linalg.norm(lps[:3, 1])        # row index (j) runs along this
    di, dj, dk = (float(np.linalg.norm(lps[:3, n])) for n in range(3))
    vals = data[np.isfinite(data)]
    hi = float(np.percentile(vals, 99.9)) or 1.0
    img = np.clip(data / hi * 4000, 0, 4095).round().astype(np.uint16)
    study, series, frame = (generate_uid(entropy_srcs=[subject, t]) for t in ("study", "series", "frame"))
    dest.mkdir(parents=True, exist_ok=True)
    for k in range(img.shape[2]):
        sop = generate_uid(entropy_srcs=[subject, "sop", str(k)])
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = meta.MediaStorageSOPClassUID, sop, "MR"
        ds.PatientName = ds.PatientID = subject
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.SeriesNumber, ds.InstanceNumber = 1, k + 1
        ds.SeriesDescription = description
        ds.ImageOrientationPatient = [float(v) for v in (*row_dir, *col_dir)]
        ds.ImagePositionPatient = [float(v) for v in (lps @ np.array([0, 0, k, 1]))[:3]]
        ds.PixelSpacing = [dj, di]
        ds.SliceThickness = ds.SpacingBetweenSlices = dk
        pixels = np.ascontiguousarray(img[:, :, k].T)        # rows = j, columns = i
        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = pixels.tobytes()
        pydicom.dcmwrite(dest / f"{k:04d}.dcm", ds, enforce_file_format=True)


def lemon_subjects(limit: int) -> list[str]:
    import re
    xml = get(f"{OPENNEURO}?list-type=2&delimiter=/&prefix=ds000221/sub-").decode()
    subs = re.findall(r"<Prefix>ds000221/(sub-\d+)/</Prefix>", xml)
    return subs[:limit]


def fetch_lemon(sub: str) -> Path | None:
    dest = SRC / "lemon" / sub
    if (dest / ".complete").exists():
        return dest
    url = f"{OPENNEURO}/ds000221/{sub}/ses-02/anat/{sub}_ses-02_acq-lowres_FLAIR.nii.gz"
    try:
        raw = gzip.decompress(get(url))
    except Exception:
        return None
    data, A = read_nifti(raw)
    nifti_to_dicom(data, A, dest, sub, "FLAIR (OpenNeuro ds000221, converted from NIfTI)")
    (dest / ".complete").touch()
    return dest


# ------------------------------------------------------------------------------------------ review

def analyze(folder: Path):
    from cortexview import pipeline
    from cortexview.export import window_8bit
    res = pipeline.run(sorted(folder.glob("*.dcm")))
    img = window_8bit(np.asarray(res.image, np.float64), res.slice.head)
    return res, img


def overlay(img: np.ndarray, lesion: np.ndarray) -> Image.Image:
    rgb = np.repeat(img[..., None], 3, -1).astype(np.float32)
    rgb[lesion] = 0.45 * rgb[lesion] + 0.55 * np.array([230, 40, 40])
    return Image.fromarray(rgb.astype(np.uint8))


def survey(n_upenn: int, n_lemon: int) -> None:
    from metrics_engine import report
    jobs = [("upenn", s["PatientID"], lambda s=s: fetch_upenn(s)) for s in upenn_flairs(n_upenn)]
    jobs += [("lemon", sub, lambda sub=sub: fetch_lemon(sub)) for sub in lemon_subjects(n_lemon)]
    tiles, rows = [], []
    for src, pid, fetch in jobs:
        try:
            folder = fetch()
            if folder is None:
                print(f"{pid}: no FLAIR")
                continue
            res, img = analyze(folder)
        except Exception as e:
            print(f"{pid}: failed ({type(e).__name__}: {e})")
            continue
        m = report(res).layers["lesion"]
        rows.append({"src": src, "id": pid, "slice": res.slice_index, "n": int(res.volume.shape[0]),
                     "lesion_cm2": None if m.area_mm2 is None else round(m.area_mm2 / 100, 1),
                     "diam_mm": m.max_diameter_mm, "side": m.hemisphere, "conf": m.confidence,
                     "warnings": res.warnings})
        print(json.dumps(rows[-1]))
        t = overlay(img, res.slice.masks["lesion"]).resize((220, int(220 * img.shape[0] / img.shape[1])))
        canvas = Image.new("RGB", (220, t.height + 16), "white")
        canvas.paste(t, (0, 16))
        ImageDraw.Draw(canvas).text((3, 2), f"{pid}  {rows[-1]['lesion_cm2']} cm2 {m.hemisphere}", fill="black")
        tiles.append(canvas)
    (SRC / "survey.json").write_text(json.dumps(rows, indent=1))
    cols = 6
    for page in range(0, len(tiles), 24):
        chunk = tiles[page:page + 24]
        h = max(t.height for t in chunk)
        sheet = Image.new("RGB", (cols * 222, ((len(chunk) + cols - 1) // cols) * (h + 2)), "white")
        for i, t in enumerate(chunk):
            sheet.paste(t, ((i % cols) * 222, (i // cols) * (h + 2)))
        sheet.save(SRC / f"survey_{page // 24}.png")
    print("review sheets in", SRC)


# ------------------------------------------------------------------------------------------ build

def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sid, c in CASES.items():
        folder = SRC / c["src"] / c["source_id"]
        res, img = analyze(folder)
        zpath = OUT / f"{sid}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(folder.glob("*.dcm")):
                zf.write(f, f"{sid}/{f.name}")
        thumb = Image.fromarray(img)
        thumb.thumbnail((112, 112))
        thumb.save(OUT / f"{sid}.png", optimize=True)
        manifest.append({
            "id": sid, "title": c["title"], "detail": c["detail"], "kind": c["kind"],
            "file": f"samples/{sid}.zip", "thumb": f"samples/{sid}.png",
            "size_mb": round(zpath.stat().st_size / 2**20, 1), "slices": int(res.volume.shape[0]),
            "credit": CREDITS[c["src"]],
        })
        print(sid, manifest[-1]["size_mb"], "MB")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest)} samples -> {OUT.relative_to(ROOT)}")


def warm(api: str) -> None:
    import time
    import uuid
    for entry in json.loads((OUT / "manifest.json").read_text(encoding="utf-8")):
        data = (OUT.parent / entry["file"]).read_bytes()
        b = uuid.uuid4().hex
        head = (f'--{b}\r\nContent-Disposition: form-data; name="files"; filename="{entry["id"]}.zip"\r\n'
                "Content-Type: application/zip\r\n\r\n")
        body = head.encode() + data + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(f"{api}/api/analyze", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        out = {"status": "pending"}
        for _ in range(90):
            out = json.loads(get(api + r["render"]["poll_url"]))
            if out["status"] != "pending":
                break
            time.sleep(1)
        print(f'{entry["id"]:24} {out["status"]:8} {out.get("backend", out.get("reason", ""))}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey")
    s.add_argument("--upenn", type=int, default=36)
    s.add_argument("--lemon", type=int, default=8)
    sub.add_parser("build")
    w = sub.add_parser("warm")
    w.add_argument("--api", default="http://localhost:8000")
    a = ap.parse_args()
    if a.cmd == "survey":
        survey(a.upenn, a.lemon)
    elif a.cmd == "build":
        build()
    else:
        warm(a.api)
