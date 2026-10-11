"""Export ReMIND slices as contract-C2 folders (e.g. the sample slices for FT).

For each patient: pick the series with the C3 rule, take doctor labels from SEG files in the same
coordinate frame (tumour -> lesion, ventricles -> part of CSF, cerebrum -> brain boundary), segment the
rest classically, pick the C3 auto slice (largest lesion) plus optional extra slices, and write
slices/<sNNNN>/. Folder names are anonymous; the private ReMIND-ID mapping is appended to a CSV outside
the repo (default ~/.cortexview/id_map.csv).

Usage (from repo root):
  backend/.venv/Scripts/python.exe -I scripts/seg/export_remind.py --patients 1-5 --out data/out/slices
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from cortexview import dicom_io as io  # noqa: E402
from cortexview.export import write_slice_folder  # noqa: E402
from cortexview.segment import analyze, brain_slices  # noqa: E402


def parse_patients(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def next_id(id_map: Path) -> int:
    if not id_map.exists():
        return 1
    with id_map.open(newline="") as fh:
        ids = [int(r["slice_id"][1:]) for r in csv.DictReader(fh)]
    return max(ids, default=0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remind", default=str(ROOT / "data/archive/remind"))
    ap.add_argument("--out", default=str(ROOT / "data/out/slices"))
    ap.add_argument("--patients", default="1-5")
    ap.add_argument("--split", default="train", choices=["train", "eval", "demo"])
    ap.add_argument("--extra-slices", type=int, default=0,
                    help="extra slices per patient, evenly spread over the brain (besides the auto slice)")
    ap.add_argument("--require-lesion", action="store_true", help="skip patients without a usable tumour label")
    ap.add_argument("--id-map", default=str(Path.home() / ".cortexview" / "id_map.csv"))
    args = ap.parse_args()

    id_map = Path(args.id_map)
    id_map.parent.mkdir(parents=True, exist_ok=True)
    new_map = not id_map.exists()
    counter = next_id(id_map)
    with id_map.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_map:
            writer.writerow(["slice_id", "dataset", "patient", "series_description", "slice_index"])
        for pid in parse_patients(args.patients):
            folder = Path(args.remind) / f"ReMIND-{pid:03d}" / "Preop"
            if not folder.exists():
                continue
            series, seg_infos, _ = io.scan(sorted(folder.rglob("*.dcm")))
            info = io.choose_series(series, seg_infos)
            if info is None:
                print(f"ReMIND-{pid:03d}: no series that can be shown axially")
                continue
            vol = io.load_info(info)
            masks = [m for g in seg_infos if g.frame_of_reference == vol.frame_of_reference
                     for m in io.load_seg(pydicom.dcmread(g.source))]
            doctor, warnings = io.doctor_labels_on(vol, masks)
            if args.require_lesion and not doctor.get("lesion", np.zeros(1, bool)).any():
                print(f"ReMIND-{pid:03d}: no tumour label on the chosen series, skipped")
                continue
            seg = analyze(vol.data, vol.spacing, vol.sequence, doctor=doctor)
            picks = [seg.slice_index]
            if args.extra_slices:
                ok = brain_slices(seg)
                picks += [int(ok[int(i)]) for i in np.linspace(0, len(ok) - 1, args.extra_slices + 2)[1:-1]]
            for k in dict.fromkeys(picks):                           # unique, keep order
                sid = f"s{counter:04d}"
                cut = seg.cut(k)
                ref = {"lesion": doctor["lesion"][k]} if "lesion" in doctor and doctor["lesion"][k].any() else {}
                meta = {
                    "source": "remind", "split": args.split,
                    "sequence": vol.sequence, "sequence_source": vol.sequence_source,
                    "plane": "axial", "slice_index": int(k), "n_slices": int(vol.shape[0]),
                    "pixel_spacing_mm": [round(float(vol.spacing[1]), 4), round(float(vol.spacing[2]), 4)],
                    "slice_thickness_mm": round(float(vol.spacing[0]), 4),
                    "patient_left": vol.patient_left,
                    "layer_source": cut.source, "layer_confidence": cut.confidence,
                    "warnings": sorted(set(seg.warnings + vol.warnings + warnings)),
                }
                write_slice_folder(args.out, sid, vol.data[k], cut.masks, meta, head=cut.head, reference=ref)
                writer.writerow([sid, "ReMIND", f"ReMIND-{pid:03d}", vol.description, int(k)])
                print(f"{sid}: {vol.sequence} slice {k}/{vol.shape[0]}  sources {cut.source}")
                counter += 1


if __name__ == "__main__":
    main()
