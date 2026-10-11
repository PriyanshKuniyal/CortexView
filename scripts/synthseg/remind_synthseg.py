"""
ReMIND -> SynthSeg -> (1) GM/WM reference masks for evaluation, (2) guide.png files for LoRA training.

Run on Linux (Lightning Studio). Subcommands, in order:

    python remind_synthseg.py select   --work W [--limit 50]   # query IDC, download DICOM
    python remind_synthseg.py prepare  --work W                # DICOM -> NIfTI, SEG -> masks on the T1 grid
    python remind_synthseg.py synthseg --work W --threads 16   # run mri_synthseg (needs FreeSurfer >= 7.4)
    python remind_synthseg.py build    --work W                # GM/WM/ignore masks, manifest, QC PNGs
    python remind_synthseg.py guides   --work W --palette palette.json

Every step is resumable: finished patients are skipped on rerun.

Patient set: Preop patients whose 3D_AX_T1_postcontrast series has BOTH a tumor SEG and a cerebrum SEG
drawn on that same series (53 patients in IDC as of Oct 2026). All masks therefore share one grid and one
frame of reference, so no cross-series registration is needed.

Outputs are a "SynthSeg reference (silver standard)", never ground truth.
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---- SynthSeg / FreeSurfer label mapping v1 (check against FreeSurferColorLUT for your version) ----
WM_LABELS = [2, 41, 7, 46]
GM_LABELS = [3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58]
IGNORE_LABELS = [16, 28, 60]                  # brainstem, ventral DC: mixed GM/WM
CSF_LABELS = [4, 5, 14, 15, 24, 43, 44]       # ventricles + extra-cerebral CSF
CEREBRAL_GMWM = [2, 3, 41, 42]                # comparable with ReMIND "cerebrum" (no cerebellum/brainstem)
EXPECTED_SYNTHSEG_LABELS = set([0] + WM_LABELS + GM_LABELS + IGNORE_LABELS + CSF_LABELS)

# For guide.png: every brain pixel needs exactly one label.
GUIDE_GM = GM_LABELS
GUIDE_WM = WM_LABELS + IGNORE_LABELS          # brainstem / ventral DC drawn as white matter
GUIDE_CSF = CSF_LABELS

T1 = "3D_AX_T1_postcontrast"
SEG_ROLES = {
    "tumor": f"tumor seg - MR ref: {T1}",
    "cerebrum": f"cerebrum seg - MR ref: {T1}",
    "ventricles": f"ventricles seg - MR ref: {T1}",
    "cavity": f"previous_resection_cavity seg - MR ref: {T1}",
}


def log(*a):
    print(*a, flush=True)


def paths(work):
    w = Path(work)
    return {
        "work": w,
        "dicom": w / "dicom",
        "selection": w / "selection.csv",
        "deriv": w / "derivatives" / "synthseg",
        "guides": w / "guides",
    }


def read_selection(p):
    rows = list(csv.DictReader(open(p["selection"])))
    pts = {}
    for r in rows:
        pts.setdefault(r["PatientID"], {})[r["role"]] = r["SeriesInstanceUID"]
    return dict(sorted(pts.items()))


def series_dir(p, pid, uid):
    hits = list((p["dicom"] / pid).glob(f"*_{uid}"))
    return hits[0] if hits else None


# ------------------------------------------------------------------------------------------ select
def cmd_select(a):
    from idc_index import IDCClient

    p = paths(a.work)
    p["work"].mkdir(parents=True, exist_ok=True)
    c = IDCClient()
    roles_sql = ",".join(f"'{d}'" for d in [T1, *SEG_ROLES.values()])
    df = c.sql_query(f"""
        SELECT PatientID, SeriesDescription, SeriesInstanceUID, series_size_MB
        FROM index
        WHERE collection_id = 'remind' AND StudyDescription = 'Preop'
          AND SeriesDescription IN ({roles_sql})""")
    desc2role = {T1: "t1"} | {v: k for k, v in SEG_ROLES.items()}
    df["role"] = df["SeriesDescription"].map(desc2role)

    have = df.groupby("PatientID")["role"].apply(set)
    eligible = sorted(pid for pid, r in have.items() if {"t1", "tumor", "cerebrum"} <= r)
    dup = df.groupby(["PatientID", "role"]).size()
    multi = sorted({pid for (pid, role), n in dup.items() if n > 1 and pid in eligible})
    if multi:
        log(f"WARNING: patients with >1 series for a role (first one used): {multi}")
    if a.limit:
        eligible = eligible[: a.limit]
    sel = (df[df["PatientID"].isin(eligible)]
           .sort_values(["PatientID", "role", "SeriesInstanceUID"])
           .drop_duplicates(["PatientID", "role"]))
    sel[["PatientID", "role", "SeriesInstanceUID", "SeriesDescription", "series_size_MB"]].to_csv(
        p["selection"], index=False)
    log(f"{len(eligible)} eligible patients, {len(sel)} series, "
        f"{sel['series_size_MB'].sum() / 1024:.1f} GB -> {p['selection']}")
    if a.no_download:
        return
    c.download_from_selection(
        downloadDir=str(p["dicom"]),
        seriesInstanceUID=sel["SeriesInstanceUID"].tolist(),
        dirTemplate="%PatientID/%Modality_%SeriesInstanceUID",
        quiet=False,
    )
    log("download done")


# ------------------------------------------------------------------------------------------ prepare
# ReMIND's BINARY SEGs (all written by QIICR dcmqi, SoftwareVersions ef9e227) are mis-packed when the frame size
# N = Rows*Columns is not a multiple of 8. Within-byte bit offsets follow the standard (k*N mod 8), but the writer
# advances the byte pointer as if the per-frame remainder were lowbit(N mod 8) instead of N mod 8, so frame k
# really starts at bit  8*(k*(N//8) + (k*c)//8) + (k*N % 8),  c = r & -r,  r = N % 8.
# For r in {0, 1, 2, 4} that equals the standard layout; for r in {3, 5, 6, 7} standard decoders (pydicom,
# highdicom) return frames shifted sideways by a growing amount, wrapping around the frame edge. Found on the
# pilot (ReMIND-004 cerebrum, r=6) and fitted on all 202 SEGs by minimum 3-D mask surface area.
def seg_frame_starts(F, N, c):
    k = np.arange(F)
    return 8 * (k * (N // 8) + (k * c) // 8) + (k * (N % 8)) % 8


def decode_binary_seg_frames(ds):
    """Unpack a BINARY SEG's frames, repairing the dcmqi packing bug. Returns (frames[F,R,C] uint8, note)."""
    F, R, C = int(ds.NumberOfFrames), int(ds.Rows), int(ds.Columns)
    N, r = R * C, (R * C) % 8
    bits = np.unpackbits(np.frombuffer(ds.PixelData, np.uint8), bitorder="little")

    def decode(c):
        out = np.zeros((F, N), np.uint8)
        for i, s in enumerate(seg_frame_starts(F, N, c)):
            x = bits[s:s + N]
            out[i, :len(x)] = x
        return out.reshape(F, R, C)

    def surface(a):
        a = np.pad(a, 1).astype(np.int8)
        return sum(int(np.abs(np.diff(a, axis=ax)).sum()) for ax in range(3))

    c_bug = r & -r
    if c_bug == r:                                        # r in {0,1,2,4}: standard and buggy layouts coincide
        return decode(r), ""
    std, fix = decode(r), decode(c_bug)
    s_std, s_fix = surface(std), surface(fix)
    if s_fix <= s_std:                                    # ties (tiny masks) go to the fitted writer model
        return fix, f"dcmqi bit-packing repaired (N%8={r}, surface {s_std}->{s_fix})"
    return std, f"N%8={r} but standard packing fits better (surface {s_std} vs {s_fix})"


def seg_to_mask_on(seg_files, ref_img, t1_for_uid, t1_series_uid):
    """Read a DICOM SEG, check it refers to the T1 series / frame of reference, resample onto ref_img.

    BINARY SEGs are unpacked here (see decode_binary_seg_frames) and placed on a regular grid from their own
    per-frame positions; other segmentation types go through highdicom.
    """
    import highdicom as hd
    import pydicom
    import SimpleITK as sitk

    ds = pydicom.dcmread(str(seg_files[0]))
    notes = []
    refs = [s.SeriesInstanceUID for s in getattr(ds, "ReferencedSeriesSequence", [])]
    if refs and t1_series_uid not in refs:
        notes.append("seg references a different series")
    if ds.FrameOfReferenceUID != t1_for_uid:
        notes.append("frame of reference mismatch")
        return None, notes

    if ds.SegmentationType != "BINARY":
        seg = hd.seg.segread(str(seg_files[0]))
        try:
            vol = seg.get_volume(combine_segments=True, relabel=True)
        except ValueError:                                # overlapping segments can't be combined
            vol = seg.get_volume()
            vol = vol.with_array(vol.array.any(axis=-1).astype(np.uint8)) if vol.array.ndim == 4 else vol
            notes.append("overlapping segments merged")
        s_img = sitk.Cast(vol.to_simpleitk() > 0, sitk.sitkUInt8)
        return sitk.Resample(s_img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8), notes

    frames, note = decode_binary_seg_frames(ds)
    if note:
        notes.append(note)
    sh = ds.SharedFunctionalGroupsSequence[0]
    iop = np.array(sh.PlaneOrientationSequence[0].ImageOrientationPatient, float)
    row_sp, col_sp = (float(x) for x in sh.PixelMeasuresSequence[0].PixelSpacing)
    pos = np.array([fg.PlanePositionSequence[0].ImagePositionPatient
                    for fg in ds.PerFrameFunctionalGroupsSequence], float)
    normal = np.cross(iop[:3], iop[3:])
    along = pos @ normal
    # frames of all segments (and any empty slices dcmqi left out) go onto one regular grid along the normal
    steps = np.diff(np.unique(np.round(along, 3)))
    step = float(getattr(sh.PixelMeasuresSequence[0], "SpacingBetweenSlices", 0) or (steps.min() if len(steps) else 1.0))
    idx = np.round((along - along.min()) / step).astype(int)
    if len(steps) and np.abs((along - along.min()) / step - idx).max() > 0.05:
        notes.append("irregular SEG slice positions")
    vol = np.zeros((idx.max() + 1,) + frames.shape[1:], np.uint8)
    for i, k in enumerate(idx):
        vol[k] |= frames[i]
    origin = pos[np.argmin(along)]
    s_img = sitk.GetImageFromArray(vol)
    s_img.SetSpacing((col_sp, row_sp, step))
    s_img.SetOrigin(tuple(origin))
    s_img.SetDirection(tuple(np.column_stack([iop[:3], iop[3:], normal]).ravel()))
    return sitk.Resample(s_img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8), notes


def cmd_prepare(a):
    import pydicom
    import SimpleITK as sitk

    p = paths(a.work)
    pts = read_selection(p)
    # the pip `dcm2niix` package installs next to the venv's python, which may not be on PATH
    dcm2niix = shutil.which("dcm2niix") or shutil.which("dcm2niix", path=str(Path(sys.executable).parent))
    if dcm2niix is None:
        sys.exit("dcm2niix not found (pip install dcm2niix)")
    for pid, roles in pts.items():
        out = p["deriv"] / pid
        redo = a.redo_segs and (out / ".prepared").exists() and (out / "image.nii.gz").exists()
        if (out / ".prepared").exists() and not redo:
            continue
        out.mkdir(parents=True, exist_ok=True)
        notes = []
        t1_dir = series_dir(p, pid, roles["t1"])
        if t1_dir is None:
            log(f"{pid}: T1 not downloaded, skipping")
            continue
        if redo:                                          # keep image.nii.gz (SynthSeg may be reading it)
            old = json.loads((out / "prepare_notes.json").read_text()) if (out / "prepare_notes.json").exists() else []
            notes = [n for n in old if n.startswith("dcm2niix")]
        else:
            if not convert_t1(dcm2niix, t1_dir, out, notes, pid):
                continue
        ref = sitk.ReadImage(str(out / "image.nii.gz"))
        first = next(t1_dir.glob("*.dcm"))
        t1_for = pydicom.dcmread(first, stop_before_pixels=True).FrameOfReferenceUID
        for role in SEG_ROLES:
            if role not in roles:
                continue
            sdir = series_dir(p, pid, roles[role])
            files = sorted(sdir.glob("*.dcm")) if sdir else []
            if not files:
                notes.append(f"{role}: missing")
                continue
            mask, n = seg_to_mask_on(files, ref, t1_for, roles["t1"])
            notes += [f"{role}: {x}" for x in n]
            if mask is not None:
                sitk.WriteImage(mask, str(out / f"{role}.nii.gz"), True)
        (out / "prepare_notes.json").write_text(json.dumps(notes, indent=1))
        (out / ".prepared").touch()
        log(f"{pid}: {'SEG masks redone' if redo else 'prepared'} {'(' + '; '.join(notes) + ')' if notes else ''}")


def convert_t1(dcm2niix, t1_dir, out, notes, pid):
    """DICOM T1 series -> out/image.nii.gz. Returns False (and writes prepare_error.txt) on failure."""
    tmp = out / "_dcm2niix"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    r = subprocess.run([dcm2niix, "-z", "y", "-b", "n", "-f", "image", "-o", str(tmp), str(t1_dir)],
                       capture_output=True, text=True)
    niis = sorted(tmp.glob("*.nii.gz"))
    if r.returncode != 0 or len(niis) != 1:
        (out / "prepare_error.txt").write_text(r.stdout + r.stderr + f"\n{len(niis)} nifti files\n")
        log(f"{pid}: dcm2niix produced {len(niis)} files (rc={r.returncode}); see prepare_error.txt")
        return False
    if "non-uniform" in (r.stdout + r.stderr).lower() or "variable slice" in (r.stdout + r.stderr).lower():
        notes.append("dcm2niix reported non-uniform slices")
    shutil.move(str(niis[0]), out / "image.nii.gz")
    shutil.rmtree(tmp)
    return True


# ------------------------------------------------------------------------------------------ synthseg
def cmd_synthseg(a):
    p = paths(a.work)
    exe = a.synthseg_cmd.split()
    if shutil.which(exe[0]) is None:
        sys.exit(f"'{exe[0]}' not found. Source FreeSurfer first: "
                 "export FREESURFER_HOME=...; source $FREESURFER_HOME/SetUpFreeSurfer.sh")
    pids = sorted(d.name for d in p["deriv"].iterdir() if (d / ".prepared").exists()) if p["deriv"].exists() else []
    if a.only:
        pids = [x for x in pids if x in a.only.split(",")]
    for i, pid in enumerate(pids, 1):
        d = p["deriv"] / pid
        if (d / ".done").exists():
            continue
        cmd = exe + ["--i", str(d / "image.nii.gz"), "--o", str(d / "synthseg_1mm.nii.gz"),
                     "--robust", "--vol", str(d / "volumes.csv"), "--qc", str(d / "qc.csv"),
                     "--threads", str(a.threads), "--cpu"]
        log(f"[{i}/{len(pids)}] {pid}: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        (d / "synthseg.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0 or not (d / "synthseg_1mm.nii.gz").exists():
            log(f"{pid}: FAILED (rc={r.returncode}), see synthseg.log")
            continue
        (d / ".done").touch()


# ------------------------------------------------------------------------------------------ build
def read_qc_min(path):
    try:
        # skip blank rows: on Windows, SynthSeg's csv writer emits "\r\r\n" line endings
        rows = [r for r in csv.reader(open(path, newline="")) if any(c.strip() for c in r)]
        vals = [float(v) for v in rows[1][1:] if v.strip()]
        return round(min(vals), 3) if vals else None
    except Exception:
        return None


def dilate_mm(mask, spacing_zyx, mm):
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return mask
    return distance_transform_edt(~mask, sampling=spacing_zyx) <= mm


def load_np(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img   # array is (z, y, x)


def qc_png(out_png, t1, gm, wm, ign, tumor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brain_z = np.where((gm | wm).any(axis=(1, 2)))[0]
    if len(brain_z) == 0:
        return
    zs = [brain_z[int(len(brain_z) * f)] for f in (0.2, 0.5, 0.8)]
    if tumor is not None and tumor.any():
        zs.append(int(tumor.sum(axis=(1, 2)).argmax()))
    lo, hi = np.percentile(t1[t1 > 0], [1, 99]) if (t1 > 0).any() else (0, 1)
    fig, ax = plt.subplots(len(zs), 3, figsize=(9, 3 * len(zs)))
    ax = np.atleast_2d(ax)
    for r, z in enumerate(zs):
        base = np.clip((t1[z] - lo) / (hi - lo + 1e-6), 0, 1)
        ov = np.dstack([base] * 3)
        o1 = ov.copy()
        o1[gm[z]] = 0.5 * o1[gm[z]] + 0.5 * np.array([0.9, 0.3, 0.6])
        o1[wm[z]] = 0.5 * o1[wm[z]] + 0.5 * np.array([0.3, 0.8, 1.0])
        o2 = ov.copy()
        o2[ign[z]] = 0.4 * o2[ign[z]] + 0.6 * np.array([1.0, 0.8, 0.0])
        for c, im in enumerate([ov, o1, o2]):
            ax[r, c].imshow(im)
            ax[r, c].axis("off")
        ax[r, 0].set_title(f"z={z}", fontsize=8)
    ax[0, 1].set_title("GM (pink) / WM (blue)", fontsize=8)
    ax[0, 2].set_title("ignore (yellow)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def cmd_build(a):
    import SimpleITK as sitk

    p = paths(a.work)
    (p["deriv"] / "qc_png").mkdir(parents=True, exist_ok=True)
    (p["deriv"] / "label_mapping_v1.json").write_text(json.dumps({
        "white_matter": WM_LABELS, "gray_matter": GM_LABELS, "ignore": IGNORE_LABELS,
        "unused_csf_ventricles": CSF_LABELS,
        "ignore_also": "ReMIND tumor (+3 mm) and previous resection cavity (+3 mm)",
        "status": "SynthSeg reference (silver standard), not ground truth"}, indent=1))
    rows = []
    only = set(a.only.split(",")) if a.only else None
    for d in sorted(x for x in p["deriv"].iterdir() if (x / ".done").exists() and (only is None or x.name in only)):
        pid = d.name
        ref = sitk.ReadImage(str(d / "image.nii.gz"))
        if not (d / "synthseg_native.nii.gz").exists():
            ss = sitk.ReadImage(str(d / "synthseg_1mm.nii.gz"))
            nat = sitk.Resample(ss, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, ss.GetPixelID())
            sitk.WriteImage(nat, str(d / "synthseg_native.nii.gz"), True)
        lab, _ = load_np(d / "synthseg_native.nii.gz")
        t1, _ = load_np(d / "image.nii.gz")
        sp = ref.GetSpacing()[::-1]                              # (z, y, x) mm
        vox_cm3 = float(np.prod(sp)) / 1000.0
        unknown = sorted(set(np.unique(lab).tolist()) - EXPECTED_SYNTHSEG_LABELS)

        gm = np.isin(lab, GM_LABELS)
        wm = np.isin(lab, WM_LABELS)
        ign = np.isin(lab, IGNORE_LABELS)
        tumor = load_np(d / "tumor.nii.gz")[0].astype(bool) if (d / "tumor.nii.gz").exists() else None
        cav = load_np(d / "cavity.nii.gz")[0].astype(bool) if (d / "cavity.nii.gz").exists() else None
        cer = load_np(d / "cerebrum.nii.gz")[0].astype(bool) if (d / "cerebrum.nii.gz").exists() else None
        ven = load_np(d / "ventricles.nii.gz")[0].astype(bool) if (d / "ventricles.nii.gz").exists() else None
        if tumor is not None:
            ign |= dilate_mm(tumor, sp, 3.0)
        if cav is not None:
            ign |= dilate_mm(cav, sp, 3.0)

        for name, m in [("gray_matter", gm), ("white_matter", wm), ("ignore", ign)]:
            o = sitk.GetImageFromArray(m.astype(np.uint8))
            o.CopyInformation(ref)
            sitk.WriteImage(o, str(d / f"{name}.nii.gz"), True)

        cgw = np.isin(lab, CEREBRAL_GMWM)
        frac_in = round(float((cgw & cer).sum() / max(cgw.sum(), 1)), 3) if cer is not None else ""
        frac_ven = round(float(((gm | wm) & ven).sum() / max(ven.sum(), 1)), 3) if ven is not None else ""
        qcmin = read_qc_min(d / "qc.csv")
        notes = json.loads((d / "prepare_notes.json").read_text()) if (d / "prepare_notes.json").exists() else []
        if unknown:
            notes.append(f"unexpected SynthSeg labels {unknown}")
        rows.append({
            "PatientID": pid, "synthseg_qc_min": qcmin,
            "gm_cm3": round(gm.sum() * vox_cm3, 1), "wm_cm3": round(wm.sum() * vox_cm3, 1),
            "tumor_excluded": tumor is not None, "cavity_excluded": cav is not None,
            "frac_cerebral_gmwm_inside_remind_cerebrum": frac_in,
            "frac_remind_ventricles_covered_by_gmwm": frac_ven,
            "status": "", "notes": "; ".join(notes)})
        qc_png(p["deriv"] / "qc_png" / f"{pid}.png", t1, gm, wm, ign, tumor)
        log(f"{pid}: built")

    if not rows:
        sys.exit("no patients with SynthSeg output yet")
    # flags
    for k in ("gm_cm3", "wm_cm3"):
        v = np.array([r[k] for r in rows], float)
        med, mad = np.median(v), np.median(np.abs(v - np.median(v))) + 1e-6
        for r in rows:
            if abs(r[k] - med) > 3 * 1.4826 * mad:
                r["notes"] += f"; {k} outlier"
    for r in rows:
        bad = (r["synthseg_qc_min"] is not None and r["synthseg_qc_min"] < a.qc_threshold) \
            or (r["frac_cerebral_gmwm_inside_remind_cerebrum"] != "" and r["frac_cerebral_gmwm_inside_remind_cerebrum"] < 0.9) \
            or (r["frac_remind_ventricles_covered_by_gmwm"] != "" and r["frac_remind_ventricles_covered_by_gmwm"] > 0.05) \
            or "outlier" in r["notes"] or "mismatch" in r["notes"] or "unexpected" in r["notes"]
        r["status"] = "review" if bad else "ok"
    with open(p["deriv"] / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    log(f"{len(rows)} patients: {n_ok} ok, {len(rows) - n_ok} review. "
        f"Median GM {np.median([r['gm_cm3'] for r in rows]):.0f} cm3, "
        f"WM {np.median([r['wm_cm3'] for r in rows]):.0f} cm3. QC images: {p['deriv'] / 'qc_png'}")


# ------------------------------------------------------------------------------------------ guides
def head_mask(t1):
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    img = sitk.GetImageFromArray(t1.astype(np.float32))
    m = sitk.GetArrayFromImage(sitk.OtsuThreshold(img, 0, 1)).astype(bool)
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    m = ndi.binary_closing(m, iterations=2)
    for z in range(m.shape[0]):                       # fill holes slice by slice
        m[z] = ndi.binary_fill_holes(m[z])
    return m


def to_display(sl, radiological):
    # sl is an axial slice indexed [x, y] in RAS+ orientation (x -> patient right, y -> anterior)
    out = sl.T[::-1, :]                               # anterior at the top
    return out[:, ::-1] if radiological else out      # radiological: patient right on screen left


def square_crop(arrs, mask, margin=8):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return arrs
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx, half = (y0 + y1) // 2, (x0 + x1) // 2, max(y1 - y0, x1 - x0) // 2 + margin
    out = []
    for a in arrs:
        pad = np.pad(a, half, mode="constant")
        out.append(pad[cy: cy + 2 * half, cx: cx + 2 * half])
    return out


def cmd_guides(a):
    import nibabel as nib
    from PIL import Image

    p = paths(a.work)
    pal = {int(k): v for k, v in json.loads(Path(a.palette).read_text()).items()}
    for k in (0, 1, 2, 3, 4, 10):
        assert k in pal, f"palette.json is missing label {k}"
    lut = np.zeros((256, 3), np.uint8)
    for k, v in pal.items():
        lut[k] = v
    for sub in ("guide", "labels", "image", "meta"):
        (p["guides"] / sub).mkdir(parents=True, exist_ok=True)

    manifest = {}
    if (p["deriv"] / "manifest.csv").exists():
        manifest = {r["PatientID"]: r for r in csv.DictReader(open(p["deriv"] / "manifest.csv"))}
    pids = [pid for pid, r in sorted(manifest.items()) if a.include_review or r["status"] == "ok"]
    rng = random.Random(a.seed)
    shuffled = pids[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(pids) * a.holdout)) if a.holdout > 0 else 0
    split = {pid: ("test" if i < n_test else "train") for i, pid in enumerate(shuffled)}

    def canon(path):
        return np.asanyarray(nib.as_closest_canonical(nib.load(str(path))).dataobj)

    written = []
    for i, pid in enumerate(pids):
        d = p["deriv"] / pid
        img_n = nib.as_closest_canonical(nib.load(str(d / "image.nii.gz")))
        t1 = np.asanyarray(img_n.dataobj).astype(np.float32)
        sx, sy, sz = img_n.header.get_zooms()[:3]
        lab = canon(d / "synthseg_native.nii.gz").astype(np.int32)
        tumor = canon(d / "tumor.nii.gz").astype(bool)
        ven = canon(d / "ventricles.nii.gz").astype(bool) if (d / "ventricles.nii.gz").exists() else None
        cav = canon(d / "cavity.nii.gz").astype(bool) if (d / "cavity.nii.gz").exists() else None
        for nm, m in [("synthseg_native", lab), ("tumor", tumor), ("ventricles", ven), ("cavity", cav)]:
            if m is not None and m.shape != t1.shape:
                sys.exit(f"{pid}: {nm} shape {m.shape} != image {t1.shape} after canonical reorientation")

        brain = lab > 0
        head = head_mask(np.transpose(t1, (2, 1, 0))).transpose(2, 1, 0) | brain
        L = np.zeros(lab.shape, np.uint8)
        L[head] = 1
        L[np.isin(lab, GUIDE_WM)] = 3
        L[np.isin(lab, GUIDE_GM)] = 2
        L[np.isin(lab, GUIDE_CSF)] = 4
        if ven is not None:
            L[ven] = 4
        if cav is not None:
            L[cav] = 4                                   # fluid-filled cavity drawn as CSF
        L[tumor] = 10                                    # findings override anatomy

        # slice choice: largest tumor slice, plus (every other patient) a tumor-free slice
        area_brain = brain.sum(axis=(0, 1))
        area_tumor = tumor.sum(axis=(0, 1))
        picks = [("tumor", int(area_tumor.argmax()))] if area_tumor.max() > 0 else []
        if i % a.no_tumor_every == 0:
            ok = np.where((area_brain >= 0.25 * area_brain.max()) & (area_tumor == 0))[0]
            far = [z for z in ok if all(abs(z - pz) * sz >= 20 for _, pz in picks)]
            if far:
                picks.append(("no_tumor", int(rng.choice(far))))

        lo, hi = np.percentile(t1[brain], [1, 99.5]) if brain.any() else (0, 1)
        for kind, z in picks:
            lab2 = to_display(L[:, :, z], a.radiological)
            img2 = to_display(np.clip((t1[:, :, z] - lo) / (hi - lo + 1e-6), 0, 1), a.radiological)
            # in-plane voxels may be non-square: resample both to square pixels (row spacing) before
            # the square crop, so the crop and the final resize keep true anatomical proportions
            px_r, px_c = float(sy), float(sx)         # rows run along y, columns along x
            h, w = lab2.shape
            if abs(px_r - px_c) > 1e-3:
                w = int(round(w * px_c / px_r))
                lab2 = np.asarray(Image.fromarray(lab2).resize((w, h), Image.NEAREST))
                img2 = np.asarray(Image.fromarray(img2.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
            lab2, img2 = square_crop([lab2, img2], lab2 > 0)
            name = f"{pid}_s{z:03d}"
            labs = np.asarray(Image.fromarray(lab2).resize((a.size, a.size), Image.NEAREST))
            Image.fromarray(labs).save(p["guides"] / "labels" / f"{name}.png")
            Image.fromarray(lut[labs]).save(p["guides"] / "guide" / f"{name}.png")
            Image.fromarray((np.clip(img2, 0, 1) * 255).astype(np.uint8)).resize(
                (a.size, a.size), Image.BILINEAR).save(p["guides"] / "image" / f"{name}.png")
            fov_mm = lab2.shape[0] * px_r
            meta = {
                "patient": pid, "split": split[pid], "slice_index_ras": z, "kind": kind,
                "orientation": "radiological" if a.radiological else "neurological",
                "pixel_mm": round(fov_mm / a.size, 4),
                "tumor_area_mm2": round(float(area_tumor[z]) * float(sx * sy), 1),
                "labels_present": sorted(int(v) for v in np.unique(labs)),
                "source": "SynthSeg 1mm robust (silver standard) + ReMIND manual tumor SEG",
            }
            (p["guides"] / "meta" / f"{name}.json").write_text(json.dumps(meta, indent=1))
            written.append(meta)
        log(f"{pid} ({split[pid]}): {len(picks)} guide(s)")

    with open(p["guides"] / "splits.csv", "w", newline="") as f:
        csv.writer(f).writerows([("PatientID", "split")] + sorted(split.items()))
    n_t = sum(m["kind"] == "tumor" for m in written)
    log(f"{len(written)} guides ({n_t} with tumor, {len(written) - n_t} without) from {len(pids)} patients; "
        f"{sum(v == 'test' for v in split.values())} patients held out as test. -> {p['guides']}")


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-download", action="store_true")
    s = sub.add_parser("prepare")
    s.add_argument("--redo-segs", action="store_true",
                   help="re-derive SEG masks for already-prepared patients (keeps image.nii.gz)")
    s = sub.add_parser("synthseg"); s.add_argument("--threads", type=int, default=os.cpu_count())
    s.add_argument("--synthseg-cmd", default="mri_synthseg")
    s.add_argument("--only", default="", help="comma-separated patient IDs (for the pilot)")
    s = sub.add_parser("build"); s.add_argument("--qc-threshold", type=float, default=0.65)
    s.add_argument("--only", default="", help="comma-separated patient IDs (for the pilot)")
    s = sub.add_parser("guides"); s.add_argument("--palette", required=True)
    s.add_argument("--size", type=int, default=512)
    s.add_argument("--neurological", dest="radiological", action="store_false")
    s.add_argument("--no-tumor-every", type=int, default=2, help="add a tumor-free slice every Nth patient")
    s.add_argument("--holdout", type=float, default=0.25)
    s.add_argument("--include-review", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    for sp_ in sub.choices.values():
        sp_.add_argument("--work", default="remind_work")
    a = ap.parse_args()
    {"select": cmd_select, "prepare": cmd_prepare, "synthseg": cmd_synthseg,
     "build": cmd_build, "guides": cmd_guides}[a.cmd](a)


if __name__ == "__main__":
    main()
