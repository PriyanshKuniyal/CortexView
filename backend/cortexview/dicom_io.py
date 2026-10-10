"""DICOM loading: MR series -> Volume, DICOM-SEG -> doctor masks, sequence detection.

Nothing here logs header values or filenames (contract C3: no patient info in logs).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from .volume import (Volume, affine_from_planes, check_uniform, reorient_to_axial, resample_nearest,
                     slice_order)

SEG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.66.4"


# ---------------------------------------------------------------- reading + grouping

def read_datasets(paths) -> list[Dataset]:
    """Read every DICOM file; non-DICOM files are skipped (the caller decides about `not_dicom`)."""
    out = []
    for p in paths:
        try:
            out.append(pydicom.dcmread(str(p)))
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
    return out


def read_folder(folder) -> list[Dataset]:
    return read_datasets(sorted(Path(folder).rglob("*.dcm")))


def is_seg(ds: Dataset) -> bool:
    return str(ds.get("SOPClassUID", "")) == SEG_SOP_CLASS or ds.get("Modality") == "SEG"


def group_series(datasets) -> tuple[dict[str, list[Dataset]], list[Dataset]]:
    """Split into {SeriesInstanceUID: [MR slices]} and a list of SEG datasets."""
    series: dict[str, list[Dataset]] = defaultdict(list)
    segs = []
    for ds in datasets:
        if is_seg(ds):
            segs.append(ds)
        elif ds.get("Modality") == "MR" and "PixelData" in ds:
            series[str(ds.SeriesInstanceUID)].append(ds)
    return dict(series), segs


# ---------------------------------------------------------------- sequence detection

_SEQ_RULES = [  # (regex on SeriesDescription, sequence), first match wins
    (r"flair", "FLAIR"),
    (r"t1.*(post|\+c|gd|gad|contrast|\bce\b|mprage.*gd)", "T1_post"),
    (r"t1.*pre|t1", "T1_pre"),
    (r"t2", "T2"),
]


def detect_sequence(ds: Dataset) -> tuple[str, str]:
    """(sequence, sequence_source): SeriesDescription keywords, else TR/TE/TI rules, else unknown."""
    desc = str(ds.get("SeriesDescription", "")).lower().replace("_", " ")
    if "precontrast" in desc or "pre contrast" in desc:
        return ("T1_pre", "header") if "t1" in desc else ("unknown", "default")
    for pattern, seq in _SEQ_RULES:
        if re.search(pattern, desc):
            if seq == "T1_pre" and ds.get("ContrastBolusAgent"):
                return "T1_post", "header"
            return seq, "header"
    tr, te, ti = (_num(ds.get(k)) for k in ("RepetitionTime", "EchoTime", "InversionTime"))
    if tr and te:
        if ti and ti > 1500 and te > 70:
            return "FLAIR", "tr_te_ti"
        if tr < 1000 and te < 30:
            return ("T1_post" if ds.get("ContrastBolusAgent") else "T1_pre"), "tr_te_ti"
        if tr > 2000 and te > 70:
            return "T2", "tr_te_ti"
    return "unknown", "default"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- MR series

def load_series(slices: list[Dataset], to_axial: bool = True) -> Volume:
    """Stack single-frame MR slices into a Volume (rescale slope/intercept applied)."""
    if any(int(ds.get("NumberOfFrames", 1)) > 1 for ds in slices):
        raise ValueError("multi-frame MR not supported yet")
    iop = [float(v) for v in slices[0].ImageOrientationPatient]
    pos = [[float(v) for v in ds.ImagePositionPatient] for ds in slices]
    order = slice_order(iop, pos)
    slices = [slices[i] for i in order]
    pos = [pos[i] for i in order]
    data = np.stack([
        ds.pixel_array.astype(np.float32) * float(ds.get("RescaleSlope", 1) or 1)
        + float(ds.get("RescaleIntercept", 0) or 0)
        for ds in slices
    ])
    seq, src = detect_sequence(slices[0])
    vol = Volume(
        data=data,
        affine=affine_from_planes(iop, slices[0].PixelSpacing, pos),
        sequence=seq,
        sequence_source=src,
        frame_of_reference=str(slices[0].get("FrameOfReferenceUID", "")) or None,
        series_uid=str(slices[0].SeriesInstanceUID),
        description=str(slices[0].get("SeriesDescription", "")),
    )
    if not check_uniform(pos, iop):
        vol.warnings.append("nonuniform_slices")
    if src == "default":
        vol.warnings.append("sequence_unknown")
    return reorient_to_axial(vol) if to_axial else vol


# ---------------------------------------------------------------- DICOM-SEG

@dataclass
class SegMask:
    role: str | None            # "lesion" | "ventricles" | "brain" | None (unrecognised)
    label: str                  # SegmentLabel, for warnings only
    algorithm: str              # MANUAL | AUTOMATIC | SEMIAUTOMATIC
    data: np.ndarray            # bool [k, j, i] on the SEG's own grid
    affine: np.ndarray
    frame_of_reference: str | None


_ROLE_RULES = [
    (r"tumou?r|glioma|lesion|neoplas|metasta|edema|oedema|enhanc", "lesion"),
    (r"ventric", "ventricles"),
    (r"cerebrum|whole.?brain|^brain$|brain.?mask", "brain"),
]


def segment_role(item: Dataset) -> str | None:
    words = [str(item.get("SegmentLabel", "")), str(item.get("SegmentDescription", ""))]
    for seq in ("SegmentedPropertyTypeCodeSequence",):
        if seq in item and len(item[seq].value):
            words.append(str(item[seq].value[0].get("CodeMeaning", "")))
    text = " ".join(words).lower()
    if "cavity" in text or "resection" in text:
        return None  # previous_resection_cavity etc. are not a layer
    for pattern, role in _ROLE_RULES:
        if re.search(pattern, text):
            return role
    return None


def doctor_labels_on(vol: Volume, masks: list[SegMask],
                     allow_frame_mismatch: bool = False) -> tuple[dict[str, np.ndarray], list[str]]:
    """Map doctor SEG masks onto `vol`'s voxel grid via patient coordinates (nearest-neighbour).
    Masks with the same role are merged. Returns ({role: bool mask}, warnings).

    `allow_frame_mismatch`: ReMIND series are co-registered but carry different FrameOfReferenceUIDs
    after anonymisation, so offline evaluation turns this on. The app keeps the strict C3 rule."""
    out: dict[str, np.ndarray] = {}
    warnings = []
    for m in masks:
        if m.role is None:
            warnings.append("seg_segment_ignored")
            continue
        if m.frame_of_reference != vol.frame_of_reference:
            if not allow_frame_mismatch:
                warnings.append("seg_frame_mismatch")
                continue
        mapped = resample_nearest(m.data.astype(np.uint8), m.affine, vol.shape, vol.affine).astype(bool)
        if not np.array_equal(m.affine, vol.affine) or m.data.shape != vol.shape:
            warnings.append("seg_resampled")
        out[m.role] = out[m.role] | mapped if m.role in out else mapped
    return out, sorted(set(warnings))


def load_seg(ds: Dataset) -> list[SegMask]:
    """One SegMask per segment. Frames may be sparse (empty slices omitted), so each frame is placed
    on the slice grid by its position along the plane normal."""
    pix = ds.pixel_array
    if pix.ndim == 2:
        pix = pix[None]
    frac = str(ds.get("SegmentationType", "BINARY")).upper() == "FRACTIONAL"
    shared = ds.SharedFunctionalGroupsSequence[0] if "SharedFunctionalGroupsSequence" in ds else Dataset()
    per_frame = ds.PerFrameFunctionalGroupsSequence

    def group(fg, name):
        if name in fg:
            return fg[name].value[0]
        if name in shared:
            return shared[name].value[0]
        raise ValueError(f"SEG frame lacks {name}")

    by_segment: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    iop = spacing = gap = None
    for f, fg in enumerate(per_frame):
        seg_no = int(group(fg, "SegmentIdentificationSequence").ReferencedSegmentNumber)
        iop = [float(v) for v in group(fg, "PlaneOrientationSequence").ImageOrientationPatient]
        pm = group(fg, "PixelMeasuresSequence")
        spacing = [float(v) for v in pm.PixelSpacing]
        gap = _num(pm.get("SpacingBetweenSlices")) or _num(pm.get("SliceThickness"))
        ipp = np.array([float(v) for v in group(fg, "PlanePositionSequence").ImagePositionPatient])
        frame = pix[f] > (pix.max() / 2 if frac else 0)
        by_segment[seg_no].append((ipp, frame))

    items = {int(it.SegmentNumber): it for it in ds.SegmentSequence}
    normal = np.cross(iop[:3], iop[3:])
    out = []
    for seg_no, frames in sorted(by_segment.items()):
        d = np.array([ipp @ normal for ipp, _ in frames])
        step = gap or (float(np.min(np.diff(np.unique(d)))) if len(d) > 1 else 1.0)
        first = int(np.argmin(d))
        index = np.rint((d - d[first]) / step).astype(int)
        data = np.zeros((index.max() + 1,) + frames[0][1].shape, dtype=bool)
        for (_, frame), k in zip(frames, index):
            data[k] |= frame
        affine = affine_from_planes(iop, spacing, [frames[first][0]])
        affine[:3, 0] = normal / np.linalg.norm(normal) * step
        item = items.get(seg_no, Dataset())
        out.append(SegMask(
            role=segment_role(item),
            label=str(item.get("SegmentLabel", f"segment {seg_no}")),
            algorithm=str(item.get("SegmentAlgorithmType", "")),
            data=data,
            affine=affine,
            frame_of_reference=str(ds.get("FrameOfReferenceUID", "")) or None,
        ))
    return out
