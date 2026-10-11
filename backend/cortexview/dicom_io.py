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
            out.append(pydicom.dcmread(p if hasattr(p, "read") else str(p)))
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        if hasattr(p, "seek"):
            p.seek(0)
    return out


def read_folder(folder) -> list[Dataset]:
    return read_datasets(sorted(Path(folder).rglob("*.dcm")))


def is_seg(ds: Dataset) -> bool:
    return str(ds.get("SOPClassUID", "")) == SEG_SOP_CLASS or ds.get("Modality") == "SEG"


@dataclass
class SeriesInfo:
    """Header-only summary of one MR series (no pixels decoded)."""
    uid: str
    sources: list                     # paths or file-like objects, one per slice
    sequence: str
    sequence_source: str
    acquired_plane: str               # axial | coronal | sagittal (from the slice orientation)
    spacing_mm: tuple                 # (slice gap, row, column) approx.
    frame_of_reference: str | None

    @property
    def axial_ok(self) -> bool:
        """Can be shown as axial slices without resampling: acquired axial, or a 3D series whose voxels
        are close to cubes (it is then just re-stacked)."""
        sp = np.asarray(self.spacing_mm, float)
        return self.acquired_plane == "axial" or (len(self.sources) >= 40 and sp.max() / sp.min() <= 1.5)


@dataclass
class SegInfo:
    source: object
    frame_of_reference: str | None
    roles: list                       # role of each segment ("lesion", "ventricles", "brain", None)
    referenced_series: list           # SeriesInstanceUIDs the SEG says it was drawn on (may be empty)


def scan(sources) -> tuple[dict[str, SeriesInfo], list[SegInfo], int]:
    """Read headers only (fast) and group into MR series and SEG files.
    Returns ({series uid: SeriesInfo}, [SegInfo], number of non-DICOM / unusable files)."""
    by_uid: dict[str, list] = defaultdict(list)
    first: dict[str, Dataset] = {}
    segs, bad = [], 0
    seen: set[str] = set()
    for src in sources:
        try:
            ds = pydicom.dcmread(src, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError, ValueError, KeyError):
            bad += 1
            continue
        if hasattr(src, "seek"):
            src.seek(0)
        sop = str(ds.get("SOPInstanceUID", ""))
        if sop:
            if sop in seen:
                continue            # the same file uploaded twice (e.g. a folder and its zip): use it once
            seen.add(sop)
        if is_seg(ds):
            items = ds.get("SegmentSequence", [])
            refs = [str(r.SeriesInstanceUID) for r in ds.get("ReferencedSeriesSequence", [])
                    if "SeriesInstanceUID" in r]
            segs.append(SegInfo(src, str(ds.get("FrameOfReferenceUID", "")) or None,
                                [segment_role(it) for it in items], refs))
        elif ds.get("Modality") == "MR" and "ImageOrientationPatient" in ds and "PixelSpacing" in ds:
            uid = str(ds.SeriesInstanceUID)
            by_uid[uid].append(src)
            first.setdefault(uid, ds)
        else:
            bad += 1
    series = {}
    for uid, srcs in by_uid.items():
        ds = first[uid]
        iop = np.array([float(v) for v in ds.ImageOrientationPatient])
        normal = np.abs(np.cross(iop[:3], iop[3:]))
        plane = ("sagittal", "coronal", "axial")[int(np.argmax(normal))]
        gap = _num(ds.get("SpacingBetweenSlices")) or _num(ds.get("SliceThickness")) or 1.0
        seq, seq_src = detect_sequence(ds)
        series[uid] = SeriesInfo(uid, srcs, seq, seq_src, plane,
                                 (gap, float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])),
                                 str(ds.get("FrameOfReferenceUID", "")) or None)
    return series, segs, bad


SEQUENCE_PREFERENCE = ("T1_post", "FLAIR", "T2")


def choose_series(series: dict[str, SeriesInfo], segs: list[SegInfo]) -> SeriesInfo | None:
    """Contract C3 auto choice: the axial series a doctor tumour label was drawn on, else axial
    T1-post, then FLAIR, then T2, then any axial MR. Ties: more slices first, then finer voxels."""
    cands = [s for s in series.values() if s.axial_ok]
    if not cands:
        return None

    def finest(group):
        return sorted(group, key=lambda s: (-len(s.sources), float(np.prod(s.spacing_mm)), s.uid))[0]

    def preferred(group):
        for seq in SEQUENCE_PREFERENCE:
            hit = [s for s in group if s.sequence == seq]
            if hit:
                return finest(hit)
        return finest(group)

    tumour = [g for g in segs if "lesion" in g.roles]
    for g in tumour:
        hit = [s for s in cands if s.uid in g.referenced_series]
        if hit:
            return preferred(hit)
    # no explicit reference: any series in the label's coordinate frame can show it
    frames = {g.frame_of_reference for g in tumour if g.frame_of_reference}
    hit = [s for s in cands if s.frame_of_reference in frames]
    return preferred(hit or cands)


def load_info(info: SeriesInfo, to_axial: bool = True) -> Volume:
    """Decode the pixels of one scanned series."""
    return load_series(read_datasets(info.sources), to_axial=to_axial)


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
    suspect: bool = False       # frames look sheared/wrapped: the file probably decodes wrongly


def looks_sheared(data: np.ndarray, affine: np.ndarray) -> bool:
    """Some DICOM-SEG writers pack 1-bit frames in a way that, read per the standard, shifts every frame
    sideways and wraps it around the frame edge (seen in several ReMIND files whose frame size is not a
    multiple of 8 bits). Signature: the mask's centre drifts more than 1.5 mm sideways per 1.5 mm of depth
    AND over 20% of frames touch both the left and right frame edges. Real anatomy does neither."""
    gap = float(np.linalg.norm(affine[:3, 0]))
    col_mm = float(np.linalg.norm(affine[:3, 2]))
    area = data.reshape(data.shape[0], -1).sum(1)
    ok = np.where(area > max(20, 0.05 * area.max()))[0] if area.max() else []
    if len(ok) < 6:
        return False
    step = max(1, int(round(1.5 / max(gap, 1e-6))))
    cx = {int(k): float(np.nonzero(data[k])[1].mean()) * col_mm for k in ok}
    shifts = [abs(cx[k + step] - cx[k]) for k in cx if k + step in cx]
    if len(shifts) < 3:
        return False
    wrap = np.mean([data[k][:, :2].any() and data[k][:, -2:].any() for k in ok])
    return float(np.median(shifts)) > 1.5 and float(wrap) > 0.2


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


def doctor_labels_on(vol: Volume, masks: list[SegMask], allow_frame_mismatch: bool = False,
                     allow_suspect: bool = False) -> tuple[dict[str, np.ndarray], list[str]]:
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
        if m.suspect and not allow_suspect:
            warnings.append("seg_decode_suspect")       # would paint a garbled label: not used
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


def _seg_frame_starts(n_frames: int, n_px: int, carry: int) -> np.ndarray:
    """Bit offset of each frame. carry = n_px % 8 is the standard layout (frames packed back to back);
    a smaller carry models the dcmqi writer bug (from scripts/synthseg/remind_synthseg.py)."""
    k = np.arange(n_frames)
    return 8 * (k * (n_px // 8) + (k * carry) // 8) + (k * (n_px % 8)) % 8


def binary_seg_frames(ds: Dataset) -> np.ndarray | None:
    """Unpack an uncompressed BINARY SEG's frames [F, R, C], repairing the dcmqi bit-packing bug.

    Some ReMIND SEGs (QIICR dcmqi, SoftwareVersions ef9e227) are packed wrongly when Rows*Columns % 8 is
    3, 5, 6 or 7; read per the standard, every frame comes out shifted sideways and wrapped around. Both
    layouts are decoded and the one with the smaller surface (a real outline is smooth, a sheared one is
    ragged) wins; ties go to the repaired layout. None if the pixel data is compressed."""
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        return None
    F, R, C = int(ds.get("NumberOfFrames", 1)), int(ds.Rows), int(ds.Columns)
    N, r = R * C, (R * C) % 8
    bits = np.unpackbits(np.frombuffer(ds.PixelData, np.uint8), bitorder="little")

    def decode(carry):
        out = np.zeros((F, N), np.uint8)
        for i, s in enumerate(_seg_frame_starts(F, N, carry)):
            x = bits[s:s + N]
            out[i, :len(x)] = x
        return out.reshape(F, R, C)

    def surface(a):
        a = np.pad(a, 1).astype(np.int8)
        return sum(int(np.abs(np.diff(a, axis=ax)).sum()) for ax in range(3))

    bug = r & -r                                  # lowest set bit of r
    if bug == r:                                  # r in {0, 1, 2, 4}: both layouts are the same
        return decode(r)
    std, fix = decode(r), decode(bug)
    return fix if surface(fix) <= surface(std) else std


def load_seg(ds: Dataset) -> list[SegMask]:
    """One SegMask per segment. Frames may be sparse (empty slices omitted), so each frame is placed
    on the slice grid by its position along the plane normal."""
    frac = str(ds.get("SegmentationType", "BINARY")).upper() == "FRACTIONAL"
    pix = None if frac else binary_seg_frames(ds)
    if pix is None:
        pix = ds.pixel_array
    if pix.ndim == 2:
        pix = pix[None]
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
            suspect=looks_sheared(data, affine),
        ))
    return out
