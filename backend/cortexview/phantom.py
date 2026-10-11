"""Synthetic head phantom with a known answer, for tests and offline demos (TASKS §3, P1).

A nested-ellipsoid "head": scalp fat, skull bone, outer CSF, a folded gray-matter ribbon, white matter,
two ventricles and an optional lesion. Brightness per tissue follows the sequence (T1 / T2 / FLAIR), plus
a smooth coil-shading gradient and noise from a fixed seed, so every call returns the same pixels.

It can be written out as a real DICOM MR series and a DICOM-SEG, so the whole pipeline
(dicom_io -> segment) can be tested end to end. All identifiers are fake.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from .layers import label_of

MR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.4"
SEG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.66.4"
UID_PREFIX = "1.2.826.0.1.3680043.10.1497."          # fixed prefix: generated UIDs are reproducible

# relative brightness of each tissue per sequence (deep white matter = 1.0 on T1)
INTENSITY = {
    "T1_post": {"air": 0.0, "fat": 1.7, "bone": 0.06, "csf": 0.32, "gm": 0.78, "wm": 1.0, "lesion": 0.45},
    "T1_pre": {"air": 0.0, "fat": 1.7, "bone": 0.06, "csf": 0.32, "gm": 0.78, "wm": 1.0, "lesion": 0.55},
    "T2": {"air": 0.0, "fat": 1.2, "bone": 0.06, "csf": 2.0, "gm": 1.15, "wm": 0.8, "lesion": 1.6},
    "FLAIR": {"air": 0.0, "fat": 1.0, "bone": 0.06, "csf": 0.2, "gm": 1.08, "wm": 0.9, "lesion": 1.9},
}
DESCRIPTION = {"T1_post": "PHANTOM_AX_T1_postcontrast", "T1_pre": "PHANTOM_AX_T1_precontrast",
               "T2": "PHANTOM_AX_T2", "FLAIR": "PHANTOM_AX_T2_FLAIR"}


def make_phantom(shape=(100, 140, 120), spacing=(1.5, 1.5, 1.5), sequence: str = "T1_post",
                 lesion: bool = True, noise: float = 0.03, seed: int = 0):
    """Returns (image float32 [k, j, i], truth label map uint8 with C1 labels, info dict).
    Axis 0 runs inferior -> superior, rows anterior -> posterior, columns patient right -> left."""
    sp = np.asarray(spacing, float)
    c = (np.asarray(shape) - 1) / 2.0
    k, j, i = np.indices(shape, dtype=np.float32)
    z, y, x = (k - c[0]) * sp[0], (j - c[1]) * sp[1], (i - c[2]) * sp[2]

    def ell(a, b, cc, dz=0.0, dy=0.0, dx=0.0):
        return ((z - dz) / a) ** 2 + ((y - dy) / b) ** 2 + ((x - dx) / cc) ** 2

    # head radii (mm): superior-inferior, anterior-posterior, left-right
    a, b, cc = 0.47 * shape[0] * sp[0], 0.46 * shape[1] * sp[1], 0.46 * shape[2] * sp[2]
    head = ell(a, b, cc) <= 1
    scale = lambda t: ell(a - t, b - t, cc - t) <= 1          # shell `t` mm inside the head surface
    inside_bone = scale(6.0)                                  # 6 mm of scalp fat
    intracranial = scale(12.0)                                # 6 mm of bone
    brain = scale(14.5)                                       # 2.5 mm of outer CSF

    # folded cortex: the GM/WM boundary sits 3-6 mm under the brain surface, wavy like gyri
    r = np.sqrt(ell(a - 14.5, b - 14.5, cc - 14.5))
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    fold = 0.5 + 0.5 * np.sin(7 * theta) * np.cos(5 * phi)
    wm = brain & (r <= 1 - (3.0 + 3.0 * fold) / (a - 14.5))
    # sulci: thin CSF clefts cut into the cortex where the folds dip deepest
    sulci = brain & ~wm & (fold > 0.85) & (r > 1 - 5.0 / (a - 14.5))

    # ventricles: two small ellipsoids either side of the midline
    vent = (ell(14, 22, 6, dz=4, dy=-4, dx=-8) <= 1) | (ell(14, 22, 6, dz=4, dy=-4, dx=8) <= 1)

    labels = np.zeros(shape, np.uint8)
    labels[head] = label_of("skull")
    labels[intracranial] = label_of("csf")
    labels[brain] = label_of("gray_matter")
    labels[wm] = label_of("white_matter")
    labels[sulci | vent] = label_of("csf")
    les = np.zeros(shape, bool)
    if lesion:
        les = ell(11, 11, 11, dz=8, dy=-10, dx=-0.22 * cc) <= 1   # in the patient's right hemisphere
        labels[les] = label_of("lesion")

    t = INTENSITY[sequence]
    img = np.zeros(shape, np.float32)
    img[head] = t["fat"]
    img[inside_bone] = t["bone"]
    img[intracranial] = t["csf"]
    img[brain] = t["gm"]
    img[wm] = t["wm"]
    img[sulci | vent] = t["csf"]
    img[les] = t["lesion"]
    # slight partial-volume blur, coil shading (+-8% front to back), noise from a fixed seed
    from scipy import ndimage as ndi
    img = ndi.gaussian_filter(img, 0.6)
    img *= (1 + 0.08 * y / (b + 1e-6)).astype(np.float32)
    rng = np.random.default_rng(seed)
    img += rng.normal(0, noise, shape).astype(np.float32) * head
    img = np.clip(img, 0, None) * 1000
    info = {"spacing": tuple(float(s) for s in sp), "sequence": sequence, "lesion": les}
    return img.astype(np.float32), labels, info


def _uid(*parts) -> str:
    return generate_uid(prefix=UID_PREFIX, entropy_srcs=[str(p) for p in parts])


def _base(sop_class: str, sop_uid: str, modality: str) -> Dataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = UID_PREFIX + "1"
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.Modality = modality
    ds.PatientName = "PHANTOM^SYNTHETIC"
    ds.PatientID = "PHANTOM"
    ds.StudyInstanceUID = _uid("study")
    ds.StudyDate = ds.SeriesDate = _dt.date(2026, 1, 1).strftime("%Y%m%d")
    ds.FrameOfReferenceUID = _uid("frame")
    return ds


def write_series(img, spacing, out_dir, sequence: str = "T1_post", name: str = "series") -> list[Path]:
    """Write an axial MR series, one file per slice (16-bit, rescale slope 1)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    series_uid = _uid("series", name, sequence)
    data = np.clip(np.rint(img), 0, 65535).astype(np.uint16)
    # columns run patient right->left (+x), rows anterior->posterior (+y): a standard axial slice
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    paths = []
    for kk in range(data.shape[0]):
        ds = _base(MR_SOP_CLASS, _uid("mr", name, sequence, kk), "MR")
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 1
        ds.InstanceNumber = kk + 1
        ds.SeriesDescription = DESCRIPTION[sequence]
        ds.ImageOrientationPatient = iop
        ds.ImagePositionPatient = [0.0, 0.0, float(kk * spacing[0])]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[2])]
        ds.SliceThickness = float(spacing[0])
        ds.SpacingBetweenSlices = float(spacing[0])
        ds.Rows, ds.Columns = data.shape[1], data.shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        if sequence == "T1_post":
            ds.ContrastBolusAgent = "GADOLINIUM (SYNTHETIC)"
        ds.PixelData = data[kk].tobytes()
        p = out / f"{kk:04d}.dcm"
        pydicom.dcmwrite(p, ds, enforce_file_format=True)
        paths.append(p)
    return paths


def write_seg(mask, spacing, out_path, label: str = "Tumor", name: str = "series",
              sequence: str = "T1_post") -> Path:
    """Write a binary single-segment DICOM-SEG on the same grid as write_series (empty frames omitted)."""
    mask = np.asarray(mask, bool)
    frames = [kk for kk in range(mask.shape[0]) if mask[kk].any()]
    ds = _base(SEG_SOP_CLASS, _uid("seg", name, label), "SEG")
    ds.SeriesInstanceUID = _uid("segseries", name, label)
    ds.SeriesNumber = 300
    ds.InstanceNumber = 1
    ds.SeriesDescription = f"{label} seg - MR ref: {DESCRIPTION[sequence]}"
    ds.SegmentationType = "BINARY"
    ds.Rows, ds.Columns = mask.shape[1], mask.shape[2]
    ds.NumberOfFrames = len(frames)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 1
    ds.HighBit = 0
    ds.PixelRepresentation = 0
    seg = Dataset()
    seg.SegmentNumber = 1
    seg.SegmentLabel = label
    seg.SegmentDescription = label
    seg.SegmentAlgorithmType = "MANUAL"
    prop = Dataset()
    prop.CodeValue, prop.CodingSchemeDesignator, prop.CodeMeaning = "108369006", "SCT", "Tumor"
    seg.SegmentedPropertyTypeCodeSequence = [prop]
    ds.SegmentSequence = [seg]
    ref = Dataset()
    ref.SeriesInstanceUID = _uid("series", name, sequence)    # the series written by write_series
    ref.ReferencedInstanceSequence = []
    ds.ReferencedSeriesSequence = [ref]
    shared = Dataset()
    po = Dataset(); po.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pm = Dataset(); pm.PixelSpacing = [float(spacing[1]), float(spacing[2])]
    pm.SliceThickness = pm.SpacingBetweenSlices = float(spacing[0])
    shared.PlaneOrientationSequence = [po]
    shared.PixelMeasuresSequence = [pm]
    ds.SharedFunctionalGroupsSequence = [shared]
    per = []
    for kk in frames:
        fg = Dataset()
        pp = Dataset(); pp.ImagePositionPatient = [0.0, 0.0, float(kk * spacing[0])]
        si = Dataset(); si.ReferencedSegmentNumber = 1
        fg.PlanePositionSequence = [pp]
        fg.SegmentIdentificationSequence = [si]
        per.append(fg)
    ds.PerFrameFunctionalGroupsSequence = per
    bits = np.packbits(mask[frames].ravel().astype(np.uint8), bitorder="little")
    if len(bits) % 2:
        bits = np.append(bits, np.uint8(0))
    ds.PixelData = bits.tobytes()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(out, ds, enforce_file_format=True)
    return out
