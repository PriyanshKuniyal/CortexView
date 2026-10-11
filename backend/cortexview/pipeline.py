"""Contract C3 analysis flow, from uploaded DICOM files to one segmented axial slice.

    files -> scan headers -> choose series (C3 rule) -> load pixels -> doctor labels (same frame only)
          -> [lesion from a FLAIR in the same frame, if no doctor lesion] -> segment (3D)
          -> pick slice (C3 rule or the caller's) -> Result

This is what POST /api/analyze calls. It is stateless, does no logging, and never puts header values or
filenames into messages (C3). Metrics (C4: `metrics_engine.report(result)`) and rendering (C5) are separate steps
that consume Result.
"""
from __future__ import annotations

import io as _io
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pydicom

from . import dicom_io as io
from .segment import Params, Segmentation, analyze
from .volume import Volume, resample_nearest

MIN_3D_SLICES = 20       # fewer slices than this: segment the chosen slice on its own (2D)


def _segmenter():
    """Choose the primary AI segmenter unless classical mode was explicitly requested."""
    if os.environ.get("CORTEXVIEW_SEGMENTER", "ai").strip().lower() != "classical":
        from .segment_ai import T1, analyze as ai, available
        return ai, available()[0], T1
    return analyze, False, set()


class AnalyzeError(Exception):
    """An error with a C3 error code and HTTP status."""

    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code, self.status, self.message = code, status, message


@dataclass
class Result:
    volume: Volume                    # the chosen series, axial
    segmentation: Segmentation        # whole volume (3D) or the single slice (2D fallback)
    slice: Segmentation               # the chosen slice: masks, confidence, source
    slice_index: int
    warnings: list[str] = field(default_factory=list)

    @property
    def image(self) -> np.ndarray:
        return self.volume.data[self.slice_index]


MAX_UNZIPPED_BYTES = 1_000_000_000      # zip-bomb guard (the upload itself is capped at 300 MB by C3)


def expand_zips(sources, max_bytes: int = MAX_UNZIPPED_BYTES) -> list:
    """Replace any zip archive among `sources` by its members (in memory). Other sources pass through."""
    out, total = [], 0
    for src in sources:
        members = None
        if zipfile.is_zipfile(src):
            if hasattr(src, "seek"):
                src.seek(0)
            try:
                with zipfile.ZipFile(src) as zf:
                    members = []
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        total += info.file_size
                        if total > max_bytes:
                            raise AnalyzeError("too_large", 413, "The unpacked upload is too large.")
                        members.append(_io.BytesIO(zf.read(info)))
            except (zipfile.BadZipFile, OSError):
                members = None              # looked like a zip but is not one: treat as a plain file
        if hasattr(src, "seek"):
            src.seek(0)
        out.extend(members if members is not None else [src])
    return out


def run(sources, slice_index: int | None = None, series_uid: str | None = None,
        params: Params = Params()) -> Result:
    """`sources`: paths or binary file objects: .dcm files and/or zip archives of them."""
    series, seg_infos, n_bad = io.scan(expand_zips(sources))
    if not series:
        if n_bad and not seg_infos:
            raise AnalyzeError("not_dicom", 422, "None of the files could be read as DICOM.")
        raise AnalyzeError("not_mr", 422, "No MR image series was found in the upload.")
    if series_uid is not None:
        info = series.get(series_uid)
        if info is None:
            raise AnalyzeError("no_axial_series", 422, "The requested series is not in the upload.")
    else:
        info = io.choose_series(series, seg_infos)
    if info is None or not info.axial_ok:
        raise AnalyzeError("no_axial_series", 422, "No series can be shown as axial slices.")

    vol = io.load_info(info)
    warnings = list(vol.warnings)
    same_frame = [g for g in seg_infos if g.frame_of_reference == vol.frame_of_reference]
    if len(same_frame) < len(seg_infos):
        warnings.append("seg_frame_mismatch")        # drawn on a scan in another coordinate frame
    masks = [m for g in same_frame for m in io.load_seg(pydicom.dcmread(g.source))]
    doctor, w = io.doctor_labels_on(vol, masks)
    warnings += w

    if slice_index is not None and not 0 <= slice_index < vol.shape[0]:
        raise AnalyzeError("bad_slice_index", 422, f"slice_index must be between 0 and {vol.shape[0] - 1}.")

    seg_fn, ai_ready, ai_sequences = _segmenter()
    ai_runs = ai_ready and vol.sequence in ai_sequences and vol.shape[0] >= MIN_3D_SLICES
    hint = None
    if "lesion" not in doctor and vol.sequence != "FLAIR" and not ai_runs:
        hint = _lesion_from_flair(series, info, vol, params)

    if vol.shape[0] < MIN_3D_SLICES:
        warnings.append("low_slice_count")
        k = vol.shape[0] // 2 if slice_index is None else slice_index
        if slice_index is None and "lesion" in doctor and doctor["lesion"].any():
            k = int(np.argmax(doctor["lesion"].reshape(vol.shape[0], -1).sum(1)))
        seg2d = seg_fn(vol.data[k], vol.spacing[1:], vol.sequence,
                        {r: m[k] for r, m in doctor.items()}, params,
                        lesion_hint=None if hint is None else hint[k])
        return Result(vol, seg2d, seg2d, k, sorted(set(warnings + seg2d.warnings)))

    seg = seg_fn(vol.data, vol.spacing, vol.sequence, doctor, params,
                  tissue_slices="auto" if slice_index is None else slice_index, lesion_hint=hint)
    k = seg.slice_index if slice_index is None else slice_index
    internal = {"tissue_partial"}                 # only the shown slice needs CSF/GM/WM: not news to a user
    return Result(vol, seg, seg.cut(k), k, sorted(set(warnings + seg.warnings) - internal))


def _lesion_from_flair(series, shown, vol: Volume, params: Params):
    """Automatic lesion found on a FLAIR series of the same scan (same coordinate frame), mapped onto
    the shown series. None when there is no such FLAIR or it finds nothing."""
    flairs = [s for s in series.values() if s.sequence == "FLAIR" and s.uid != shown.uid
              and s.frame_of_reference and s.frame_of_reference == shown.frame_of_reference]
    if not flairs:
        return None
    best = sorted(flairs, key=lambda s: (-len(s.sources), float(np.prod(s.spacing_mm)), s.uid))[0]
    fv = io.load_info(best)
    if fv.shape[0] < MIN_3D_SLICES:
        return None
    fseg = analyze(fv.data, fv.spacing, "FLAIR", p=params, tissue_slices=0)
    lesion = fseg.masks["lesion"]
    if not lesion.any():
        return None
    return resample_nearest(lesion.astype(np.uint8), fv.affine, vol.shape, vol.affine).astype(bool)
