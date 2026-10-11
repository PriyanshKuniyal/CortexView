"""3D volume + geometry helpers.

A Volume is a 3D array indexed [k, j, i] = [slice, row, column] plus a 4x4 affine that maps
(k, j, i, 1) to DICOM patient coordinates in mm (LPS: +x = patient Left, +y = Posterior, +z = Superior).
Everything here is lossless (permute/flip) or nearest-neighbour; nothing interpolates intensities.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Volume:
    data: np.ndarray                      # [k, j, i]
    affine: np.ndarray                    # 4x4, index -> patient LPS mm
    sequence: str = "unknown"             # T1_pre | T1_post | T2 | FLAIR | unknown
    sequence_source: str = "default"      # header | tr_te_ti | default | dataset
    frame_of_reference: str | None = None
    series_uid: str | None = None
    description: str = ""                 # SeriesDescription; never logged
    acquired_plane: str | None = None     # plane of the original slices, before reorient_to_axial
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def spacing(self) -> np.ndarray:
        """Voxel size in mm along [k, j, i]."""
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def plane(self) -> str:
        """Acquisition plane of axis 0 (the slice axis)."""
        normal = np.abs(self.affine[:3, 0])
        return ("sagittal", "coronal", "axial")[int(np.argmax(normal))]

    @property
    def patient_left(self) -> str | None:
        """Which image side is the patient's left, when slices are shown with column 0 on the left."""
        x = self.affine[0, 2]  # x component of the column direction
        if abs(x) < 1e-3:
            return None
        return "image_right" if x > 0 else "image_left"


def affine_from_planes(iop, pixel_spacing, positions) -> np.ndarray:
    """Build the index->patient affine from DICOM ImageOrientationPatient, PixelSpacing and the
    sorted ImagePositionPatient of every slice."""
    iop = np.asarray(iop, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]           # row_dir: along a row (column index i grows)
    dr, dc = float(pixel_spacing[0]), float(pixel_spacing[1])
    positions = np.asarray(positions, dtype=float)
    a = np.eye(4)
    if len(positions) > 1:
        a[:3, 0] = (positions[-1] - positions[0]) / (len(positions) - 1)
    else:
        a[:3, 0] = np.cross(row_dir, col_dir)     # unit thickness for a single slice
    a[:3, 1] = col_dir * dr
    a[:3, 2] = row_dir * dc
    a[:3, 3] = positions[0]
    return a


def slice_order(iop, positions) -> np.ndarray:
    """Indices that sort slices along the plane normal."""
    iop = np.asarray(iop, dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    return np.argsort(np.asarray(positions, dtype=float) @ normal, kind="stable")


def check_uniform(positions, iop, tol_mm: float = 0.05) -> bool:
    iop = np.asarray(iop, dtype=float)
    d = np.diff(np.asarray(positions, dtype=float) @ np.cross(iop[:3], iop[3:]))
    return len(d) == 0 or float(np.ptp(d)) <= tol_mm


def reorient_to_axial(vol: Volume) -> Volume:
    """Permute/flip axes (no resampling) so that axis 0 runs inferior->superior, rows run
    anterior->posterior (anterior at the top) and columns run patient right->left (radiological view).
    Oblique scans keep their in-plane tilt."""
    vol = replace(vol, acquired_plane=vol.acquired_plane or vol.plane)
    dirs = vol.affine[:3, :3]
    dominant = np.argmax(np.abs(dirs), axis=0)    # patient axis (0=x,1=y,2=z) of each array axis
    if sorted(dominant.tolist()) != [0, 1, 2]:
        return replace(vol, warnings=vol.warnings + ["orientation_ambiguous"])
    want = [2, 1, 0]                               # new axis n should follow patient axis want[n]
    perm = [int(np.where(dominant == p)[0][0]) for p in want]
    data = np.transpose(vol.data, perm)
    aff = np.eye(4)
    aff[:3, :3] = vol.affine[:3, :3][:, perm]
    aff[:3, 3] = vol.affine[:3, 3]
    for n, p in enumerate(want):
        if aff[p, n] < 0:
            data = np.flip(data, axis=n)
            aff[:3, 3] = aff[:3, 3] + aff[:3, n] * (data.shape[n] - 1)
            aff[:3, n] = -aff[:3, n]
    return replace(vol, data=np.ascontiguousarray(data), affine=aff)


def resample_nearest(src: np.ndarray, src_affine: np.ndarray,
                     shape: tuple[int, int, int], affine: np.ndarray, fill=0) -> np.ndarray:
    """Nearest-neighbour resample `src` onto a target grid, using patient coordinates."""
    out = np.full(shape, fill, dtype=src.dtype)
    m = np.linalg.inv(src_affine) @ affine        # target index -> source index
    jj, ii = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    base = m[:3, 1:2] * jj.ravel() + m[:3, 2:3] * ii.ravel() + m[:3, 3:4]
    lim = np.array(src.shape)[:, None]
    for k in range(shape[0]):
        idx = np.rint(base + m[:3, 0:1] * k).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < lim), axis=0)
        flat = np.zeros(idx.shape[1], dtype=src.dtype) + fill
        flat[ok] = src[idx[0, ok], idx[1, ok], idx[2, ok]]
        out[k] = flat.reshape(shape[1], shape[2])
    return out
