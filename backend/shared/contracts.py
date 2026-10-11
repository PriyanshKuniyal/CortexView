"""CortexView shared contracts, as Pydantic models (docs/contracts.md).

Mirrored by hand in frontend/src/cortexview/api/types.ts. Change both together.
C3 (API) and C4 (MetricsReport) are here.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Confidence = Literal["high", "medium", "low", "empty"]
Hemisphere = Literal["left", "right", "both", "midline"]


class Intensity(BaseModel):
    """On the 0-1 windowed slice; relative within one scan only (MRI brightness has no unit)."""
    mean: float
    std: float
    ratio_to_brain: float | None          # mean / mean of gray + white matter; null for skull


class LayerMetrics(BaseModel):
    present: bool
    area_px: int
    area_mm2: float | None                # null if spacing unknown
    pct_intracranial: float | None        # null for skull (outside the intracranial space)
    n_components: int                     # separate blobs (8-connected)
    centroid_px: tuple[float, float] | None              # [x, y]
    bbox_px: tuple[int, int, int, int] | None            # [x0, y0, x1, y1], inclusive
    max_diameter_mm: float | None         # lesion only
    volume_ml: float | None               # only when the whole 3D mask is known
    intensity: Intensity | None           # null when the layer is empty
    hemisphere: Hemisphere | None         # null if orientation unknown
    anchor_px: tuple[int, int] | None     # [x, y], the point deepest inside the largest blob
    confidence: Confidence
    warnings: list[str]


class TextItem(BaseModel):
    layer: str
    template_id: str                      # shared/text_templates.json
    text: str


class MetricsReport(BaseModel):
    intracranial_area_mm2: float | None   # brain + CSF + lesion; the denominator for %
    layers: dict[str, LayerMetrics]       # one per layers.json entry, in peel order
    text: list[TextItem]


# ------------------------------------------------------------------------------------------ C3 API

Sequence = Literal["T1_pre", "T1_post", "T2", "FLAIR", "unknown"]
RenderBackend = Literal["local-lora-v1", "workers-klein-4b", "workers-klein-9b", "workers-flux2-dev",
                        "flat-fallback", "cache"]


class SliceInfo(BaseModel):
    width: int
    height: int
    mri_png: str                          # data URL, 8-bit greyscale, windowed inside the head
    slice_index: int                      # 0-based
    n_slices: int
    plane: str
    pixel_spacing_mm: tuple[float, float] | None   # [row, col]
    sequence: Sequence
    sequence_source: str
    patient_left: Literal["image_left", "image_right"] | None


class LayerResult(BaseModel):
    id: str
    source: Literal["auto", "doctor"]
    confidence: Confidence
    flat_png: str                         # data URL, RGBA: layer colour inside the mask, transparent outside
    illustrated_png: str | None = None    # filled in by GET /api/render


class ApiWarning(BaseModel):
    code: str
    message: str


class RenderRef(BaseModel):
    status: Literal["pending", "done", "failed"]
    poll_url: str | None = None
    poll_interval_ms: int | None = None


class AnalyzeResponse(BaseModel):
    job_id: str
    disclaimer: str
    slice: SliceInfo
    layers: list[LayerResult]             # one per layers.json entry, in peel order
    metrics: MetricsReport
    render: RenderRef
    warnings: list[ApiWarning]


class Fidelity(BaseModel):
    mean_iou: float
    phantom_lesion_pct: float
    passed: bool


class IllustratedLayer(BaseModel):
    id: str
    illustrated_png: str


class RenderPending(BaseModel):
    status: Literal["pending"] = "pending"


class RenderDone(BaseModel):
    status: Literal["done"] = "done"
    backend: RenderBackend
    fidelity: Fidelity | None
    layers: list[IllustratedLayer]


class RenderFailed(BaseModel):
    status: Literal["failed"] = "failed"
    reason: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorBody(BaseModel):
    error: ErrorDetail
