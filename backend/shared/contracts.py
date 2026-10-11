"""API contracts bundled with the Vercel backend deployment."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Confidence = Literal["high", "medium", "low", "empty"]


class Intensity(BaseModel):
    mean: float
    std: float
    ratio_to_brain: float | None


class LayerMetrics(BaseModel):
    present: bool
    area_px: int
    area_mm2: float | None
    pct_intracranial: float | None
    n_components: int
    centroid_px: tuple[float, float] | None
    bbox_px: tuple[int, int, int, int] | None
    max_diameter_mm: float | None
    volume_ml: float | None
    intensity: Intensity | None
    hemisphere: Literal["left", "right", "both", "midline"] | None
    anchor_px: tuple[int, int] | None
    confidence: Confidence
    warnings: list[str]


class TextItem(BaseModel):
    layer: str
    template_id: str
    text: str


class MetricsReport(BaseModel):
    intracranial_area_mm2: float | None
    layers: dict[str, LayerMetrics]
    text: list[TextItem]


class SliceInfo(BaseModel):
    width: int
    height: int
    mri_png: str
    slice_index: int
    n_slices: int
    plane: str
    pixel_spacing_mm: tuple[float, float] | None
    sequence: Literal["T1_pre", "T1_post", "T2", "FLAIR", "unknown"]
    sequence_source: str
    patient_left: Literal["image_left", "image_right"] | None


class LayerResult(BaseModel):
    id: str
    source: Literal["auto", "doctor"]
    confidence: Confidence
    flat_png: str
    illustrated_png: str | None = None


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
    layers: list[LayerResult]
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
    backend: Literal["local-lora-v1", "workers-klein-4b", "workers-klein-9b", "workers-flux2-dev", "flat-fallback", "cache"]
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