"""CortexView shared contracts, as Pydantic models (docs/contracts.md).

Mirrored by hand in frontend/src/cortexview/api/types.ts. Change both together.
So far only C4 (MetricsReport) is here.
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
