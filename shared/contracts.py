"""
shared/contracts.py

Data contracts for CortexView.
Defines Pydantic models for Contract C4 (MetricsReport).
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class IntensityMetrics(BaseModel):
    """Pixel intensity statistics for a layer on the active slice."""
    mean: Optional[float] = None
    std: Optional[float] = None
    ratio_to_brain: Optional[float] = None


class LayerMetrics(BaseModel):
    """Quantitative measurements for a single anatomical or finding layer."""
    layer_id: str
    present: bool
    confidence: str  # "high" | "medium" | "low" | "empty"
    warnings: List[str] = Field(default_factory=list)
    
    # Measurements on active 2D slice
    area_px: int
    area_mm2: Optional[float] = None
    pct_intracranial: Optional[float] = None
    n_components: int
    centroid_px: Optional[Tuple[float, float]] = None    # [x, y]
    bbox_px: Optional[Tuple[int, int, int, int]] = None  # [x0, y0, x1, y1] inclusive
    anchor_px: Optional[Tuple[int, int]] = None          # [x, y] pin coordinate
    
    # Lesion-specific & orientation metrics
    max_diameter_mm: Optional[float] = None              # Lesion only
    hemisphere: Optional[str] = None                     # "left" | "right" | "midline" | "both"
    
    # 3D whole-volume measurement
    volume_ml: Optional[float] = None                    # Total volume in mL
    
    # Intensity profiling
    intensity: IntensityMetrics


class MetricsReport(BaseModel):
    """Complete Contract C4 response schema returned by the metrics engine."""
    version: int = 1
    slice_index: int
    n_slices: int
    spacing_mm: Optional[Tuple[float, float, float]] = None  # (dz, dy, dx) in mm
    intracranial_area_mm2: Optional[float] = None
    layers: Dict[str, LayerMetrics]