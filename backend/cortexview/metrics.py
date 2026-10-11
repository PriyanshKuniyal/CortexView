"""
backend/cortexview/metrics.py
==============================
Deterministic Metrics Engine for CortexView (Track MP).

Core Philosophy:
----------------
"Generative where it helps, deterministic where it matters."
- ZERO AI / LLM logic in this module.
- All numbers, bounding boxes, pin anchors, and percentages are derived
  purely from deterministic classical image geometry and voxel math.
- Same input volume always produces byte-identical output.

Contract Adherence (docs/contracts.md):
----------------------------------------
- C1: Reads layer IDs and integer labels dynamically from shared/layers.json.
      Never hardcodes label numbers.
- C2: Operates on unpadded slice pixels in Cartesian [x, y] coordinate space
      (NumPy indices [row, col] = [y, x] are explicitly mapped to [x, y]).
- C4: Produces MetricsReport schema matching backend and frontend contracts,
      including intracranial normalization and 3D volume (volume_ml).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage
import scipy.spatial

from shared.contracts import IntensityMetrics, LayerMetrics, MetricsReport


# ==============================================================================
# [Checkpoint 2.2] Dynamic Layer Configuration Loader (Contract C1)
# ==============================================================================
def load_layers_config(config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically loads anatomical and finding layer definitions from shared/layers.json.

    Why this exists (Contract C1):
    -----------------------------
    To ensure frontend, rendering (FLUX/FT), and metrics always share the single
    source of truth for layer definitions without hardcoding label values in code:
      - 0:  background
      - 1:  skull / scalp
      - 2:  gray_matter
      - 3:  white_matter
      - 4:  csf / ventricles
      - 10: lesion (suspected lesion)

    Args:
        config_path: Optional explicit file path to shared/layers.json.
                     If None, resolves relative to repository root.

    Returns:
        Dictionary mapping layer_id (str) to layer dictionary properties:
        {
            "skull": {"id": "skull", "label": 1, "name": "Skull / scalp", ...},
            "gray_matter": {"id": "gray_matter", "label": 2, ...},
            ...
        }
    """
    if config_path is None:
        # Resolve repo root assuming path: <root>/backend/cortexview/metrics.py
        repo_root = Path(__file__).resolve().parent.parent.parent
        config_path = str(repo_root / "shared" / "layers.json")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Return a dictionary mapping layer_id to layer properties
    return {layer["id"]: layer for layer in data.get("layers", [])}


# ==============================================================================
# [Checkpoint 2.3] Main Metrics Entrypoint Signature (Contract C4)
# ==============================================================================
def compute_metrics(
    labels: np.ndarray,
    image: np.ndarray,
    slice_index: int,
    meta: Dict[str, Any],
    layer_info: Dict[str, Dict[str, Any]],
) -> MetricsReport:
    """
    Computes all deterministic spatial and intensity metrics across the 3D volume
    and active 2D slice.

    Args:
        labels: 3D uint8 array of shape (z, y, x).
                Every voxel contains exactly one integer label (0, 1, 2, 3, 4, 10).
        image: 3D float32 array of shape (z, y, x) windowed to [0.0, 1.0].
        slice_index: Integer index of the 2D slice chosen for display along z-axis.
        meta: Dictionary containing spatial calibration and DICOM orientation:
              - 'spacing_mm': Tuple (dz, dy, dx) in millimeters, or None if unknown.
              - 'patient_left': 'image_left' | 'image_right' | None.
              - 'n_slices': Total slices in volume (z dimension).
        layer_info: Upstream metadata per layer {layer_id: {"source": ..., "confidence": ...}}.

    Returns:
        MetricsReport Pydantic model adhering to Contract C4.
    """

    # ==========================================================================
    # [Checkpoint 2.4] Extract Active 2D Slice and Physical Spacing (Contract C2)
    # ==========================================================================
    # Extract the 2D displayed plane from the 3D volume:
    # NumPy array indexing convention is [z, y, x] -> slice is [y, x] = [row, col]
    slice_labels: np.ndarray = labels[slice_index]
    slice_image: np.ndarray = image[slice_index]

    n_slices: int = int(labels.shape[0])
    patient_left: Optional[str] = meta.get("patient_left")

    # Spacing resolution: (dz, dy, dx) in mm per voxel.
    # Note: dy and dx may NOT be equal (anisotropic in-plane acquisition).
    spacing = meta.get("spacing_mm")
    if spacing is not None:
        dz = float(spacing[0])
        dy = float(spacing[1])
        dx = float(spacing[2])
    else:
        dz, dy, dx = None, None, None

    # Load layer configuration dictionary (Contract C1)
    layers_config = load_layers_config()

    # ==========================================================================
    # [Checkpoint 2.5] Intracranial Baseline & Midline Calculation (Contract C4)
    # ==========================================================================
    # Definition of Intracranial Space:
    # Everything inside the cranial vault = Gray Matter (2) + White Matter (3)
    #                                       + CSF (4) + Lesion (10).
    # Background (0) and Skull/Scalp (1) are strictly EXCLUDED from intracranial space.
    intracranial_mask: np.ndarray = np.isin(slice_labels, [2, 3, 4, 10])
    intracranial_px: int = int(np.count_nonzero(intracranial_mask))

    # Intracranial area in mm²:
    # Area = pixel_count * dy * dx (null if spacing is unknown)
    if intracranial_px > 0 and dy is not None and dx is not None:
        intracranial_area_mm2: Optional[float] = round(float(intracranial_px * dy * dx), 1)
    else:
        intracranial_area_mm2 = None

    # Intracranial Midline (for Contract C4 hemisphere assignment):
    # Calculated as the column midpoint of the intracranial bounding box.
    if intracranial_px > 0:
        intra_cols = np.where(intracranial_mask)[1]
        midline_col: float = float((intra_cols.min() + intra_cols.max()) / 2.0)
    else:
        midline_col = float(slice_labels.shape[1] / 2.0)

    # ==========================================================================
    # [Checkpoint 2.6] Single-Pass 3D Whole-Volume Voxel Counting (volume_ml)
    # ==========================================================================
    # Speed constraint (< 100 ms):
    # Rather than scanning the 3D volume for each layer separately, np.bincount
    # tallies voxel counts across all 200+ slices in a single C-speed pass (~5 ms).
    # minlength=11 guarantees labels 0 through 10 have dedicated bins (max label is 10).
    voxel_counts: np.ndarray = np.bincount(labels.ravel(), minlength=11)

    # Whole-volume 3D measurements are valid only if z > 1 and physical spacing is known.
    # Single slice uploads (z == 1) or uncalibrated scans (Kaggle) return None.
    is_3d_valid: bool = (n_slices > 1) and (spacing is not None)

    # Reference mean intensity of normal brain parenchyma (Gray Matter + White Matter)
    # Used as the denominator for intensity.ratio_to_brain in Contract C4.
    brain_mask: np.ndarray = np.isin(slice_labels, [2, 3])
    if np.any(brain_mask):
        brain_mean_intensity: Optional[float] = float(np.mean(slice_image[brain_mask]))
    else:
        brain_mean_intensity = None

    # ==========================================================================
    # Per-Layer Metrics Computation Loop
    # ==========================================================================
    layer_reports: Dict[str, LayerMetrics] = {}

    for layer_id, layer_def in layers_config.items():
        label_val: int = int(layer_def["label"])
        base_confidence: str = layer_info.get(layer_id, {}).get("confidence", "high")
        warnings: List[str] = []

        # Mask for current layer on active 2D slice
        mask: np.ndarray = (slice_labels == label_val)
        area_px: int = int(np.count_nonzero(mask))

        # Check for global metadata warnings
        if spacing is None:
            warnings.append("spacing_unknown")
        if patient_left is None:
            warnings.append("orientation_unknown")

        # ======================================================================
        # [Checkpoint 2.7] Handle Empty Layers (present: false)
        # ======================================================================
        if area_px == 0:
            layer_reports[layer_id] = LayerMetrics(
                layer_id=layer_id,
                present=False,
                confidence="empty",
                warnings=warnings,
                area_px=0,
                area_mm2=None,
                volume_ml=None,
                pct_intracranial=None,
                n_components=0,
                centroid_px=None,
                bbox_px=None,
                anchor_px=None,
                max_diameter_mm=None,
                hemisphere=None,
                intensity=IntensityMetrics(mean=None, std=None, ratio_to_brain=None),
            )
            continue

        # ======================================================================
        # [Checkpoint 2.8] 2D Area in mm² and Intracranial Percentage
        # ======================================================================
        # 1. 2D Area in mm² (area_px * dy * dx)
        if dy is not None and dx is not None:
            area_mm2 = round(float(area_px * dy * dx), 1)
        else:
            area_mm2 = None

        # 2. Percentage of intracranial space:
        # Skull is extracranial (outside the brain vault), so pct_intracranial is None.
        if layer_id == "skull" or intracranial_px == 0:
            pct_intracranial = None
        else:
            pct_intracranial = round(float((area_px / intracranial_px) * 100.0), 1)

        # 3. 3D Volume in mL:
        # voxel_count * dz * dy * dx / 1000.0 (where 1000 mm³ = 1 mL)
        if is_3d_valid:
            layer_voxel_count = int(voxel_counts[label_val])
            volume_ml = round(float(layer_voxel_count * dz * dy * dx / 1000.0), 1)
        else:
            volume_ml = None

        # ======================================================================
        # [Checkpoint 2.9] Connected Components (8-connectivity) & Bounding Box
        # ======================================================================
        # Count connected blobs on the 2D slice using standard 8-connectivity
        structure_8 = np.ones((3, 3), dtype=bool)
        labeled_blobs, num_components = scipy.ndimage.label(mask, structure=structure_8)
        n_components = int(num_components)
        if n_components > 10:
            warnings.append("many_components")

        # Bounding box [x0, y0, x1, y1] (inclusive Cartesian pixel coordinates)
        rows, cols = np.where(mask)
        y0, y1 = int(rows.min()), int(rows.max())
        x0, x1 = int(cols.min()), int(cols.max())
        bbox_px = (x0, y0, x1, y1)

        # ======================================================================
        # [Checkpoint 2.10] Centroid Calculation [x, y] in Slice Pixels
        # ======================================================================
        centroid_x = round(float(cols.mean()), 1)
        centroid_y = round(float(rows.mean()), 1)
        centroid_px = (centroid_x, centroid_y)

        # Isolate largest connected component (for anchor pin and lesion diameter)
        blob_sizes = np.bincount(labeled_blobs.ravel())
        blob_sizes[0] = 0  # ignore background
        largest_blob_id = int(np.argmax(blob_sizes))
        largest_component_mask = (labeled_blobs == largest_blob_id)

        # ======================================================================
        # [Checkpoint 2.11] Anchor Pin Coordinate Calculation [x, y] (EDT)
        # [Checkpoint 2.16] Distance Transform with Non-Square Spacing (dy, dx)
        # ======================================================================
        # Places the pin at the deepest interior point farthest from the boundary.
        # Uses sampling=(dy, dx) to accurately scale non-square pixels.
        edt_sampling = (dy, dx) if (dy is not None and dx is not None) else (1.0, 1.0)
        dist_map = scipy.ndimage.distance_transform_edt(largest_component_mask, sampling=edt_sampling)
        # Deterministic tie-breaker: first index in row-major order
        flat_best_idx = int(np.argmax(dist_map))
        best_row, best_col = np.unravel_index(flat_best_idx, dist_map.shape)
        anchor_px = (int(best_col), int(best_row))

        # ======================================================================
        # [Checkpoint 2.12] Lesion Maximum Diameter Calculation (Convex Hull)
        # [Checkpoint 2.17] Pairwise Distances Between Hull Vertices
        # ======================================================================
        max_diameter_mm: Optional[float] = None
        if layer_id == "lesion" and dx is not None and dy is not None:
            comp_rows, comp_cols = np.where(largest_component_mask)
            if len(comp_rows) >= 3:
                # Convert pixel indices to physical millimeter space
                pts_mm = np.column_stack([comp_cols * dx, comp_rows * dy])
                try:
                    hull = scipy.spatial.ConvexHull(pts_mm)
                    hull_pts = pts_mm[hull.vertices]
                    pairwise_dists = scipy.spatial.distance.pdist(hull_pts)
                    max_diameter_mm = round(float(np.max(pairwise_dists)), 1)
                except Exception:
                    # Fallback for degenerate collinear points
                    pairwise_dists = scipy.spatial.distance.pdist(pts_mm)
                    max_diameter_mm = round(float(np.max(pairwise_dists)), 1)
            elif len(comp_rows) == 2:
                p1 = np.array([comp_cols[0] * dx, comp_rows[0] * dy])
                p2 = np.array([comp_cols[1] * dx, comp_rows[1] * dy])
                max_diameter_mm = round(float(np.linalg.norm(p1 - p2)), 1)
            else:
                max_diameter_mm = round(float(max(dx, dy)), 1)

        # ======================================================================
        # [Checkpoint 2.13] Hemisphere Classification
        # [Checkpoint 2.18] Anatomical Patient Orientation Mapping (patient_left)
        # ======================================================================
        # Rule: >= 85% mass on one side -> that side;
        # Else if centroid within 5 mm of midline -> "midline"; else "both".
        hemisphere: Optional[str] = None
        if patient_left is not None:
            total_mask_px = len(cols)
            left_side_px = int(np.count_nonzero(cols < midline_col))
            right_side_px = int(np.count_nonzero(cols > midline_col))

            if (left_side_px / total_mask_px) >= 0.85:
                img_side = "image_left"
            elif (right_side_px / total_mask_px) >= 0.85:
                img_side = "image_right"
            else:
                dist_to_midline_mm = abs(centroid_x - midline_col) * (dx if dx is not None else 1.0)
                if dist_to_midline_mm <= 5.0:
                    img_side = "midline"
                else:
                    img_side = "both"

            # Map image column side to patient anatomical side via patient_left
            if img_side == "midline":
                hemisphere = "midline"
            elif img_side == "both":
                hemisphere = "both"
            elif img_side == "image_left":
                hemisphere = "left" if patient_left == "image_left" else "right"
            elif img_side == "image_right":
                hemisphere = "right" if patient_left == "image_left" else "left"

        # ======================================================================
        # [Checkpoint 2.14] Intensity Profiling & Statistics
        # [Checkpoint 2.19] Ratio to Normal Brain (Gray + White Matter)
        # ======================================================================
        layer_pixel_values = slice_image[mask]
        mean_val = round(float(np.mean(layer_pixel_values)), 2)
        std_val = round(float(np.std(layer_pixel_values)), 2)

        # Skull is bone/extracranial; ratio_to_brain is meaningless and set to None.
        if layer_id == "skull" or brain_mean_intensity is None or brain_mean_intensity == 0:
            ratio_to_brain = None
        else:
            ratio_to_brain = round(float(mean_val / brain_mean_intensity), 2)

        intensity_report = IntensityMetrics(
            mean=mean_val,
            std=std_val,
            ratio_to_brain=ratio_to_brain,
        )

        # ======================================================================
        # [Checkpoint 2.15] Warning Flags & Confidence Adjustments
        # [Checkpoint 2.20] Small Region Confidence Downgrade
        # ======================================================================
        confidence = base_confidence
        if area_px < 50:
            confidence = "low"
            warnings.append("small_region")

        # ======================================================================
        # [Checkpoint 2.21] Assemble LayerMetrics Object
        # ======================================================================
        layer_reports[layer_id] = LayerMetrics(
            layer_id=layer_id,
            present=True,
            confidence=confidence,
            warnings=warnings,
            area_px=area_px,
            area_mm2=area_mm2,
            volume_ml=volume_ml,
            pct_intracranial=pct_intracranial,
            n_components=n_components,
            centroid_px=centroid_px,
            bbox_px=bbox_px,
            anchor_px=anchor_px,
            max_diameter_mm=max_diameter_mm,
            hemisphere=hemisphere,
            intensity=intensity_report,
        )

    # ==========================================================================
    # [Checkpoint 2.22] Return Validated MetricsReport (Contract C4)
    # ==========================================================================
    return MetricsReport(
        version=1,
        slice_index=slice_index,
        n_slices=n_slices,
        spacing_mm=(dz, dy, dx) if spacing is not None else None,
        intracranial_area_mm2=intracranial_area_mm2,
        layers=layer_reports,
    )