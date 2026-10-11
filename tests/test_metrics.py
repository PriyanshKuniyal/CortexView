"""
tests/test_metrics.py
======================
Unit tests for CortexView Deterministic Metrics Engine (Track MP).
Tests exact hand-computed geometric properties and Contract C4 compliance.
"""

from typing import Any, Dict, Tuple 
import numpy as np
from backend.cortexview.metrics import compute_metrics


# ==============================================================================
# [Checkpoint 3.2] Synthetic Head Phantom Fixture Helper
# ==============================================================================
def create_test_volume(z: int = 10, n: int = 128) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Creates a synthetic 3D phantom brain volume with known geometry:
      - Label 1: Skull (outer ring)
      - Label 2: Gray matter (cortex)
      - Label 3: White matter (subcortical)
      - Label 4: CSF / ventricles (central core)
      - Label 10: Lesion (focal blob on one hemisphere)
    """
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n / 2, xx - n / 2)
    sl = np.zeros((n, n), np.uint8)
    sl[r < 55] = 1   # skull
    sl[r < 50] = 2   # gray matter
    sl[r < 40] = 3   # white matter
    sl[r < 10] = 4   # csf
    sl[np.hypot(yy - 64, xx - 85) < 6] = 10  # lesion off-center

    labels = np.repeat(sl[None], z, axis=0)
    image = np.select([labels == k for k in (1, 2, 3, 4, 10)], [0.3, 0.5, 0.7, 0.1, 0.9], 0).astype(np.float32)
    meta = {
        "spacing_mm": (1.0, 0.5, 0.5),
        "patient_left": "image_right",
        "n_slices": z,
    }
    return labels, image, meta


# ==============================================================================
# [Checkpoint 3.3] Test 1: 10x10 Square Verification
# ==============================================================================
def test_square_10x10():
    """
    Verifies area_px == 100, exact bounding box, and centered centroid/anchor.
    """
    z, h, w = 1, 100, 100
    labels = np.zeros((z, h, w), dtype=np.uint8)
    image = np.ones((z, h, w), dtype=np.float32) * 0.5

    # Place a 10x10 lesion at rows [20..29], cols [30..39]
    labels[0, 20:30, 30:40] = 10

    meta = {
        "spacing_mm": (1.0, 1.0, 1.0),
        "patient_left": "image_left",
        "n_slices": 1,
    }
    layer_info = {"lesion": {"confidence": "high"}}

    report = compute_metrics(labels, image, slice_index=0, meta=meta, layer_info=layer_info)
    lesion = report.layers["lesion"]

    assert lesion.present is True
    assert lesion.area_px == 100
    assert lesion.area_mm2 == 100.0
    assert lesion.n_components == 1
    assert lesion.bbox_px == (30, 20, 39, 29)  # [x0, y0, x1, y1]
    assert lesion.centroid_px == (34.5, 24.5)   # [x, y]
    # Deepest point in a 10x10 box sits in the central 2x2 pixels
    assert lesion.anchor_px in [(34, 24), (34, 25), (35, 24), (35, 25)]


# ==============================================================================
# [Checkpoint 3.4] Test 2: Circular Disk Diameter
# ==============================================================================
def test_circular_disk_diameter():
    """
    A disk of radius r=10 px with 1.0 mm spacing has max_diameter ≈ 20.0 mm (+/- 1.0 mm).
    """
    z, n = 1, 100
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - 50, xx - 50)

    labels = np.zeros((z, n, n), dtype=np.uint8)
    labels[0, r <= 10] = 10  # lesion disk
    image = np.ones((z, n, n), dtype=np.float32) * 0.5

    meta = {
        "spacing_mm": (1.0, 1.0, 1.0),
        "patient_left": "image_left",
        "n_slices": 1,
    }
    report = compute_metrics(labels, image, slice_index=0, meta=meta, layer_info={})
    lesion = report.layers["lesion"]

    assert lesion.present is True
    assert lesion.max_diameter_mm is not None
    assert 19.0 <= lesion.max_diameter_mm <= 21.0


# ==============================================================================
# [Checkpoint 3.5] Test 3: Anisotropic Non-Square Pixels (dy=1.0, dx=0.5)
# ==============================================================================
def test_anisotropic_pixels():
    """
    10x10 pixel box with dy=1.0, dx=0.5 mm:
    Physical area = 100 * 1.0 * 0.5 = 50.0 mm².
    Diagonal length = sqrt((9*1.0)^2 + (9*0.5)^2) ≈ 10.1 mm.
    """
    z, h, w = 1, 100, 100
    labels = np.zeros((z, h, w), dtype=np.uint8)
    labels[0, 20:30, 30:40] = 10
    image = np.ones((z, h, w), dtype=np.float32) * 0.5

    meta = {
        "spacing_mm": (1.0, 1.0, 0.5),  # dz=1.0, dy=1.0, dx=0.5
        "patient_left": "image_left",
        "n_slices": 1,
    }
    report = compute_metrics(labels, image, slice_index=0, meta=meta, layer_info={})
    lesion = report.layers["lesion"]

    assert lesion.area_px == 100
    assert lesion.area_mm2 == 50.0
    assert 10.0 <= lesion.max_diameter_mm <= 11.5


# ==============================================================================
# [Checkpoint 3.6] Test 4: Two Blobs & Anchor Selection
# ==============================================================================
def test_two_blobs_components_and_anchor():
    """
    Verifies n_components == 2, and anchor pin is anchored inside the LARGER blob.
    """
    z, h, w = 1, 100, 100
    labels = np.zeros((z, h, w), dtype=np.uint8)
    # Big blob: 10x10 at [20..29, 20..29] (100 px)
    labels[0, 20:30, 20:30] = 10
    # Small blob: 3x3 at [70..72, 70..72] (9 px)
    labels[0, 70:73, 70:73] = 10
    image = np.ones((z, h, w), dtype=np.float32) * 0.5

    meta = {
        "spacing_mm": (1.0, 1.0, 1.0),
        "patient_left": "image_left",
        "n_slices": 1,
    }
    report = compute_metrics(labels, image, slice_index=0, meta=meta, layer_info={})
    lesion = report.layers["lesion"]

    assert lesion.n_components == 2
    # Anchor must be inside the bigger blob [20..29, 20..29]
    ax, ay = lesion.anchor_px
    assert 20 <= ax <= 29
    assert 20 <= ay <= 29


# ==============================================================================
# [Checkpoint 3.7] Test 5: Hemisphere Orientation Toggle
# ==============================================================================
def test_hemisphere_orientation_toggle():
    """
    Lesion on image left correctly flips between 'left' and 'right' when patient_left changes.
    """
    z, h, w = 1, 100, 100
    labels = np.zeros((z, h, w), dtype=np.uint8)
    # Intracranial brain background spanning columns 10 to 90 (midline at 50)
    labels[0, 10:90, 10:90] = 2  # Gray matter
    # Place lesion on image left: columns 15 to 25
    labels[0, 40:60, 15:25] = 10
    image = np.ones((z, h, w), dtype=np.float32) * 0.5

    # 5a: With patient_left == 'image_left' -> anatomical 'left'
    meta_left = {"spacing_mm": (1.0, 1.0, 1.0), "patient_left": "image_left", "n_slices": 1}
    rep_left = compute_metrics(labels, image, slice_index=0, meta=meta_left, layer_info={})
    assert rep_left.layers["lesion"].hemisphere == "left"

    # 5b: With patient_left == 'image_right' -> anatomical 'right'
    meta_right = {"spacing_mm": (1.0, 1.0, 1.0), "patient_left": "image_right", "n_slices": 1}
    rep_right = compute_metrics(labels, image, slice_index=0, meta=meta_right, layer_info={})
    assert rep_right.layers["lesion"].hemisphere == "right"


# ==============================================================================
# [Checkpoint 3.8] Test 6: Missing Metadata & Edge Cases
# ==============================================================================
def test_missing_metadata_and_edge_cases():
    """
    - spacing_mm is None -> area_mm2, volume_ml, max_diameter_mm are None + warnings
    - single slice z=1 -> volume_ml is None
    - empty layer -> present=False, confidence='empty', area_px=0
    """
    z, h, w = 1, 100, 100
    labels = np.zeros((z, h, w), dtype=np.uint8)
    labels[0, 20:30, 20:30] = 10  # Only lesion present, others empty
    image = np.ones((z, h, w), dtype=np.float32) * 0.5

    meta_no_spacing = {
        "spacing_mm": None,
        "patient_left": None,
        "n_slices": 1,
    }
    report = compute_metrics(labels, image, slice_index=0, meta=meta_no_spacing, layer_info={})

    # Lesion with unknown spacing
    lesion = report.layers["lesion"]
    assert lesion.present is True
    assert lesion.area_mm2 is None
    assert lesion.volume_ml is None
    assert lesion.max_diameter_mm is None
    assert "spacing_unknown" in lesion.warnings
    assert "orientation_unknown" in lesion.warnings

    # Gray matter is completely empty
    gm = report.layers["gray_matter"]
    assert gm.present is False
    assert gm.confidence == "empty"
    assert gm.area_px == 0
    assert gm.area_mm2 is None
    assert gm.volume_ml is None


# ==============================================================================
# [Checkpoint 3.9] Test 7: Strict Byte-Level Determinism Check
# ==============================================================================
def test_determinism_byte_identical():
    """
    Running twice with identical inputs produces character-for-character identical JSON.
    Uses create_test_volume helper from Checkpoint 3.2.
    """
    labels, image, meta = create_test_volume(z=10, n=128)
    layer_info = {"lesion": {"confidence": "high"}}

    report1 = compute_metrics(labels, image, slice_index=5, meta=meta, layer_info=layer_info)
    report2 = compute_metrics(labels, image, slice_index=5, meta=meta, layer_info=layer_info)

    # Universal Pydantic JSON serialization (v1 and v2 compatible)
    json1 = report1.model_dump_json() if hasattr(report1, "model_dump_json") else report1.json()
    json2 = report2.model_dump_json() if hasattr(report2, "model_dump_json") else report2.json()

    assert json1 == json2, "Output JSON must be 100% deterministic and byte-identical!"