"""Primary AI segmenter for CortexView.

This is a 3D nnU-Net trained for T1/T1-post volumes and returns the same
``Segmentation`` contract as the classical implementation. Unsupported scans
and unavailable inference runtimes deliberately fall back to the classical
segmenter, with ``segmenter_fallback`` in the warnings.

The default weights live at ``ml/weights/cortexview_nnunet`` when that copy is
available. Set ``CORTEXVIEW_AI_WEIGHTS`` to replace it without editing
application code.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from ..layers import layer_ids
from ..segment import Params, Segmentation, _auto_slice, _exclusive, analyze as classical


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "ml" / "weights" / "cortexview_nnunet"
CHECKPOINT = Path("fold_all") / "checkpoint_final.pth"
log = logging.getLogger("cortexview")


def _weights() -> Path:
    """The override if it holds a checkpoint, else the repo copy (a stale path from another machine is ignored)."""
    override = os.environ.get("CORTEXVIEW_AI_WEIGHTS", "").strip()
    if override and (Path(override) / CHECKPOINT).is_file():
        return Path(override)
    return DEFAULT_WEIGHTS


WEIGHTS = _weights()
T1 = {"T1_post", "T1_pre"}
# nnU-Net class IDs, defined by the supplied model's dataset.json.
CLASSES = {1: "skull", 2: "gray_matter", 3: "white_matter", 4: "csf", 5: "lesion"}
CONF_CUTS = (0.75, 0.9)

_predictor = None


def available() -> tuple[bool, str]:
    """Return whether this process can execute the bundled AI model."""
    try:
        import torch  # noqa: F401
        import nnunetv2  # noqa: F401
    except ImportError:
        return False, "PyTorch/nnU-Net not installed"
    import torch
    if not torch.cuda.is_available():
        return False, "no NVIDIA GPU"
    checkpoint = WEIGHTS / CHECKPOINT
    if not checkpoint.is_file():
        return False, f"no weights in {WEIGHTS}"
    if checkpoint.stat().st_size < 1_000_000:
        return False, f"{checkpoint} is a Git LFS pointer; run `git lfs install` then `git lfs pull`"
    return True, ""


def _load():
    """Lazily load one deterministic, full-FP32 predictor per process."""
    global _predictor
    if _predictor is None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        class FP32Predictor(nnUNetPredictor):
            def _internal_maybe_mirror_and_predict(self, x):
                with torch.autocast("cuda", enabled=False):
                    return super()._internal_maybe_mirror_and_predict(x.float())

        predictor = FP32Predictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=True, device=torch.device("cuda", 0),
            verbose=False, verbose_preprocessing=False, allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(WEIGHTS), use_folds=("all",), checkpoint_name="checkpoint_final.pth"
        )
        _predictor = predictor
    return _predictor


def predict(img: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Return nnU-Net classes and selected-class probabilities for ``[k, j, i]``."""
    segmentation, probabilities = _load().predict_single_npy_array(
        np.asarray(img, np.float32)[None], {"spacing": [float(s) for s in spacing]},
        None, None, True,
    )
    return segmentation.astype(np.uint8), probabilities.max(0).astype(np.float32)


def _grade(probability: float) -> str:
    return "low" if probability < CONF_CUTS[0] else ("medium" if probability < CONF_CUTS[1] else "high")


def analyze(img, spacing, sequence: str = "unknown", doctor: dict | None = None,
            p: Params = Params(), tissue_slices=None, lesion_hint=None) -> Segmentation:
    """Segment a supported 3D T1 volume, otherwise use the classical fallback."""
    img = np.asarray(img, np.float32)
    usable, reason = available()
    if not usable:
        log.warning("AI segmenter unavailable (%s); using classical", reason)
    elif img.ndim != 3 or sequence not in T1:
        log.warning("AI segmenter needs a 3D T1 volume (got %dD %s); using classical", img.ndim, sequence)
    else:
        try:
            return _analyze(img, spacing, doctor)
        except Exception as e:
            # An upload must remain usable if a CUDA or model error occurs.
            log.warning("AI segmenter failed (%s: %s); using classical", type(e).__name__, e)
    result = classical(img, spacing, sequence, doctor, p, tissue_slices, lesion_hint)
    result.warnings.append("segmenter_fallback")
    return result


def _analyze(img: np.ndarray, spacing, doctor: dict | None) -> Segmentation:
    doctor = {name: np.asarray(mask, bool) for name, mask in (doctor or {}).items() if mask is not None}
    classes, probabilities = predict(img, spacing)
    raw = {layer: np.zeros(img.shape, bool) for layer in layer_ids()}
    for class_id, layer in CLASSES.items():
        raw[layer] = classes == class_id
    head = classes > 0
    source = {layer: "auto" for layer in layer_ids()}
    warnings = []
    if "lesion" in doctor:
        raw["lesion"] = doctor["lesion"] & head
        source["lesion"] = "doctor"
    if "ventricles" in doctor:
        raw["csf"] |= doctor["ventricles"] & head & ~raw["lesion"]
        warnings.append("csf_includes_doctor_ventricles")
    masks = _exclusive(raw, img.shape)
    intracranial = head & ~masks["skull"]
    confidence = {
        layer: ("empty" if not mask.any() else "high" if source[layer] == "doctor"
                else _grade(float(probabilities[mask].mean())))
        for layer, mask in masks.items()
    }
    stats = {"lesion_ml": float(masks["lesion"].sum() * np.prod(spacing) / 1000)}
    return Segmentation(masks, confidence, source, warnings, head, intracranial, stats,
                        _auto_slice(masks["lesion"], intracranial), None)


def segment_volume(volume, meta: dict | None = None) -> dict[str, np.ndarray]:
    return analyze_volume(volume, meta).masks


def analyze_volume(volume, meta: dict | None = None) -> Segmentation:
    meta = meta or {}
    if hasattr(volume, "data"):
        img, spacing, sequence = volume.data, volume.spacing, volume.sequence
    else:
        img, spacing, sequence = volume, meta.get("spacing_mm"), meta.get("sequence", "unknown")
    return analyze(img, spacing, meta.get("sequence", sequence), meta.get("doctor"),
                   meta.get("params", Params()), meta.get("tissue_slices"))


def segment(slice_img, meta: dict | None = None) -> dict[str, np.ndarray]:
    meta = meta or {}
    if meta.get("volume") is not None:
        index = int(meta["slice_index"])
        return analyze_volume(meta["volume"], dict(meta, tissue_slices=index)).cut(index).masks
    return analyze(slice_img, meta.get("pixel_spacing_mm") or [1.0, 1.0],
                   meta.get("sequence", "unknown"), meta.get("doctor"),
                   meta.get("params", Params())).masks
