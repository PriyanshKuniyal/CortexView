"""AI segmenter (contract C6): a 3D nnU-Net trained on ReMIND (T1-post, tumour) and healthy IXI scans.

Same interface as the classical segmenter: `analyze(img, spacing, sequence, doctor, p, tissue_slices,
lesion_hint)` returns a classical `Segmentation`, so the pipeline, export and metrics work unchanged.

Falls back to the classical segmenter, with warning `segmenter_fallback`, when:
  - the series isn't T1 (the model was trained on T1 only), or it's a single slice (the model is 3D),
  - there is no NVIDIA GPU, PyTorch/nnU-Net aren't installed, or the weights are missing,
  - anything fails during inference.

Weights: ml/weights/ (gitignored), or the folder in CORTEXVIEW_AI_WEIGHTS.
Inference is deterministic and in full 32-bit precision (C6).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..segment import Params, Segmentation, _auto_slice, _exclusive, analyze as classical
from ..layers import layer_ids

WEIGHTS = Path(os.environ.get("CORTEXVIEW_AI_WEIGHTS",
                              Path(__file__).resolve().parents[3] / "ml" / "weights" / "cortexview_nnunet"))
T1 = {"T1_post", "T1_pre"}
# nnU-Net class -> layer (see ml/build_dataset.py)
CLASSES = {1: "skull", 2: "gray_matter", 3: "white_matter", 4: "csf", 5: "lesion"}
# confidence from the model's own certainty: mean probability of the chosen class over the layer
CONF_CUTS = (0.75, 0.9)

_predictor = None


def available() -> tuple[bool, str]:
    """(usable, reason if not)."""
    try:
        import torch  # noqa: F401
        import nnunetv2  # noqa: F401
    except ImportError:
        return False, "PyTorch/nnU-Net not installed"
    import torch
    if not torch.cuda.is_available():
        return False, "no NVIDIA GPU"
    if not (WEIGHTS / "fold_all" / "checkpoint_final.pth").exists():
        return False, f"no weights in {WEIGHTS}"
    return True, ""


def _load():
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
            """nnU-Net predicts under float16 autocast on GPU; C6 asks for full 32-bit precision."""

            def _internal_maybe_mirror_and_predict(self, x):
                # innermost autocast wins: the network itself runs in float32
                with torch.autocast("cuda", enabled=False):
                    return super()._internal_maybe_mirror_and_predict(x.float())

        pred = FP32Predictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                             perform_everything_on_device=True, device=torch.device("cuda", 0),
                             verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        pred.initialize_from_trained_model_folder(str(WEIGHTS), use_folds=("all",),
                                                  checkpoint_name="checkpoint_final.pth")
        _predictor = pred
    return _predictor


def predict(img: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """(class map [k, j, i] with nnU-Net classes 0-5, probability of the chosen class [k, j, i])."""
    pred = _load()
    seg, prob = pred.predict_single_npy_array(np.asarray(img, np.float32)[None],
                                              {"spacing": [float(s) for s in spacing]},
                                              None, None, True)
    return seg.astype(np.uint8), prob.max(0).astype(np.float32)


def _grade(p):
    return "low" if p < CONF_CUTS[0] else ("medium" if p < CONF_CUTS[1] else "high")


def analyze(img, spacing, sequence: str = "unknown", doctor: dict | None = None, p: Params = Params(),
            tissue_slices=None, lesion_hint=None) -> Segmentation:
    """AI segmentation of a 3D T1 volume; anything else goes to the classical segmenter."""
    img = np.asarray(img, np.float32)
    ok, why = available()
    if img.ndim != 3 or sequence not in T1:
        ok = False
    if ok:
        try:
            return _analyze(img, spacing, sequence, doctor)
        except Exception:                               # never fail the request: classical still works
            pass
    seg = classical(img, spacing, sequence, doctor, p, tissue_slices, lesion_hint)
    seg.warnings.append("segmenter_fallback")
    return seg


def _analyze(img, spacing, sequence, doctor) -> Segmentation:
    doctor = {k: np.asarray(v, bool) for k, v in (doctor or {}).items() if v is not None}
    cls, prob = predict(img, spacing)
    raw = {lid: np.zeros(img.shape, bool) for lid in layer_ids()}
    for c, lid in CLASSES.items():
        raw[lid] = cls == c
    head = cls > 0
    source = {lid: "auto" for lid in layer_ids()}            # C3: "auto" | "doctor"
    warnings = []
    if "lesion" in doctor:                              # doctor labels win (C6: ReMIND wins on overlap)
        raw["lesion"] = doctor["lesion"] & head
        source["lesion"] = "doctor"
    if "ventricles" in doctor:
        raw["csf"] |= doctor["ventricles"] & head & ~raw["lesion"]
        warnings.append("csf_includes_doctor_ventricles")
    masks = _exclusive(raw, img.shape)
    ic = head & ~masks["skull"]
    conf = {}
    for lid, m in masks.items():
        if not m.any():
            conf[lid] = "empty"
        elif source[lid] == "doctor":
            conf[lid] = "high"
        else:
            conf[lid] = _grade(float(prob[m].mean()))
    stats = {"lesion_ml": float(masks["lesion"].sum() * np.prod(spacing) / 1000)}
    return Segmentation(masks, conf, source, warnings, head, ic, stats, _auto_slice(masks["lesion"], ic), None)


def segment_volume(volume, meta: dict | None = None) -> dict[str, np.ndarray]:
    """C6 interface, as classical segment_volume."""
    return analyze_volume(volume, meta).masks


def analyze_volume(volume, meta: dict | None = None) -> Segmentation:
    meta = meta or {}
    if hasattr(volume, "data"):
        img, spacing, seq = volume.data, volume.spacing, volume.sequence
    else:
        img, spacing, seq = volume, meta.get("spacing_mm"), meta.get("sequence", "unknown")
    return analyze(img, spacing, meta.get("sequence", seq), meta.get("doctor"))
