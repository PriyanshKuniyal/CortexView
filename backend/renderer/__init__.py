"""Contract C5 rendering: guide (flat colour map) -> illustration, behind a fidelity gate.

    render(guide_rgb, masks, mri8) -> RenderResult

Backends are tried in order (env RENDER_BACKENDS, default "cache,local,workers,flat"):

    cache    illustrations saved earlier for the same guide (demo insurance; works offline)
    local    FT's LoRA server behind a Cloudflare Tunnel: POST {LOCAL_RENDER_URL}/render
    workers  Cloudflare Workers AI, FLUX.2 klein, prompt + guide as reference image
    flat     the guide itself, lightly shaded by the MRI. Always available, never fails.

A backend that is not configured is skipped. Each AI backend gets up to RENDER_ATTEMPTS seeds; an
image that fails the fidelity gate (fidelity.py) is retried with the next seed. Everything stops at
RENDER_DEADLINE_S, well before the frontend's 60 s polling limit, and falls back to flat.

Only PNG pixels leave the backend: no DICOM, no headers, no file names (C3, finetune ground rules).
Nothing is logged.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import fidelity

ROOT = Path(__file__).resolve().parents[2]

# Colour names must match contract C1. The prompt only ever says which regions exist: never numbers,
# measurements or diagnoses (finetune guide, ground rules).
REGION_WORDS = {"skull": "tan = skull and scalp", "gray_matter": "mauve = gray matter",
                "white_matter": "off-white = white matter", "csf": "blue = fluid spaces", "lesion": "red = lesion"}


def build_prompt(present: list[str]) -> str:
    """Generation prompt for one slice, from the layers present on it (in C1 order)."""
    regions = ", ".join(REGION_WORDS[l] for l in present if l in REGION_WORDS)
    no_lesion = "" if "lesion" in present else "There is no lesion: do not add any red area. "
    return ("Redraw image 0 as a clean medical textbook illustration of an axial brain cross-section. "
            "Keep every region exactly where it is in image 0, with the same outline and the same color "
            f"per region: {regions}. {no_lesion}"
            "Soft shading and subtle tissue texture inside each region. Plain black background. "
            "No text, no labels, no arrows.")
CF_URL = "https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}"
CF_MODELS = {"workers-klein-4b": "@cf/black-forest-labs/flux-2-klein-4b",
             "workers-klein-9b": "@cf/black-forest-labs/flux-2-klein-9b",
             "workers-flux2-dev": "@cf/black-forest-labs/flux-2-dev"}       # FLUX.2 [dev], 32B
CF_STEPPED = {"workers-flux2-dev"}      # klein is distilled (steps fixed at 4); dev takes `steps`, billed per step
CF_REF_MAX = 496            # Workers AI: each reference image must be smaller than 512x512


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Config:
    backends: list[str] = field(default_factory=lambda: ["cache", "local", "workers", "flat"])
    attempts: int = 3
    call_timeout_s: float = 30.0
    deadline_s: float = 45.0
    size: int = 768
    local_url: str = ""
    local_key: str = ""
    cf_account: str = ""
    cf_token: str = ""
    cf_backend: str = "workers-klein-4b"
    steps: int = 20
    cache_dir: Path = ROOT / "data" / "render_cache"
    cache_record: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            backends=[b.strip() for b in _env("RENDER_BACKENDS", "cache,local,workers,flat").split(",") if b.strip()],
            attempts=int(_env("RENDER_ATTEMPTS", "3")),
            call_timeout_s=float(_env("RENDER_TIMEOUT_S", "30")),
            deadline_s=float(_env("RENDER_DEADLINE_S", "45")),
            size=int(_env("RENDER_SIZE", "768")),
            local_url=_env("LOCAL_RENDER_URL").rstrip("/"),
            local_key=_env("LOCAL_RENDER_KEY"),
            cf_account=_env("CF_ACCOUNT_ID"),
            cf_token=_env("CF_API_TOKEN"),
            cf_backend=_env("CF_MODEL", "workers-klein-4b"),
            steps=int(_env("RENDER_STEPS", "20")),
            cache_dir=Path(_env("RENDER_CACHE_DIR") or ROOT / "data" / "render_cache"),
            cache_record=_env("RENDER_CACHE_RECORD", "0") == "1",
        )

    def available(self) -> list[str]:
        ok = {"cache": self.cache_dir.is_dir() or self.cache_record,
              "local": bool(self.local_url),
              "workers": bool(self.cf_account and self.cf_token),
              "flat": True}
        return [b for b in self.backends if ok.get(b)]


@dataclass
class RenderResult:
    image_rgb: np.ndarray                 # HxWx3 uint8 at the guide's size
    backend: str                          # contract C3 name
    mean_iou: float
    per_layer_iou: dict[str, float]
    phantom_lesion_pct: float
    passed: bool
    attempts: int
    latency_ms: int


class _Skip(Exception):
    """This backend can't produce an image right now; move on to the next one."""


def render(guide_rgb: np.ndarray, masks: dict[str, np.ndarray], mri8: np.ndarray | None = None,
           intracranial: np.ndarray | None = None, cfg: Config | None = None) -> RenderResult:
    cfg = cfg or Config.from_env()
    t0 = time.monotonic()
    h, w = guide_rgb.shape[:2]
    key = hashlib.sha256(np.ascontiguousarray(guide_rgb).tobytes() + f"{h}x{w}".encode()).hexdigest()
    attempts = 0
    prompt = build_prompt([lid for lid, m in masks.items() if np.asarray(m).any()])

    def result(img, backend, s):
        return RenderResult(img, backend, s.mean_iou, s.per_layer_iou, s.phantom_lesion_pct, s.passed,
                            attempts, int((time.monotonic() - t0) * 1000))

    for backend in cfg.available():
        if backend == "flat":
            break
        if backend == "cache":
            img = _cache_get(cfg, key, (w, h))
            if img is not None:
                return result(img, "cache", fidelity.score(img, masks, intracranial))
            continue
        for seed in range(cfg.attempts):
            remaining = cfg.deadline_s - (time.monotonic() - t0)
            if remaining < 5:
                break
            attempts += 1
            try:
                if backend == "local":
                    png, name = _call_local(cfg, guide_rgb, seed, min(cfg.call_timeout_s, remaining)), "local-lora-v1"
                else:
                    png, name = _call_workers(cfg, guide_rgb, prompt, seed, min(cfg.call_timeout_s, remaining)), cfg.cf_backend
                img = _decode_to(png, (w, h))
            except _Skip:
                break                      # backend down or misconfigured: don't burn more seeds on it
            except Exception:
                continue
            s = fidelity.score(img, masks, intracranial)
            if s.passed:
                if cfg.cache_record:
                    _cache_put(cfg, key, img)
                return result(img, name, s)

    img = flat_render(guide_rgb, mri8)
    return result(img, "flat-fallback", fidelity.score(img, masks, intracranial))


def flat_render(guide_rgb: np.ndarray, mri8: np.ndarray | None) -> np.ndarray:
    """The guide with light shading from the MRI (88-100% brightness), so texture shows but every pixel
    keeps its layer's colour family."""
    if mri8 is None:
        return guide_rgb.copy()
    shade = 0.88 + 0.12 * (mri8.astype(np.float64) / 255.0)
    return np.clip(guide_rgb * shade[..., None], 0, 255).round().astype(np.uint8)


def cut_layers(image_rgb: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One RGBA image per layer: the illustration inside the mask, transparent outside. AI output can
    never show outside a mask, and the edges match the segmentation exactly (C5)."""
    out = {}
    for lid, m in masks.items():
        rgba = np.zeros(image_rgb.shape[:2] + (4,), np.uint8)
        m = np.asarray(m, bool)
        rgba[m, :3] = image_rgb[m]
        rgba[m, 3] = 255
        out[lid] = rgba
    return out


# ------------------------------------------------------------------------------------------ backends

def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _resized_guide(guide_rgb: np.ndarray, longest: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Nearest-neighbour resize (never blend layer colours), longest side = `longest`, sides divisible by 16."""
    h, w = guide_rgb.shape[:2]
    s = longest / max(h, w)
    size = (max(16, int(round(w * s / 16)) * 16), max(16, int(round(h * s / 16)) * 16))
    return np.asarray(Image.fromarray(guide_rgb).resize(size, Image.NEAREST)), size


def _decode_to(data: bytes, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img)


def _multipart(fields: dict[str, str | tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        if isinstance(value, tuple):
            fname, data, ctype = value
            head = (f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
                    f"Content-Type: {ctype}\r\n\r\n").encode()
        else:
            head, data = f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(), value.encode()
        parts.append(b"--" + boundary.encode() + b"\r\n" + head + data + b"\r\n")
    body = b"".join(parts) + b"--" + boundary.encode() + b"--\r\n"
    return body, f"multipart/form-data; boundary={boundary}"


def _post(url: str, fields: dict, headers: dict, timeout: float) -> bytes:
    body, ctype = _multipart(fields)
    req = urllib.request.Request(url, data=body, method="POST", headers={**headers, "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (400, 401, 403, 404, 429):
            raise _Skip() from None        # bad request, wrong key/URL or quota used up: another seed won't help
        raise
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), TimeoutError):
            raise _Skip() from None        # unreachable (tunnel down, no network)
        raise


def _call_local(cfg: Config, guide_rgb: np.ndarray, seed: int, timeout: float) -> bytes:
    """FT's LoRA server (finetune guide D4): multipart `guide` PNG + `seed`, returns a PNG."""
    guide, _ = _resized_guide(guide_rgb, cfg.size)
    return _post(cfg.local_url + "/render", {"guide": ("guide.png", _png(guide), "image/png"), "seed": str(seed)},
                 {"X-CortexView-Key": cfg.local_key}, timeout)


def _call_workers(cfg: Config, guide_rgb: np.ndarray, prompt: str, seed: int, timeout: float) -> bytes:
    """Workers AI REST API (multipart). The reference image must be < 512x512."""
    model = CF_MODELS.get(cfg.cf_backend)
    if model is None:
        raise _Skip()
    ref, _ = _resized_guide(guide_rgb, CF_REF_MAX)
    _, (w, h) = _resized_guide(guide_rgb, cfg.size)
    fields = {"prompt": prompt, "width": str(w), "height": str(h), "seed": str(seed)}
    if cfg.cf_backend in CF_STEPPED:
        fields["steps"] = str(cfg.steps)
    fields["input_image_0"] = ("guide.png", _png(ref), "image/png")
    raw = _post(CF_URL.format(acct=cfg.cf_account, model=model), fields,
                {"Authorization": f"Bearer {cfg.cf_token}"}, timeout)
    body = json.loads(raw)
    if not body.get("success", True):
        raise RuntimeError("workers ai error")
    return base64.b64decode((body.get("result") or body)["image"])   # REST wraps the model output in `result`


# ------------------------------------------------------------------------------------------ cache

def _cache_get(cfg: Config, key: str, size: tuple[int, int]) -> np.ndarray | None:
    p = cfg.cache_dir / f"{key}.png"
    if not p.is_file():
        return None
    try:
        return _decode_to(p.read_bytes(), size)
    except Exception:
        return None


def _cache_put(cfg: Config, key: str, img: np.ndarray) -> None:
    try:
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        (cfg.cache_dir / f"{key}.png").write_bytes(_png(img))
    except OSError:
        pass
