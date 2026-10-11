"""CortexView API (contract C3).

    POST /api/analyze          files -> segmentation + metrics (+ flat layers), render starts in the background
    GET  /api/render/{job_id}  poll for the illustration (pending | done | failed)
    GET  /api/health

Run from the repo root:
    backend/.venv/Scripts/python.exe -m uvicorn api.app:app --app-dir backend --port 8000

Uploads stay in memory (never spooled to disk), nothing about the patient is logged, and render jobs live
in memory only (api/jobs.py). Config comes from environment variables or backend/.env (see .env.example).
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException, MultiPartParser

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE reader for backend/.env; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(BACKEND / ".env")

import renderer  # noqa: E402
from cortexview import pipeline  # noqa: E402
from cortexview.export import guide_rgb, label_map, window_8bit  # noqa: E402
from cortexview.layers import load_layers  # noqa: E402
from metrics_engine import report  # noqa: E402
from shared.contracts import (  # noqa: E402
    AnalyzeResponse, ApiWarning, ErrorBody, ErrorDetail, Fidelity, IllustratedLayer, LayerResult,
    RenderDone, RenderFailed, RenderPending, RenderRef, SliceInfo,
)

from .jobs import JobStore  # noqa: E402
from .messages import warning_message  # noqa: E402

# Upload size cap in bytes; 0 = no cap (the demo default). Set MAX_UPLOAD_MB to bring one back.
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "0")) * 1024 * 1024)
MAX_FILES = 5000                    # a whole ReMIND Preop folder is a few hundred files
DISCLAIMER = "Educational visualization. Not a diagnostic device."
POLL_INTERVAL_MS = 1000

# Keep uploaded files in memory: Starlette would otherwise spool anything over 1 MB to a temp file.
MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES or 1 << 62

log = logging.getLogger("cortexview")
jobs = JobStore(workers=int(os.environ.get("RENDER_WORKERS", "2")))

app = FastAPI(title="CortexView API", version="1", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------------------ errors

class ApiError(Exception):
    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code, self.status, self.message = code, status, message


def _error(code: str, status: int, message: str) -> JSONResponse:
    return JSONResponse(ErrorBody(error=ErrorDetail(code=code, message=message)).model_dump(), status_code=status)


@app.exception_handler(ApiError)
async def _api_error(_: Request, e: ApiError):
    return _error(e.code, e.status, e.message)


@app.exception_handler(pipeline.AnalyzeError)
async def _analyze_error(_: Request, e: pipeline.AnalyzeError):
    return _error(e.code, e.status, e.message)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, e: RequestValidationError):
    return _error("bad_request", 422, "The request is missing a field or has one in the wrong format.")


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, e: StarletteHTTPException):
    if e.status_code == 404:
        return _error("not_found", 404, "Not found.")
    return _error("bad_request" if e.status_code < 500 else "internal", e.status_code, "The request couldn't be handled.")


@app.exception_handler(Exception)
async def _internal_error(_: Request, e: Exception):
    log.error("internal error: %s", type(e).__name__)      # the type only: messages may contain header values
    return _error("internal", 500, "Something went wrong on the server.")


# ------------------------------------------------------------------------------------------ routes

@app.get("/api/health")
def health():
    return {"status": "ok", "render_backends": renderer.Config.from_env().available()}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request):
    length = request.headers.get("content-length")
    if MAX_UPLOAD_BYTES and length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 1024 * 1024:
        raise ApiError("too_large", 413, f"The upload is over the {MAX_UPLOAD_BYTES // 2**20} MB limit.")
    try:
        form = await request.form(max_files=MAX_FILES, max_fields=20)
    except MultiPartException:
        raise ApiError("not_dicom", 422, "The upload couldn't be read. Send .dcm files or a .zip as `files`.")
    try:
        uploads = [f for f in form.getlist("files") if hasattr(f, "read")]
        if not uploads:
            raise ApiError("not_dicom", 422, "No files were uploaded.")
        if MAX_UPLOAD_BYTES and sum(f.size or 0 for f in uploads) > MAX_UPLOAD_BYTES:
            raise ApiError("too_large", 413, f"The upload is over the {MAX_UPLOAD_BYTES // 2**20} MB limit.")
        sources = [io.BytesIO(await f.read()) for f in uploads]
        slice_index = _int_field(form.get("slice_index"))
        series_uid = form.get("series_uid") or None
    finally:
        await form.close()
    t0 = time.monotonic()
    n_files = len(sources)
    out = await run_in_threadpool(_analyze, sources, slice_index, series_uid)
    # counts and timing only: never file names, header values or anything about the patient
    log.warning("analyze: %d files, %d slices, %.1f s", n_files, out.slice.n_slices, time.monotonic() - t0)
    return out


@app.get("/api/render/{job_id}", response_model=RenderPending | RenderDone | RenderFailed)
def render_status(job_id: str):
    out = jobs.poll(job_id)
    if out is None:
        raise ApiError("not_found", 404, "This result has expired. Upload the scan again.")
    if out == "pending":
        return RenderPending()
    if isinstance(out, TimeoutError):
        return RenderFailed(reason="timeout")
    if isinstance(out, BaseException):
        log.error("render error: %s", type(out).__name__)
        return RenderFailed(reason="internal")
    return out


def _int_field(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ApiError("bad_slice_index", 422, "slice_index must be a whole number.")


# ------------------------------------------------------------------------------------------ work

def _analyze(sources, slice_index, series_uid) -> AnalyzeResponse:
    result = pipeline.run(sources, slice_index=slice_index, series_uid=series_uid)
    del sources
    metrics = report(result)
    vol, sl = result.volume, result.slice
    layers = load_layers()
    masks = {l.id: np.asarray(sl.masks[l.id], bool) for l in layers}
    mri8 = window_8bit(np.asarray(result.image, np.float64), sl.head)
    guide = guide_rgb(label_map(masks))
    intracranial = None if sl.intracranial is None else np.asarray(sl.intracranial, bool)

    job_id = jobs.submit(lambda: _render_job(guide, masks, mri8, intracranial))
    h, w = mri8.shape
    spacing = vol.spacing
    return AnalyzeResponse(
        job_id=job_id,
        disclaimer=DISCLAIMER,
        slice=SliceInfo(
            width=w, height=h, mri_png=_data_url(mri8),
            slice_index=result.slice_index, n_slices=int(vol.shape[0]), plane=vol.plane,
            pixel_spacing_mm=(round(float(spacing[1]), 4), round(float(spacing[2]), 4)),
            sequence=vol.sequence if vol.sequence in ("T1_pre", "T1_post", "T2", "FLAIR") else "unknown",
            sequence_source=vol.sequence_source, patient_left=vol.patient_left,
        ),
        layers=[LayerResult(id=l.id, source=sl.source.get(l.id, "auto"), confidence=sl.confidence[l.id],
                            flat_png=_data_url(_flat_rgba(masks[l.id], l.rgb))) for l in layers],
        metrics=metrics,
        render=RenderRef(status="pending", poll_url=f"/api/render/{job_id}", poll_interval_ms=POLL_INTERVAL_MS),
        warnings=[ApiWarning(code=c, message=warning_message(c)) for c in result.warnings],
    )


def _render_job(guide, masks, mri8, intracranial) -> RenderDone:
    r = renderer.render(guide, masks, mri8, intracranial)
    cut = renderer.cut_layers(r.image_rgb, masks)
    return RenderDone(
        backend=r.backend,
        fidelity=Fidelity(mean_iou=r.mean_iou, phantom_lesion_pct=r.phantom_lesion_pct, passed=r.passed),
        layers=[IllustratedLayer(id=lid, illustrated_png=_data_url(rgba)) for lid, rgba in cut.items()],
    )


def _flat_rgba(mask: np.ndarray, rgb) -> np.ndarray:
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask] = (*rgb, 255)
    return out


def _data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
