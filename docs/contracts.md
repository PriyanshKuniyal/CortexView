# CortexView: Shared Contracts (DRAFT v0)

These are the agreements between the pieces of CortexView: backend, segmentation, metrics, rendering and frontend.
Once frozen, changing one needs everyone's OK (TASKS §2).

- **Source of truth in code:** Pydantic models in `shared/contracts.py`, mirrored by hand in `shared/contracts.ts`.
  These are written after this draft is approved.
- **Layers and colors:** `shared/layers.json`, read by both Python and TS.
- Items marked **[DECIDE]** are still open.

---

## C1. Layer registry (`shared/layers.json`)

Colors and label numbers come from `color_scheme.png`. They replace the provisional palette in finetune-brief §5.2, so **FT must update their palette**.

| label | id | Name | RGB | Peel order |
|---|---|---|---|---|
| 0 | (background) | | (0, 0, 0) | |
| 1 | `skull` | Skull / scalp | (222, 184, 135) tan | 1 (removed first) |
| 2 | `gray_matter` | Gray matter | (196, 120, 170) mauve | 2 |
| 3 | `white_matter` | White matter | (245, 240, 225) off-white | 3 |
| 4 | `csf` | CSF / ventricles | (70, 140, 230) blue | 4 |
| 10 | `lesion` | Suspected lesion | (230, 40, 40) red | 5 (removed last) |

Rules:
- Every pixel belongs to exactly one label, so layers never overlap.
- `id` is the key used everywhere: mask filenames, JSON fields, TS types.
- `label` is the pixel value in a label map (C2), and is also the color the guide image uses for that layer.
- Adding a layer means adding one entry here. Labels 5–9 are free for future layers.
- Color names in prompts must match: tan = skull, mauve = gray matter, off-white = white matter, blue = fluid, red = lesion.

---

## C2. Slice folder format (offline: training, evaluation, demo cache)

This replaces finetune-brief §5.1. The live app doesn't write folders; this format is for data passed between team members.

```
slices/<slice_id>/
  meta.json           # see below
  mri.png             # 8-bit grayscale, windowed, padded to square (never resampled)
  label.png           # 8-bit, pixel value = layer label (0,1,2,3,4,10)
  guide.png           # RGB, label.png painted with layer colors
  masks/<id>.png      # 0/255, one per layer in layers.json, mutually exclusive; may be empty
  reference/<id>.png  # OPTIONAL: doctor/dataset masks (e.g. reference/lesion.png), never mixed with masks/
```

- `<slice_id>` is anonymous, e.g. `s0042`. It never contains a patient ID or dataset ID.
  A private mapping file outside the repo links the two.
- **Padding instead of resizing:** padding keeps one pixel equal to the same number of mm, so metrics stay correct.
  Only FT resizes, with nearest-neighbor, for the model.

`meta.json`:
```json
{
  "slice_id": "s0042",
  "source": "remind",                 // "remind" | "kaggle_lgg" | "phantom" | "upload"
  "split": "train",                   // "train" | "eval" | "demo"
  "sequence": "T1_post",              // T1_pre | T1_post | T2 | FLAIR | unknown
  "sequence_source": "header",        // "header" | "tr_te_ti" | "default" | "dataset"
  "plane": "axial",
  "slice_index": 96, "n_slices": 192,
  "pixel_spacing_mm": [0.5, 0.5],     // [row, col]; null if unknown (Kaggle)
  "slice_thickness_mm": 1.0,          // null if unknown
  "pad": [0, 0, 32, 32],              // top, bottom, left, right pixels added to make it square
  "patient_left": "image_right",      // which image side is the patient's left; null if unknown
  "reference_layers": ["lesion"]      // which reference/ masks exist
}
```

---

## C3. API (`backend` ↔ `frontend`)

### Flow: two calls, with polling
```
frontend                                   backend
   │ POST /api/analyze (files)                │  load → pick series/slice → segment → metrics   (< 1 s target)
   │ ◄──────── 200 AnalyzeResponse ───────────│  starts the render in the background
   │                                          │
   │ GET /api/render/{job_id}   every 1 s     │
   │ ◄──────── {status:"pending"} ────────────│
   │ ...                                      │
   │ ◄──────── {status:"done", layers…} ──────│  (~10 s)
```
- The frontend shows flat-color layers from the first response straight away, then swaps in the illustration.
- The frontend stops polling after **60 s** and keeps the flat layers.
- **Server state:** a job's render result is kept in memory only, deleted after it's fetched or after **5 min**, and never written to disk.

### `POST /api/analyze`
`multipart/form-data`:

| Field | Required | Meaning |
|---|---|---|
| `files` | yes, 1..n | `.dcm` files and/or one `.zip`. May mix MR images and doctor SEG files. Total cap: **[DECIDE] 300 MB**. |
| `slice_index` | no | Which slice to show; the default is auto (below). |
| `series_uid` | no | Force a particular MR series; the default is auto. |

**Doctor labels (DICOM-SEG).**
- Any uploaded file with Modality = SEG is read as a doctor label.
- Each segment is matched to a layer by its name or code:
  | Segment | How it's used |
  |---|---|
  | Tumor | `lesion` |
  | Ventricles | part of `csf` |
  | Cerebrum / brain | the brain boundary, so skull = head minus brain |
- An unrecognised segment is ignored, with a warning.
- When a doctor label is used, that layer's `source` is `"doctor"` instead of `"auto"`, and the UI says so.

**Auto series choice:**
1. the series a doctor tumor label references, if it's axial
2. otherwise axial T1-post
3. then axial FLAIR
4. then axial T2
5. then any axial MR

**Auto slice choice:** the slice with the largest lesion area if there is a lesion, otherwise the middle slice.

**Doctor label drawn on a different scan.** Example: the tumor drawn on the sagittal T2.
- If both scans share a `FrameOfReferenceUID` (the same scanner coordinate system), the label is mapped onto the displayed slice using DICOM patient coordinates (nearest-neighbor), with warning `seg_resampled`.
- Otherwise it isn't used, with warning `seg_frame_mismatch`.

**200 → `AnalyzeResponse`**
```json
{
  "job_id": "b1f…",                              // random, not derived from the data
  "disclaimer": "Educational visualization. Not a diagnostic device.",
  "slice": {
    "width": 512, "height": 512,
    "mri_png": "data:image/png;base64,…",
    "slice_index": 96, "n_slices": 192,
    "plane": "axial",
    "pixel_spacing_mm": [0.5, 0.5],
    "sequence": "T1_post", "sequence_source": "header",
    "patient_left": "image_right"
  },
  "layers": [ LayerResult, … ],                  // one per layers.json entry, in peel order
  "metrics": MetricsReport,                      // C4
  "render": { "status": "pending", "poll_url": "/api/render/b1f…", "poll_interval_ms": 1000 },
  "warnings": [ { "code": "seg_resampled", "message": "…" } ]
}
```

`LayerResult`:
```json
{
  "id": "lesion",
  "source": "doctor",                            // "auto" | "doctor"
  "confidence": "high",                          // "high" | "medium" | "low" | "empty"
  "flat_png": "data:image/png;base64,…",         // RGBA, layer color where mask=1, transparent elsewhere
  "illustrated_png": null                        // filled in by GET /api/render
}
```

### `GET /api/render/{job_id}`

| Status | Body |
|---|---|
| `200` | `{ "status": "pending" }` |
| `200` | `{ "status": "done", "backend": "workers-klein-4b", "fidelity": {"mean_iou": 0.86, "phantom_lesion_pct": 0.0, "passed": true}, "layers": [ {"id": "skull", "illustrated_png": "data:…"}, … ] }` |
| `200` | `{ "status": "failed", "reason": "timeout" }`. The frontend keeps the flat layers. |
| `404` | Unknown or expired job. |

`backend` is one of `local-lora-v1` | `workers-klein-4b` | `workers-klein-9b` | `flat-fallback` | `cache`.

### Errors (any 4xx/5xx)
```json
{ "error": { "code": "not_dicom", "message": "…human-readable…" } }
```

| Code | HTTP | Meaning |
|---|---|---|
| `not_dicom` | 422 | A file isn't DICOM or zip |
| `not_mr` | 422 | No MR image series found |
| `no_axial_series` | 422 | Nothing we can show as an axial slice |
| `too_large` | 413 | Upload is over the size cap |
| `bad_slice_index` | 422 | `slice_index` is out of range |
| `internal` | 500 | Something else went wrong |

Messages and logs never contain patient info, filenames or header values.

---

## C4. Metrics (`MetricsReport`, produced by MP)

All numbers are computed from the masks and pixel spacing. **No AI.** The same input always gives the same output.

```json
{
  "intracranial_area_mm2": 14230.5,             // brain + CSF + lesion; the denominator for %
  "layers": {
    "lesion": {
      "present": true,
      "area_px": 1529,
      "area_mm2": 382.3,                         // null if spacing unknown
      "pct_intracranial": 2.7,
      "n_components": 1,                         // separate blobs
      "centroid_px": [301.2, 188.7],             // [x, y] in slice pixels
      "bbox_px": [270, 160, 335, 220],           // [x0, y0, x1, y1]
      "max_diameter_mm": 31.4,                   // lesion only; null for other layers
      "volume_ml": 12.8,                         // only if the whole 3D mask is known (e.g. doctor SEG); else null
      "intensity": { "mean": 0.71, "std": 0.08, "ratio_to_brain": 1.48 },  // on the 0–1 windowed image
      "hemisphere": "left",                      // "left" | "right" | "both" | "midline" | null
      "anchor_px": [298, 190],                   // where the pin goes: the point deepest inside the mask
      "confidence": "high",
      "warnings": []
    }
    // … same shape for skull, gray_matter, white_matter, csf
  },
  "text": [
    { "layer": "lesion", "template_id": "lesion_area", "text": "The highlighted area covers about 3.8 cm², 2.7% of the space inside the skull." }
  ]
}
```

- MRI brightness has no physical unit, so intensity values are only relative within one scan.
- `hemisphere` comes from DICOM orientation (`patient_left`). It's null if unknown, and there are no lobe names.
- `text` only fills in fixed templates (list in a later section, P1). There is no generated prose.

---

## C5. Render (`render_slice()`, owned by FT)

The signature is unchanged from finetune-brief §10. The changes:
- The guide uses the C1 colors.
- The `masks` dict keys are the C1 `id`s.
- There's a new backend name, `cache`.
- The backend calls it in the background after `/api/analyze` returns.
- Then the backend cuts the illustration into per-layer RGBA PNGs using the masks, so AI output never shows outside a mask.

---

## Open decisions
1. **[DECIDE]** Upload size cap. In ReMIND-001 an MR series is 22–39 MB. A whole Preop folder is 26–305 MB across patients 001–028.
2. **[DECIDE]** Can the user change the slice after analysis?
   - (a) Re-POST everything with `slice_index`: simple and stateless, but re-uploads the files.
   - (b) The backend keeps the volume for N minutes: fast, but holds patient pixels in memory.
3. **[DECIDE]** Is there a doctor view vs. patient view toggle? It changes which `text` templates are used.
4. Does the chat UI send any free text with the upload? This draft assumes no: files only.
