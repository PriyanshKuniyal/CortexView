# CortexView: Shared Contracts (DRAFT v0)

These are the agreements between the pieces of CortexView: backend, segmentation, metrics, rendering and frontend.
Once frozen, changing one needs everyone's OK (TASKS §2).

- **Source of truth in code:** Pydantic models in `shared/contracts.py`, mirrored by hand in `frontend/src/cortexview/api/types.ts`.
  So far only C4 is in `contracts.py`; the rest are written after this draft is approved.
- **Layers and colors:** `shared/layers.json`, read by both Python and TS.
- Items marked **[DECIDE]** are still open.

<!-- NOTE FOR TEAMMATES AND AI AGENTS: a second segmentation backend (AI, 3D) is being built on branch
     `segmentation-AI-version`. It plugs in behind the same interface as the classical one. Read C6 before
     changing segment(), the layer list, C3 timing or C4. SynthSeg GM/WM/CSF training labels arrived 2026-10-11 (data/synthseg). -->
> **⚠ Two segmentation backends.** The classical one (`backend/cortexview/segment.py`) and an AI one (branch `segmentation-AI-version`, see **C6**) must stay interchangeable via one setting.
> If you change `segment()`, the layer list (C1), the timing in C3, or C4, check C6 too.

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
   │ POST /api/analyze (files)                │  load → pick series/slice → segment → metrics   (< 1 s target; AI segmenter ~10–25 s, see C6)
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
(The masks themselves may come from the AI segmenter, C6. The metrics are still pure math on whatever masks they get.)

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
      "volume_ml": 12.8,                         // whole scan; only if the whole 3D mask is known, else null (see below)
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
  They're measured on the slice windowed to 0–1 inside the head, the same window as `mri.png`.
  `intensity` is null for an empty layer, and `ratio_to_brain` (vs. gray + white matter) is null for skull.
- `pct_intracranial` is null for skull, which is outside the intracranial space.
- `volume_ml`: head, brain boundary and lesion are always segmented in 3D, so skull and lesion get a volume from any 3D series.
  CSF, gray and white matter only get one when every slice was tissue-classified. The API's fast path splits them on the shown slice only, so there they're null.
  A 2D upload (fewer than 20 slices) gives null for every layer.
- `hemisphere` comes from DICOM orientation (`patient_left`). It's null if unknown, and there are no lobe names.
  - The midline is fitted on the slice by left/right mirror symmetry, the same way as the segmenter's lesion midline.
  - If 85% or more of a layer is on one side, it's that side; otherwise it's `midline` if the centroid is within 5 mm of the midline, else `both`.
- `anchor_px` and `max_diameter_mm` use the layer's largest blob.
- `text` only fills in the fixed templates in `shared/text_templates.json`. There is no generated prose.
  `layer_volume` is emitted once for each layer that has a `volume_ml`.
- Per-layer `warnings`:

  | Code | Meaning |
  |---|---|
  | `small_region` | Under 50 px on this slice. An automatic layer drops to `low` confidence; a doctor label keeps its confidence. |
  | `many_components` | More than 10 separate lesion blobs on this slice. |
  | `midline_estimated` | The mirror-symmetry fit wasn't trusted, so the middle of the brain's width was used for `hemisphere`. |

- Code: `backend/metrics_engine/`. `metrics_engine.report(result)` takes a `pipeline.Result`.

---

## C5. Render (`render_slice()`, owned by FT)

The signature is unchanged from finetune-brief §10. The changes:
- The guide uses the C1 colors.
- The `masks` dict keys are the C1 `id`s.
- There's a new backend name, `cache`.
- The backend calls it in the background after `/api/analyze` returns.
- Then the backend cuts the illustration into per-layer RGBA PNGs using the masks, so AI output never shows outside a mask.

---

## C6. Segmentation backends: classical vs AI (owned by PK; AI work on branch `segmentation-AI-version`)

<!-- NOTE FOR TEAMMATES AND AI AGENTS: this section is the agreement between the classical segmenter
     (backend/cortexview/segment.py) and the AI segmenter (backend/cortexview/segment_ai/, training in ml/).
     Don't edit one segmenter's code from the other's work. Ask PK before changing anything here. -->

**Setting:** `CORTEXVIEW_SEGMENTER = classical | ai` (default `classical`).
If `ai` fails, or there's no NVIDIA GPU, the backend falls back to `classical` with warning `segmenter_fallback`.

**Interface: decided, volume in, label volume out (same as classical).**
DICOM stores one slice per `.dcm` file. Before `segment()` is called, the shared loader turns the pile of files into one 3D array:
```
upload: e.g. 192 .dcm files (one per slice)
  └─ dicom_io.load_series(...)   read with pydicom → sort by ImagePositionPatient → np.stack
                                 → apply slope/intercept → spacing + orientation from headers
                                 → reorient to axial (permute/flip only, never resampled)
volume = Volume(data [k, j, i], affine, sequence, ...)       e.g. (192, 256, 216) for a T1-post
segment(volume, meta) → label volume, same shape, values = C1 labels (0, 1, 2, 3, 4, 10)
```
- **Both segmenters use the same loader** (`backend/cortexview/dicom_io.py` + `volume.py`, owned by the classical side).
  The AI branch merges `main` and uses the loader unchanged.
- **The AI model reads T1 only** (`T1_post`, `T1_pre`): that's what its training scans are. Other sequences, single
  slices (< 20) and machines without a usable GPU or weights go to the classical segmenter with `segmenter_fallback`.
- Per-layer masks, the displayed slice and the C3 PNGs are all cut from the label volume afterwards.
  So the label volume also gives lesion volume in mL (C4) and the auto "largest lesion" slice.
- If the classical `segment()` signature changes, the AI one follows it.

**Where each AI layer's training labels come from:**
| Layer | Training labels | Who draws them | Status |
|---|---|---|---|
| `lesion` | ReMIND tumor SEG; **empty on healthy IXI scans** | **manual, doctors** | available |
| `csf`, ventricles part | ReMIND ventricles SEG where it exists, else SynthSeg ventricles | BrainLab (automatic) / SynthSeg | available |
| `csf`, outer part (sulci + around the brain) | **SynthSeg** label 24 (silver standard) | SynthSeg | available (2026-10-11) |
| `skull` | head − everything SynthSeg labels − tumor | head: classical head mask | available |
| `gray_matter`, `white_matter` | **SynthSeg** (silver standard); brainstem and ventral DC count as white matter | SynthSeg | available (2026-10-11) |

Scans (built by `ml/build_dataset.py`, split in `ml/splits_v2.json`, anonymous IDs only):
- **ReMIND T1-post** with a tumor: SynthSeg from FreeSurfer `mri_synthseg --robust` (`data/synthseg`, per
  `docs/prompts/synthseg-gm-wm.md`). Patients failing SynthSeg's QC (< 0.65) are left out.
- **Healthy IXI T1** (brain-development.org/ixi-dataset, CC BY-SA 3.0), so the model learns that a scan may have no
  tumor. Labelled with DIPY's PyTorch port of SynthSeg (`ml/synthseg_dipy.py`; same weights, non-robust variant).
- No label in a 3 mm ring around the tumor or in the resection cavity +3 mm (nnU-Net ignore label): SynthSeg's labels mean nothing there.
- Wherever ReMIND and SynthSeg both have a label, ReMIND wins (tumor > ventricles > SynthSeg).
- **Never train on Kaggle LGG** (CC BY-NC-SA would carry over to the weights). It may be used for evaluation only.
- Weights stay out of the public repo (`ml/weights/`, gitignored).

**Evaluation.** Both segmenters are scored by `ml/evaluate.py` (`--classical` for the classical one) on the same
held-out scans: 12 ReMIND patients + 33 healthy IXI scans (anonymous IDs only). Patients are never split across train and test.

**Determinism.** Inference runs in PyTorch deterministic mode, in full 32-bit precision, with pinned versions.
On the same machine and versions, the same input gives the same masks (checked by a repeat-run test).
Across different GPUs or drivers, a few edge pixels may differ.

**Timing: decided.** The C3 1 s target is relaxed for AI mode (PK, 2026-10-10).
The AI segmenter works on the whole 3D series on the GPU, which takes roughly 10–25 s.
`POST /api/analyze` may take that long when `CORTEXVIEW_SEGMENTER=ai`.

---

## Open decisions
1. SIZE_CAP = 300 mb
4.  This draft assumes no text files only. this wud be intorduced in next parts
5. ~~AI segmenter timing and how `segment()` gets the volume (C6).~~ Decided 2026-10-10: 1 s relaxed for AI mode; volume in, label volume out (see C6).
