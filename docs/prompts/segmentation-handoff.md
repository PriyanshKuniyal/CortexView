# Handoff: CortexView segmentation (repo "Plie")

You're continuing an earlier Claude Code session. **I'm starting on the segmentation part.**
Before acting, **verify the current state on disk and in git**, because parts of this may be stale.

## Environment and rules
- **System:** Windows 11. PowerShell and Git Bash are both available. Repo root: `C:\Users\Priyansh Kuniyal\Plie`.
- **Remote:** `origin` = github.com/PriyanshKuniyal/Plie, which is **PUBLIC**. Git push works.
  - The GitHub CLI is at `C:\Program Files\GitHub CLI\gh.exe` but is **not logged in**.
- **Python:** 3.13.14. Use `python -I` to run scripts.
  - **None of numpy, scipy, scikit-image, pydicom, Pillow, OpenCV, SimpleITK, nibabel or highdicom are installed.**
  - Analysis scripts so far are stdlib-only. Segmentation will realistically need **numpy + scipy + scikit-image + pydicom** (+ Pillow for PNG).
  - Highdicom is an option for DICOM-SEG, but pydicom alone can decode it.
  - **Ask me before installing anything.** Propose a venv (e.g. `backend/.venv`) and a `requirements.txt`. I declined an install once before.
- **Rules:**
  - Don't commit, push or install without asking.
  - Read-only analysis of the data is fine.
  - Explain things plainly. I'm not a medical-imaging person.
- **Git state at handoff:**
  - On branch `shared-contracts` (pushed; commit `98ab18b`). The PR isn't opened yet because `gh` isn't authenticated.
  - Uncommitted: my edits to `docs/contracts.md` (open-decisions section).
  - **`docs/finetune-brief.md` shows as deleted in the working tree.** I'm not sure that was intentional, so ask me. It's recoverable with `git restore docs/finetune-brief.md`.
  - Local `main` is 2 commits behind `origin/main`: the teammate's frontend, all under `frontend/`, with no overlap.

## The product (CortexView, hackathon: "generative image AI solving a real-world problem")
- **Flow:**
  1. A brain MRI (DICOM) is uploaded.
  2. **Deterministic classical computer vision (no deep learning in v1)** splits a slice into 5 mutually exclusive layers.
  3. FLUX.2 klein (fine-tuned by teammate FT) turns those masks into a clean illustration.
  4. A slider peels the layers off from the outside in.
- **Data pins** show metrics computed **only from math on the masks, never from AI**.
  - Pitch: *"Generative where it helps, deterministic where it matters."*
  - The AI output is clipped to the masks and scored for fidelity, so it can't invent anatomy.
- **Stack:** FastAPI backend + React/Vite/TS frontend.
  - Stateless, with no patient info in logs.
  - A "not a diagnostic device" disclaimer is always shown.
- **Team:**
  - PK = me, lead. I own segmentation, backend and integration.
  - FT = fine-tuning/rendering.
  - MP = metrics + DICOM loading (brief not written yet).
  - Frontend teammate: pushed a chat-style UI that posts to `/api/chat`. The chat is how users attach files.
- **Read first:**
  - `docs/contracts.md`: the shared contracts, DRAFT v0.
  - `shared/layers.json`
  - `docs/TASKS.md`: §3 is segmentation.
  - `docs/finetune-brief.md`: via `git show HEAD:docs/finetune-brief.md` if it's still deleted.
  - `docs/prompts/synthseg-gm-wm.md`

## Shared contracts that constrain segmentation (from `docs/contracts.md`)

### C1 layers (`shared/layers.json`)
| Label | Layer | RGB | Peel order |
|---|---|---|---|
| 0 | background | (0, 0, 0) | – |
| 1 | `skull` | (222, 184, 135) tan | 1 |
| 2 | `gray_matter` | (196, 120, 170) mauve | 2 |
| 3 | `white_matter` | (245, 240, 225) off-white | 3 |
| 4 | `csf` | (70, 140, 230) blue | 4 |
| 10 | `lesion` | (230, 40, 40) red | 5 |

- Colors come from `color_scheme.png` and replace finetune-brief §5.2's palette. FT must be told.
- Every pixel has exactly one label.

### C2 offline slice folder
```
slices/<anon_id>/
  meta.json
  mri.png             # 8-bit, windowed, padded to square (never resampled)
  label.png           # pixel values = labels
  guide.png           # RGB, the label map painted with layer colors
  masks/<id>.png      # 0/255
  reference/<id>.png  # optional, doctor/dataset ground truth
```
- `meta.json` holds:
  - spacing (or null), thickness
  - sequence + `sequence_source`
  - plane, slice index / count, pad
  - `patient_left` (which image side is the patient's left)
  - `reference_layers`
- IDs are anonymous: never TCGA or ReMIND IDs.

### C3 API
`POST /api/analyze` (multipart `files` = `.dcm` files and/or one zip, 1..n; **size cap 300 MB**; files only, no text for now):
1. Group the files by series.
2. Detect DICOM-SEG files as **doctor labels**:
   | Segment | Used for |
   |---|---|
   | Tumor | `lesion` |
   | Ventricles | part of `csf` |
   | Cerebrum | the brain boundary, so skull = head − brain |
3. Auto-pick the series:
   1. the axial series the tumor SEG references
   2. else axial T1-post
   3. then FLAIR
   4. then T2
   5. then any axial MR
4. Auto-pick the slice: largest lesion area, else the middle slice.
5. If a SEG references a different series, resample it via patient coordinates when the `FrameOfReferenceUID` matches (warning `seg_resampled`). Otherwise drop it (`seg_frame_mismatch`).
6. Return metrics + flat RGBA layers in **under 1 s** with a `job_id`.
7. The frontend polls `GET /api/render/{job_id}` every 1 s for the illustration (~10 s), and gives up after 60 s.

- Each layer has `source: auto | doctor` and `confidence: high | medium | low | empty`.
- Errors: `not_dicom`, `not_mr`, `no_axial_series`, `too_large`, `bad_slice_index`, `internal`.

### C4 metrics (MP's job, but they consume my masks)
- area px / mm², % of intracranial area, component count, centroid, bounding box
- lesion max diameter; volume only when a 3D mask is known
- intensity relative to brain
- hemisphere (no lobes)
- pin anchor at the distance-transform maximum
- confidence and warnings
- template-only text

## Data

### ReMIND (main DICOM dataset; TCIA, CC BY 4.0, Brigham & Women's)
- **Citation:** Juvekar et al., Sci Data 2024, DOI 10.1038/s41597-024-03295-z.
- **Location:** `data/archive/remind/` (gitignored).
- **Layout:** `ReMIND-XXX/Preop/<SeriesNumber>_<MR|SEG>_<SeriesDescription>/*.dcm` plus `.complete` markers.
- Real DICOM with pixel spacing. **The skull is not stripped**, and faces were defaced.
- **Series names:** `3D_AX_T1_postcontrast`, `3D_AX_T1_precontrast`, `2D_AX_T2_FLAIR`, `3D_SAG_T2_SPACE`, `2D_COR_T2_BLADE`, …
- **SEG files:** binary, one segment each.
  - tumor = MANUAL; cerebrum and ventricles = AUTOMATIC (BrainLab).
  - Each SEG names its reference series. Example, ReMIND-001:
    - tumor drawn on **3D_SAG_T2_SPACE** (sagittal!)
    - ventricles and cerebrum on 3D_AX_T1_postcontrast
  - Many tumor SEGs reference T2/FLAIR rather than T1-post.
  - Some patients have `previous_resection_cavity` / `tumor_target` SEGs.
- **Sizes:** an MR series is about 22–39 MB (ReMIND-001). A whole Preop folder is 26–305 MB.
- **There are no gray/white-matter labels.**
- **Download is INCOMPLETE:**
  - 161 of 663 series (~2.6 GB, ReMIND-001…028). Total selection: 114 patients, ~10.9 GB.
  - `ReMIND-027/Preop/1_MR_3D_AX_T1_postcontrast` is missing (failed 3×).
  - The downloader was **not running** at handoff.
  - Resume (offer to run it in the background with `run_in_background`; check `.complete` counts first):
    `python -I "/c/Users/Priyansh Kuniyal/Plie/scripts/analysis/download_remind.py" --out "/c/Users/Priyansh Kuniyal/Plie/data/archive/remind" --workers 2`
  - TCIA resets connections (WinError 10054). The script retries with 15 s / 45 s backoff plus a second pass.
  - `manifest.csv` and `README.md` there have no final counts yet. Recount after a full run how many patients have tumor + cerebrum + ventricles on the same `3D_AX_T1_postcontrast`; earlier estimates conflicted (31 vs ≤ 50).

### Kaggle LGG (`data/archive/kaggle_3m/`; Buda/Mazurowski TCGA; CC BY-NC-SA 4.0)
- **Not DICOM:** 110 patients, 3,929 slices, 256×256 TIFFs.
  - RGB = pre-contrast | FLAIR | post-contrast. A missing sequence is replaced by a copy of FLAIR (pre missing in 6 cases, post in 10).
  - No headers and no pixel spacing, so areas are px only. Head width is 148–235 px.
- **Lesion masks:** `*_mask.tif` = FLAIR abnormality, {0,255}, radiologist-verified.
  - 35% of slices are positive. Median lesion size 1,529 px; 89% are a single blob.
- **Image properties:**
  - The skull is NOT stripped.
  - Intensity varies widely between patients, so use percentile windowing.
  - 143 slices are near-blank or at skull-base level, so a "brain present" gate is needed.
  - Lesion/brain FLAIR ratio: median 1.51, but < 1.1 in 5.7% of slices.
  - Gray and white matter separate poorly on FLAIR.
  - A few cases have misregistered non-FLAIR channels.
- `data.csv` laterality 1=Left, 3=Right matches 99/107 cases. That's my inference, not documented.
- **Still tracked in the PUBLIC repo** (with `data.csv`: age/sex/race/survival). Decide later whether to untrack, purge history, or leave it.
- Use it as a **px-only lesion Dice benchmark** and a dev path. Sequence can't be auto-detected, so treat the channels as known.

### Existing scripts (`scripts/analysis/`, stdlib only)
| Script | Purpose |
|---|---|
| `tiffmini.py` | TIFF reader |
| `scan_dataset.py` | dataset scan |
| `summarize.py` | summary stats |
| `contact_sheet.py` | image grid of slices |
| `registration_check.py` | checks channel alignment |
| `download_remind.py` | ReMIND downloader |

## Segmentation plan (TASKS §3, P0 unless noted)
1. **Load + window:** apply rescale slope/intercept, then percentile-window (e.g. 1st–99th inside the head) to 0–1.
2. **Head mask:** Otsu threshold → largest connected component → fill holes.
3. **Brain mask (skull strip):** classical approach is erode → largest component → dilate → fill, inside the head; tune per sequence. Then **skull = head − brain**.
   - When a doctor cerebrum SEG exists, use it as the brain boundary.
4. **Tissue classes inside the brain:** multi-Otsu (3 classes) → CSF / GM / WM. **The intensity order depends on the sequence:**
   | Sequence | Darkest → brightest |
   |---|---|
   | T1 (pre or post) | CSF < GM < WM |
   | T2 | WM < GM < CSF |
   | FLAIR | CSF dark, GM ≳ WM (poor contrast, so set GM/WM confidence low) |
5. **Lesion:**
   - Use the doctor/dataset label if present (`source: doctor`).
   - Otherwise flag robust brightness outliers on FLAIR/T2: median + k·MAD (median absolute deviation), with minimum-size and skull-rim filters. Confidence is low on T1.
6. **Exclusivity:** priority order **lesion > csf > white_matter > gray_matter > skull**. Then give each layer a confidence flag.
7. **Export:** C2 folder (nearest-neighbor only, anonymous IDs) and in-memory arrays for the API.
   - **Send FT 5 sample slices ASAP.**
8. **Sequence detection** (shared with MP):
   1. SeriesDescription keywords
   2. else TR/TE/TI rules
   3. else a default + warning
9. **Orientation:** derive `patient_left` from ImageOrientationPatient.
10. **"Brain present" gate:** reject near-blank or skull-base slices.
11. **Determinism:** no randomness; the same input gives byte-identical masks.
12. **P2:** keep a clean `segment(slice, meta) -> dict[layer_id, mask]` interface so a U-Net could replace it later.

### Validation plan (P1, but it's the strongest slide)
| Layer | Compare against |
|---|---|
| Tumor | ReMIND manual SEG |
| Skull | head − ReMIND cerebrum |
| CSF | ReMIND ventricles (ventricles are only *part of* CSF, so report recall/overlap carefully) |
| GM/WM | **SynthSeg** "silver standard", never called ground truth. Offline only, run by a teammate per `docs/prompts/synthseg-gm-wm.md`; tumor region masked out. |
| All layers | Synthetic phantom with known answers (P1: a generator of concentric regions + a lesion) |
| Lesion (bonus) | Kaggle px-based Dice |

- Report Dice/IoU per layer.
- **The app itself stays non-AI.**

## Proposed code layout (confirm with me)
`backend/cortexview/`:
- `dicom_io.py`: load series, group, detect SEG, sequence detection
- `segment.py`
- `layers.py`: reads `shared/layers.json`
- `export.py`: C2 folder writer

Plus:
- `scripts/seg/run_on_remind.py` and `scripts/seg/eval_dice.py`
- `tests/` with the phantom
- Later: `shared/contracts.py` (Pydantic) + `shared/contracts.ts` (hand-written mirror) once the contracts are approved.

## Still-open items (ask, don't assume)
- Was deleting `docs/finetune-brief.md` intentional?
- Install permission + venv location for numpy/scipy/scikit-image/pydicom/Pillow.
- Resume the ReMIND download?
- `docs/metrics-brief.md` for MP: does MP own only metrics (current assumption, since I own segmentation)?
- What to do about the Kaggle dataset + `data.csv` in the public repo.
- Proposed `finetune-brief.md` updates: new palette, C2 folder format (`meta.json`, `label.png`, `reference/`, anonymous IDs).
- README credits for dataset licenses (CC BY 4.0 ReMIND; CC BY-NC-SA 4.0 Kaggle; FLUX.2 klein Apache-2.0). The repo LICENSE is MIT.
- Open the PR for `shared-contracts` once `gh auth login` is done, or via the GitHub web UI.

## Your first steps
1. Run `git status`, `git log -5` and `git fetch`, and report any differences from the above.
2. Check the downloader isn't running, and count the `.complete` markers.
3. Ask me the install question. Then propose the segmentation module layout and a first milestone:
   - head + brain + skull masks on 3 ReMIND-001…005 axial T1-post slices
   - a contact sheet for visual checking
   - Dice of the brain mask vs the ReMIND cerebrum SEG
4. Wait for my OK before writing code.
