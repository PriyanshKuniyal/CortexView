# CortexView: Task List

**Owners**
- **PK** = Priyansh (lead and integration)
- **FT** = fine-tuning teammate ([finetune-brief.md](finetune-brief.md))
- **MP** = metrics and preprocessing teammate (metrics-brief.md)
- **FE** = frontend owner (TBD; PK by default)
- **ALL** = whole team

**Priority:** P0 = the demo breaks without it · P1 = expected by judges · P2 = stretch

**Status:** `[ ]` todo · `[~]` in progress · `[x]` done

## Critical path
The rest of the work is blocked until these are done.

1. **Dataset analyzed (§1).** This feeds the shared contracts.
2. **Shared contracts frozen (§2).** This unblocks FT, MP and FE.
3. **Segmentation produces masks in the agreed folder format (§3).** This unblocks FT's Phase 2 and MP's metrics.
4. **Backend `/api/analyze` returns real masks and metrics (§6).** This unblocks the frontend against real data.
5. **End-to-end run on the demo cases (§8).**

---

## 0. Repo and team setup
- [ ] **P0 · PK**: Verify git pull/push to `github.com/PriyanshKuniyal/Plie`. An earlier fetch hit a network reset.
- [ ] **P0 · PK**: Add a `.gitignore` covering `.env`, `data/`, `slices/`, `datasets/`, `output/`, `*.safetensors`, `*.dcm`, `node_modules/`, `__pycache__/` and `.venv/`.
- [ ] **P0 · PK**: Agree the repo layout: `backend/`, `frontend/`, `render/` (FT), `docs/`, `scripts/`.
- [ ] **P1 · ALL**: Branch-per-person workflow, with PRs or merges into `main` and no direct force-pushes.
- [ ] **P1 · PK**: Team channel with each person's contact details, plus a shared place for secrets. Never put secrets in git.

## 1. Dataset
- [~] **P0 · PK**: Get the dataset, store it in `data/raw/`, and keep it out of git.
- [ ] **P0 · PK**: Analyze it: format, sequences, geometry and pixel spacing, ground-truth labels, PHI, license terms, and quality.
  Use the starter prompt from this chat.
- [ ] **P0 · PK**: Confirm the license allows sending de-identified PNGs to Cloudflare Workers AI. FT's Phase 2 depends on this.
- [ ] **P0 · PK**: Decide on 2D slices only, or volumes plus a slice picker.
- [ ] **P0 · PK**: Define the patient-level split (train/eval/demo) and share it with FT and MP.
  Eval patients must never be used in training.

## 2. Shared contracts (freeze early; changes need everyone's OK)
- [ ] **P0 · PK**: **Layer registry** as the single source of truth for all 5 layers (`skull`, `gray_matter`, `white_matter`, `csf`, `lesion`).
  Each layer has an id, display name, guide RGB, display RGBA, peel order and description.
  Ship it as `shared/layers.json`, read by both Python and TS.
- [ ] **P0 · PK**: Finalize the **guide palette**, replacing the provisional one in finetune-brief §5.2, and tell FT.
- [ ] **P0 · PK**: **Slice/mask folder format** (finetune-brief §5.1): confirm it, or adjust it after the dataset analysis.
- [ ] **P0 · PK + MP**: **Metrics schema**: the Pydantic model and its JSON shape, as produced by MP and consumed by the pins.
- [ ] **P0 · PK + FT**: **Render contract**: confirm `render_slice()` / `RenderResult` (finetune-brief §10).
- [ ] **P0 · PK**: **API schema** for `/api/analyze`: request, response and error shapes, plus generated or hand-written TS types for the frontend.
- [ ] **P1 · PK**: **Patient-friendly label templates** (fixed strings with number slots, no generated text). Agree them with MP.

## 3. Segmentation masks (unowned: PK takes it unless reassigned)
- [ ] **P0**: Head mask: Otsu threshold, then the largest connected component, then fill holes.
- [ ] **P0**: Skull-strip into a brain mask, and derive skull/scalp as head minus brain.
- [ ] **P0**: Tissue classes CSF / GM / WM via multi-Otsu inside the brain.
  The intensity ordering depends on the detected sequence (T1 / T2 / FLAIR).
- [ ] **P0**: Lesion mask:
  - Use the dataset's **ground-truth labels** for the demo cases if they exist.
  - Otherwise flag robust intensity outliers (median + k·MAD), with minimum-size and rim filters.
- [ ] **P0**: Make the masks mutually exclusive, and give each layer a confidence flag (`high` / `medium` / `low`) based on sequence fit.
- [ ] **P0**: Export in the agreed folder format (`mri.png`, `guide.png`, `masks/*.png`).
  - Nearest-neighbor resizing only, and no PHI in filenames.
  - **Send FT 5 sample slices ASAP**, then the full split.
- [ ] **P1**: Measure Dice/IoU against ground truth where available, and report it per layer.
- [ ] **P1**: Synthetic phantom DICOM generator (concentric regions plus a lesion) for tests and offline demos.
- [ ] **P2**: A clean interface so a trained U-Net could replace the classical approach later.

## 4. Preprocessing and metrics (delegated to MP; PK tracks)
- [ ] **P0 · MP**: DICOM loader: rescale slope/intercept, middle frame of multi-frame files, pixel spacing, strip PHI, reject non-MR files.
- [ ] **P0 · MP**: Sequence detection from header keywords, falling back to TR/TE/TI and then to a default with a warning.
- [ ] **P0 · MP**: Per-layer metrics:
  - area in mm²/cm²
  - % of intracranial area
  - intensity stats
  - centroid, bounding box, number of connected components
  - lesion largest diameter
  - volume, if 3D
- [ ] **P0 · MP**: Pin anchor for each layer: the point inside the mask farthest from its edge (distance-transform maximum).
- [ ] **P1 · MP**: Validate the metrics (our masks vs. ground-truth masks), plus unit tests on hand-computed masks.
- [ ] **P0 · PK**: Write `docs/metrics-brief.md` and hand it to MP.
  Use the new-chat starter prompt, and state whether MP owns segmentation (current assumption: no, PK does).

## 5. Generative rendering (delegated to FT; PK tracks)
- [ ] **P0 · FT**: D1 + D2: `render_slice()` on Workers AI and `fidelity.py`. **Integrate these before any training.**
- [ ] **P1 · FT**: D3: fine-tuned edit LoRA on FLUX.2 klein base 4B.
- [ ] **P1 · FT**: D4: local server for the LoRA, exposed through Cloudflare Tunnel.
- [ ] **P0 · FT**: D5: eval report comparing baseline and LoRA (table + grid). It's needed for the slides either way.
- [ ] **P0 · FT**: D6: pre-rendered illustrations for the demo cases.
- [ ] **P0 · PK**: Unblock FT: Cloudflare account and token, sample slices, final palette.

## 6. Backend orchestration (PK)
- [ ] **P0**: FastAPI scaffold with `/api/health`, CORS, an upload size cap and `.env` config (`RENDER_BACKEND`, `LOCAL_RENDER_URL/KEY`, `CF_*`).
- [ ] **P0**: `POST /api/analyze`: ingest → segment → metrics → response. Everything stays in memory and nothing is persisted.
- [ ] **P0**: **Async rendering:**
  - Return masks and metrics immediately with a `job_id`.
  - `GET /api/render/{job_id}` returns the illustration when it's ready.
- [ ] **P0**: **Layer cutting:** split the illustration into 5 transparent layer PNGs using the masks, so edges match the segmentation exactly.
- [ ] **P0**: **Render fallbacks:** local LoRA via the tunnel → Workers AI → flat-color render, each with a timeout (≈30 s) and the backend used reported in the response.
- [ ] **P0**: **Demo cache:** look up known demo inputs by hash and return the stored result instantly, so the demo still works if the network fails.
- [ ] **P1**: Structured errors: not DICOM, not MR, unsupported geometry, render failed. All logs must be PHI-free.
- [ ] **P1**: API tests: phantom upload returns 200 with 5 layers; a bad file returns 422.

## 7. Frontend (FE, TBD)
- [ ] **P0**: Vite + React + TS scaffold, with `/api` proxied to the backend.
- [ ] **P0**: Upload dropzone for `.dcm` files, with loading, progress and error states.
- [ ] **P0**: **Layer viewer:** base slice plus stacked layer PNGs, keeping the aspect ratio.
- [ ] **P0**: **Peel slider** across the 5 layers, outside in, with fade transitions. It must work with the keyboard.
- [ ] **P0**: **Data pins** anchored in image coordinates so they stay in place on resize. Clicking one opens a metrics card filled from fixed templates.
- [ ] **P0**: Disclaimer that's always visible: an educational aid, not for diagnosis.
- [ ] **P1**: Before/after toggle between the raw MRI and the illustration.
- [ ] **P1**: Legend with per-layer on/off, opacity and a confidence badge.
- [ ] **P1**: Metrics panel showing all layers and the detected sequence.
- [ ] **P1**: Fidelity badge showing the score and which render backend was used.
- [ ] **P1**: While the illustration is pending, show the masks in flat colors and swap in the illustration when it arrives.
- [ ] **P2**: Patient view vs. doctor view (simplified vs. detailed numbers).
- [ ] **P2**: Slice scrubber, if the dataset is 3D.

## 8. Integration and testing (ALL, led by PK)
- [ ] **P0**: An end-to-end script that uploads each demo DICOM and checks for 5 layers, metrics, an illustration and a fidelity pass.
- [ ] **P0**: Integration checkpoint 1: real masks and metrics showing in the UI, before rendering is wired in.
- [ ] **P0**: Integration checkpoint 2: Workers AI renders showing in the UI.
- [ ] **P1**: Integration checkpoint 3: the LoRA via the tunnel, if D5 says it beats the baseline.
- [ ] **P1**: Test on a second machine and network, which also checks the tunnel and CORS.
- [ ] **P0**: **Code freeze**, set by the deadline (TBD). After it, bug fixes only.

## 9. Demo and pitch (ALL)
- [ ] **P0 · PK**: Choose 3–5 demo cases (at least 1 with a clear lesion and 1 normal) and cache them fully.
- [ ] **P0 · PK**: Decide where the demo runs: a local laptop, or hosted (frontend on Cloudflare Pages + backend).
- [ ] **P0 · ALL**: Slides:
  - the problem
  - the two-path architecture ("generative where it helps, deterministic where it matters")
  - the fidelity gate
  - the teacher → student fine-tune
  - D5 results
  - limitations and ethics
- [ ] **P0 · ALL**: A 2–3 minute backup demo video.
- [ ] **P1 · PK**: README covering what the project is, the architecture diagram, setup and run instructions, the team, and credits and licenses (dataset, FLUX.2 klein Apache-2.0, any CC assets).
- [ ] **P1 · ALL**: Rehearse the demo twice, including the fallback path.

## 10. Open decisions
- [ ] Hackathon deadline and submission requirements (affects the timeline and code freeze).
- [ ] Frontend owner.
- [ ] Whether segmentation stays with PK or moves to MP.
- [ ] 2D only vs. 3D (after the dataset analysis).
- [ ] Hosting for the demo.
