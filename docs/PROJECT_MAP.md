# CortexView: project map

**What it is:** you upload a brain MRI slice (DICOM) and get back a color-coded medical illustration you can peel apart layer by layer. Pins on the image show measurements.
**Pitch:** "Generative where it helps, deterministic where it matters." The AI only decides how the image looks. Anatomy and every number come from masks and math.
**Event:** Banana Hacks 2026, online, Oct 9–12. Team lead and integration: Priyansh. Repo: `github.com/PriyanshKuniyal/Plie`.

## Pipeline

```
DICOM ─► preprocess (pydicom: sequence, pixel spacing, de-identify)
      ─► segmentation (classical CV) ─► labels.png (one label per pixel)
            ├─► metrics engine ─► data pins (math only)
            └─► guide.png ─► FLUX.2 klein + our LoRA ─► illustration ─► fidelity check ─► clipped to masks
```

## Labels and palette (frozen; changing them means retraining)

| ID | Layer | RGB |
|---|---|---|
| 0 | background | 0,0,0 |
| 1 | skull/scalp | 222,184,135 |
| 2 | gray matter | 196,120,170 |
| 3 | white matter | 245,240,225 |
| 4 | CSF / ventricles | 70,140,230 |
| 10+ | findings (tumor) | 230,40,40 |

The palette lives in `scripts/synthseg/palette.json` and `shared/layers.json`.

## Data

| Dataset | Use |
|---|---|
| **ReMIND** (TCIA, CC BY 4.0), downloaded from NCI Imaging Data Commons with `idc-index` | Main dataset. 53 pre-op patients with a T1 post-contrast scan plus tumor and cerebrum masks drawn on that same scan. |
| **Kaggle LGG** (`data/archive/kaggle_3m`) | Backup and extra lesion masks. TIFF format, no pixel spacing. |

**Known bug.** 27 of ReMIND's DICOM SEG files (24 patients) were mis-packed by dcmqi and decode scrambled with standard tools. This is fixed in `remind_synthseg.py` (`decode_binary_seg_frames`). Worth reporting upstream to TCIA/IDC.

## Stage 1: guides for training (done)

| Tool | Role |
|---|---|
| `scripts/synthseg/remind_synthseg.py` | Pipeline steps: `select → prepare → synthseg → build → guides` |
| **SynthSeg** (FreeSurfer 7.4.1 `mri_synthseg`, robust mode) | Gray matter, white matter and CSF labels. Runs natively on Windows (Python 3.11, TF 2.15 on CPU) through `mri_synthseg_native.py`. |
| `dcm2niix`, `highdicom`, `pydicom` | Convert DICOM to NIfTI; turn SEG files into masks |
| `SimpleITK`, `nibabel`, `scipy` | Resampling, reorientation, dilation |
| `check_seg_alignment.py` | Checks that the tumor overlay sits on the lesion |

**Output:** 80 guides (53 with tumor, 27 without) from 53 patients, split 40 train / 13 test by patient.
- Location: `data/guides/remind/` (gitignored). Report: `data/guides/OVERNIGHT_REPORT.md`.
- Work dir (DICOM, NIfTI, venv): `C:\Users\simpl\remind_work`, outside OneDrive.
- These are **silver-standard** references (SynthSeg output), never ground truth.

## Stage 2: LoRA fine-tune (next)

| Tool | Role |
|---|---|
| **FLUX.2 [klein] base-9B** (gated; FLUX non-commercial license) | Training target: edit LoRA, `guide.png` in, illustration out |
| **FLUX.2 [klein] 9B** (distilled, 4 steps) | Fast inference and evaluation with the trained LoRA |
| **ostris/ai-toolkit** | Trainer. `arch: flux2_klein_9b`; `control_path` makes it an edit LoRA |
| **Lightning AI Studio** (L40S on spot, ~$5 credits) | GPU. Base-9B is already downloaded to `/teamspace/studios/this_studio/flux2-klein-base-9b` |
| **Hugging Face** | Model downloads (token required); LoRA upload later |
| `scripts/lora/make_candidates.py` | Base model renders the train guides; also runs evaluation with `--lora` on the test guides |
| `scripts/lora/fidelity_filter.py` | Per-label IoU of output vs. guide; pass rate is the pitch number |
| `scripts/lora/build_dataset.py` | Turns picked pairs into `reference/` + `target/` folders with captions. Trigger word: `CRTXV1EW` |
| `scripts/lora/train_klein9b_edit.yaml` | Config: rank 32, lr 1e-4, 2000 steps, checkpoint every 250 |

**Plan:**
1. Render about 3 candidates per train guide.
2. Filter by IoU, then a human picks 20–30 pairs.
3. Train.
4. Choose a checkpoint by eye.
5. Report fidelity pass rate, base vs. LoRA, on the test patients.

## App

| Part | Stack |
|---|---|
| Backend | FastAPI with Pydantic models. `POST /api/analyze` returns layers and metrics in about 1 s; `GET /api/render/{id}` is polled for the illustration. |
| Frontend | Vite, React, TypeScript. TS types are generated from OpenAPI. Design reference: the CortexView prototype in `docs/design/cortexview_artifact.html`. |
| Serving | Cloudflare Tunnel (Workers AI was the baseline renderer) |
| Rules | No patient data stored or logged. Patient text uses fixed templates only. "Educational aid, not for diagnosis" is always visible. |

## Dev tools

- **Claude Code**: runs the work in the repo, driven by prompts. `PROMPT_synthseg_and_guides.md`, `PROMPT_train_lora.md` and `PROMPT_port_frontend.md` are pasted in, not stored in the repo.
- **Cursor / VS Code**, connected to Lightning over SSH.
- **Laptops:** RTX 5060 (8 GB VRAM, 16 GB RAM) and RTX 5050. Both are too small to train 9B, so training runs on Lightning.

## Repo layout

```
Plie/
  scripts/synthseg/   remind_synthseg.py, mri_synthseg_native.py, check_seg_alignment.py, palette.json, requirements.txt
  scripts/lora/       make_candidates.py, fidelity_filter.py, build_dataset.py, train_klein9b_edit.yaml
  scripts/analysis/   ReMIND download and dataset analysis
  data/               archive/ (datasets), guides/ (gitignored)
  shared/layers.json  layer names and colors (source of truth)
  frontend/  docs/
```

**Gitignored:** `license.txt` (FreeSurfer, personal), `.env`, `data/guides/`, `data/archive/remind/`, `*.safetensors`, `output/`.

## Status (Oct 11, morning)

- ✅ Palette frozen
- ✅ SynthSeg and guides finished
- ✅ Base-9B downloaded on Lightning
- ⏳ Distilled 9B: needs its separate license accepted on Hugging Face
- ⏳ LoRA training: run `PROMPT_train_lora.md`
- ⏳ Frontend port of the prototype: run `PROMPT_port_frontend.md`
- ⏳ Commit `scripts/synthseg/*` and `.gitignore`
- ⏳ Fix the "4B · Apache-2.0" copy in the prototype: it should say 9B, non-commercial license
