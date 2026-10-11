# Local LoRA training: FLUX.2 [klein] base-4B on the 8 GB laptop

Replaces the Lightning 9B plan (`docs/lora-lightning.md`). Base-4B is Apache-2.0 and not gated.

## Layout

Heavy files stay out of the OneDrive-synced repo.

| Path | What |
|---|---|
| `C:\models\flux2-klein-base-4b\` (`$PLIE_MODELS`) | base model: `scripts/lora/download_klein4b.py`; `make_candidates.py` adds a diffusers `transformer/` on first run |
| `C:\Users\simpl\lora_env\` | Python 3.12 env: torch 2.13.0+cu130, `third_party/ai-toolkit/requirements.txt` |
| `third_party/ai-toolkit/` | trainer checkout, unmodified (gitignored) |
| `lora_work/` | candidates, dataset, checkpoints, samples (gitignored) |

## Setup

```sh
uv venv C:/Users/simpl/lora_env --python 3.12
uv pip install --python C:/Users/simpl/lora_env/Scripts/python.exe torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python C:/Users/simpl/lora_env/Scripts/python.exe -r third_party/ai-toolkit/requirements.txt
C:/Users/simpl/lora_env/Scripts/python.exe scripts/lora/download_klein4b.py      # ~16 GB into C:/models
```

## Run (as done on 2026-10-11)

```sh
PY=C:/Users/simpl/lora_env/Scripts/python.exe
# 1. candidates: 61 train guides x 2 samples, ~50-65 s each (nf4 + CPU offload)
$PY scripts/lora/make_candidates.py --model C:/models/flux2-klein-base-4b --low_vram --size 512 --steps 28 --samples 2 \
    --guides data/guides/remind/guide --splits data/guides/remind/splits.csv --split train --out lora_work/candidates
# 2. picks: the run used every guide's __s0 sample, no fidelity filter (see Results for why that should change)
$PY scripts/lora/fidelity_filter.py --guides data/guides/remind/guide --candidates lora_work/candidates \
    --palette scripts/synthseg/palette.json --out_csv lora_work/fidelity.csv --preview_dir lora_work/previews \
    --picked_out lora_work/picked.txt
# 3. dataset (guides and candidates are already 512 px)
$PY scripts/lora/build_dataset.py --picked lora_work/picked.txt --guides data/guides/remind/guide \
    --candidates lora_work/candidates --out lora_work/dataset --size 512
# 4. train: 2000 steps, ~4.1 s/step, checkpoint + 4 test-patient samples every 250 steps
$PY scripts/lora/run_aitk.py scripts/lora/train_klein4b_edit.yaml
```

Needs most of the 15 GB RAM while the model loads and quantizes: close games/Discord first. A background job was
killed once for low memory.

The generation prompt in `make_candidates.py` names the palette colors. Without that, base-4B recolors freely
(cyan skull, pink white matter, white background, tumor dropped), and nothing passes the fidelity filter.

## Results: `cortexview_klein4b_edit_v1`

61 pairs, so one epoch = 61 steps; 2000 steps = ~33 epochs.

| Step | Test samples (held-out patients, trigger prompt only) |
|---|---|
| 0 | flat neon colors, no style |
| 250 | style appears |
| **500** | **best**: palette colors, shading, outlines, no false findings on the no-tumor patient (ReMIND-002) |
| 750 | colors wash out to lavender/white |
| 1000-2000 | **paints a red tumor into the cream midbrain** of the no-tumor patient, and of ReMIND-013 |

Use `lora_work/output/cortexview_klein4b_edit_v1/cortexview_klein4b_edit_v1_000000500.safetensors`.
Later checkpoints invent findings, the one error the app must never make.

Likely cause: unfiltered picks, where some targets have red where the guide has none. Next run: fidelity-filtered picks
(step 2 as written, which rejects them) and `steps: 600`.

## Not done yet

The backend does not run FLUX yet: nothing in `backend/cortexview/` loads the base model or the LoRA. Inference is
`make_candidates.py --low_vram --lora <checkpoint> --prompt "turn this brain segmentation map into a CRTXV1EW illustration"`,
~1 min per 512 px image on this laptop.
