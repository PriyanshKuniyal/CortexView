# CortexView edit LoRA on Lightning: runbook

FLUX.2 [klein] base-9B, edit LoRA (guide.png -> illustration), Ostris ai-toolkit, Lightning Studio.
State as of 2026-10-11. Everything below "GPU session" still has to be run.

## Studio layout (all persistent, under `/teamspace/studios/this_studio`)

| Path | What |
|---|---|
| `Plie/` | repo checkout (at `c202ad5`; don't `git pull` there: `main` now carries `data/synthseg` NIfTI MRI copies) |
| `Plie/data/guides/remind/` | the 80 guides (`guide/ image/ labels/ meta/ splits.csv guides_sheet.png`), checksum-identical to the Windows copy |
| `flux2-klein-base-9b/` | base model (diffusers layout + `flux-2-klein-base-9b.safetensors`, 50 GB). Gated, FLUX non-commercial license |
| `ai-toolkit/` | ostris/ai-toolkit at `f7a1fb9` (unmodified) |
| `.cache/huggingface/hub/` | prefetched `Qwen/Qwen3-8B` (text encoder) and `ai-toolkit/flux2_vae`, which ai-toolkit loads from the Hub, not from the model folder |
| `cortexview_lora/` | work dir: `smoke_dataset/` now; later `candidates/`, `dataset/`, `output/` |

## What was verified on CPU

- Guides: 80 = 61 from the 40 train patients + 19 from the 13 test patients; IDs match across all four folders.
- ai-toolkit `f7a1fb9` registers `flux2_klein_9b`, reads `control_path`, encodes control images as edit references,
  and supports `--ctrl_img` in sample prompts. Config keys checked against its own edit example.
- Loading with a local `name_or_path`: transformer from the folder's single safetensors; text encoder and VAE from the
  Hub (prefetched, not gated, no token needed).
- `make_candidates.py`: `--split train` selects 61 guides; `Flux2KleinPipeline` exists and takes the arguments it passes.
- `fidelity_filter.py` and `build_dataset.py` ran end to end on 4 identity pairs (`cortexview_lora/smoke_dataset`).
- Not checkable without a GPU: importing ai-toolkit's model package (an unrelated OmniGen2 Triton kernel needs CUDA at
  import), model loading, VRAM, speed.

## Environment changes (the Studio's single conda env `cloudspace`, Python 3.14, torch 2.14.0+cu130)

- Snapshot before any change: `~/env_before_aitk.txt` (`pip freeze`).
- Installed ai-toolkit's requirements from `~/aitk_req_py314.txt`: `requirements_base.txt` with `matplotlib==3.10.1`
  relaxed to `>=3.10` (no Python 3.14 wheel); `scipy==1.12.0` from `requirements.txt` skipped (can't build on 3.14;
  the env's scipy 1.18.1 stays). torch, torchvision, numpy, scipy and pillow were not changed. `pip check` is clean.
- Added `torchaudio 2.11.0+cu130` (ai-toolkit imports it; installed with torch pinned at 2.14.0).
- ai-toolkit's README pins torch 2.13.0; the Studio has 2.14.0. If something breaks only on GPU, that's the first suspect.

## GPU session

Pick a 48 GB GPU (L40S) or bigger: candidate generation runs the 9B transformer and the 8B text encoder in bf16
without quantization (~35 GB). Training quantizes and needs less.

### 1. Check the GPU (seconds)
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/2**30)"
cd ~/ai-toolkit && python -c "import sys; sys.path.insert(0,'.'); from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein9BModel; print('ok')"
```

### 2. Smoke test (~10 steps; loads the model, caches latents, trains, saves, samples)
```bash
cd ~/ai-toolkit && python run.py /teamspace/studios/this_studio/Plie/scripts/lora/smoke_klein9b_edit.yaml
```
Pass = a `.safetensors` and one sample image in `cortexview_lora/smoke_output/cortexview_klein9b_smoke/`. Note peak VRAM.

### 3. Candidates (train patients only, base model, no LoRA). This is the long GPU step
61 guides x 3 samples x 50 steps at 1024 px = 183 images. Resumable after a spot interruption.
```bash
cd /teamspace/studios/this_studio/cortexview_lora
python ../Plie/scripts/lora/make_candidates.py --model /teamspace/studios/this_studio/flux2-klein-base-9b \
  --guides ../Plie/data/guides/remind/guide --splits ../Plie/data/guides/remind/splits.csv --split train --out candidates
```
Time the first few images before letting it run; turn the GPU off when it's done.

### 4. Filter and pick (CPU, GPU off)
```bash
python ../Plie/scripts/lora/fidelity_filter.py --guides ../Plie/data/guides/remind/guide --candidates candidates \
  --palette ../Plie/scripts/synthseg/palette.json
```
A person looks at `previews/` and writes 20–30 candidate filenames, max one per guide, into `picked.txt`.
```bash
python ../Plie/scripts/lora/build_dataset.py --picked picked.txt --guides ../Plie/data/guides/remind/guide \
  --candidates candidates --out dataset
```

### 5. Train (GPU)
```bash
cd ~/ai-toolkit && python run.py /teamspace/studios/this_studio/Plie/scripts/lora/train_klein9b_edit.lightning.yaml
```
2000 steps, a checkpoint and 4 test-patient samples every 250 steps in `cortexview_lora/output/`. Pick a checkpoint by eye.

### 6. Evaluate on held-out patients
`make_candidates.py`'s docstring evaluates with the distilled `flux2-klein-9b`, which is not on the Studio (separate
license acceptance on Hugging Face). With the base model instead:
```bash
python ../Plie/scripts/lora/make_candidates.py --model /teamspace/studios/this_studio/flux2-klein-base-9b \
  --lora output/cortexview_klein9b_edit_v1/<checkpoint>.safetensors --guides ../Plie/data/guides/remind/guide \
  --splits ../Plie/data/guides/remind/splits.csv --split test --samples 1 \
  --prompt "turn this brain segmentation map into a CRTXV1EW illustration" --out eval_lora
```
Score `eval_lora` (and a base-model run on the same test guides) with `fidelity_filter.py`; report both pass rates.
The guides are SynthSeg silver-standard references, not ground truth: no clinical-accuracy claims.
