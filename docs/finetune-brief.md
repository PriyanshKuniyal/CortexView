# CortexView: Image Model Fine-Tuning Brief

**Owner:** you (the image-model track)
**Hardware:** your RTX 5060 (8 GB VRAM) for training, and Cloudflare Workers AI for hosted generation
**Status of the rest of the app:** being built in parallel. You'll get slices and masks from the main pipeline in the format described in §5.

---

## 1. The project in 60 seconds

CortexView is a hackathon project with the theme *generative image AI solving a real-world problem*.

A patient's brain MRI slice is uploaded. Our deterministic code splits it into 5 tissue layers (masks). **Your model turns those masks into a clean, color-coded medical illustration.** The user then peels the layers away with a slider. Data pins show clinical numbers such as area and % of brain, and those numbers come from math on the masks, never from AI.

```
MRI slice ─► segmentation (deterministic) ─► masks ──┬─► metrics engine ─► data pins (numbers)
                                                     │
                                                     └─► guide image ─► YOUR MODEL ─► illustration
                                                                                         │
                                                       fidelity check + clip to masks ◄──┘
                                                                     │
                                                                     ▼
                                                     5 illustrated layers in the viewer
```

**Ground rule:** the model never decides *where* anatomy is. The masks decide shape and position, and the model only decides the *look*. We clip its output to the masks and score how well it respected them. That's our answer to "GenAI hallucinates", and it's a central part of the pitch.

---

## 2. What you deliver (definition of done)

| # | Deliverable | Priority |
|---|---|---|
| D1 | `render_slice()` working on **Workers AI**, with the best prompt and reference recipe (baseline) | **Must** |
| D2 | `fidelity.py`: scores a generated image against the masks and decides pass or retry | **Must** |
| D3 | A LoRA fine-tuned on **FLUX.2 [klein] base 4B**: `cortexview-edit-v1.safetensors` | Should |
| D4 | A local inference server for the LoRA, exposed with **Cloudflare Tunnel**, using the same interface as D1 | Should |
| D5 | An eval report: baseline vs. LoRA on 20 held-out slices (scores table + side-by-side image grid) | **Must** (even if the LoRA loses) |
| D6 | 3–5 demo slices pre-rendered and cached, as insurance on stage | **Must** |

**Order of work:** D1 and D2 must be locked and integrated **before** you start training. If time gets tight, cut D3 and D4, never D1, D2, D5 or D6.

---

## 3. Constraints to know before you start

1. **Workers AI cannot run a custom image LoRA.** Its LoRA support covers text models only (Mistral/Gemma/Llama). So:
   - **Workers AI** gives us the no-training baseline (D1), and generates training targets as the "teacher" (§6).
   - **Your LoRA** runs on your own GPU and is served to the app through a Cloudflare Tunnel (§9).
2. **Model choice: FLUX.2 [klein] 4B.**
   - Train on `black-forest-labs/FLUX.2-klein-base-4B`, the undistilled version, which BFL recommends for fine-tuning. It's Apache 2.0.
   - Run inference on `black-forest-labs/FLUX.2-klein-4B`, the distilled version, at 4 steps. Hugging Face reports that LoRAs trained on base usually work better on the distilled model.
   - **Not the 9B:** training it needs about 22 GB of VRAM, and its license is non-commercial. We still *use* the 9B on Workers AI as the teacher.
3. **8 GB of VRAM is below BFL's stated minimum (12 GB) for klein-4B LoRA training.** You'll need every memory-saving option (§7). If it doesn't train within **1 hour** of trying, switch to renting a 24 GB GPU (RunPod / Vast.ai, roughly $0.4–0.8/h). A run takes about 1 h, so it's a few dollars. Don't burn a day fighting OOM errors.
4. **Workers AI image-input limits** (FLUX.2 klein):
   - Up to 4 reference images, sent as `input_image_0` … `input_image_3`.
   - **Each must be smaller than 512×512.** Resize to a longest side of **496 px**.
   - Steps are fixed at 4.
   - Width and height: 256–1920.
5. **Budget:**
   - The free tier is 10,000 neurons/day, which is about 80 klein-4B images at 1024² per day.
   - klein-9B costs about **$0.015/image** beyond roughly 7 free per day.
   - flux-2-dev is billed per step and costs more.

---

## 4. Phase 0: Setup (≈1–2 h)

### 4.1 Cloudflare Workers AI

You don't need to write a Cloudflare Worker. We call Workers AI's **REST API** directly from Python.

1. Create a free account at https://dash.cloudflare.com.
2. In the dashboard, go to **AI → Workers AI → Use REST API**:
   - Click **Create a Workers AI API Token**, review it, create it, and **copy the token**. It's shown only once.
   - On the same page, copy your **Account ID**.
3. Create a `.env` file in your working folder. **Never commit it.**
   ```
   CF_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   CF_API_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
4. Install the client dependencies: `pip install requests python-dotenv pillow numpy`
5. Save this as `cf_client.py`:
   ```python
   import base64, os, requests
   from dotenv import load_dotenv
   load_dotenv()

   URL = "https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}"
   KLEIN_4B = "@cf/black-forest-labs/flux-2-klein-4b"
   KLEIN_9B = "@cf/black-forest-labs/flux-2-klein-9b"

   def flux_generate(prompt, refs=(), model=KLEIN_4B, width=768, height=768,
                     seed=None, guidance=None, timeout=120) -> bytes:
       """refs: list of PNG bytes, max 4, each < 512x512. Returns PNG/JPEG bytes."""
       # FLUX.2 on Workers AI takes multipart/form-data, even for text-only prompts.
       fields = {"prompt": (None, prompt),
                 "width": (None, str(width)), "height": (None, str(height))}
       if seed is not None:
           fields["seed"] = (None, str(seed))
       if guidance is not None:
           fields["guidance"] = (None, str(guidance))
       for i, png in enumerate(refs):
           fields[f"input_image_{i}"] = (f"ref{i}.png", png, "image/png")
       r = requests.post(
           URL.format(acct=os.environ["CF_ACCOUNT_ID"], model=model),
           headers={"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}"},
           files=fields, timeout=timeout)
       r.raise_for_status()
       body = r.json()
       if not body.get("success", True):
           raise RuntimeError(body.get("errors"))
       return base64.b64decode(body["result"]["image"])

   if __name__ == "__main__":
       img = flux_generate("clean medical textbook illustration of an axial brain "
                           "cross-section, plain black background, no text")
       open("smoke_test.png", "wb").write(img)
       print("ok -> smoke_test.png")
   ```
6. Run `python cf_client.py`. You should get `smoke_test.png`.
   - If you get an error about the response shape, `print(r.text)` once and adjust the `result.image` line.
   - A 401 or 403 means the token is wrong or missing Workers AI permissions.
7. Usage and costs: **Dashboard → AI → Workers AI** shows neuron usage.

### 4.2 Local GPU environment (Windows + RTX 5060)

1. Update the **NVIDIA driver** to the latest version.
2. Install **Python 3.12**, **Git** and **Node.js 20+**.
3. Install **ai-toolkit**, the trainer recommended by BFL:
   ```
   git clone https://github.com/ostris/ai-toolkit
   cd ai-toolkit
   run_windows.bat
   ```
   `run_windows.bat` sets up the virtual environment and dependencies. If it misbehaves, follow the manual install in the repo README, or the "AI-Toolkit-Easy-Install" script linked there.
4. **Check that PyTorch supports your Blackwell GPU.** The RTX 50 series is `sm_120`. Run this inside the ai-toolkit virtual environment:
   ```python
   import torch
   print(torch.__version__, torch.cuda.get_device_name(0),
         torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
   ```
   You need to see `(12, 0)`, and `sm_120` in the arch list. If you get *"no kernel image is available for execution on the device"*, your torch build is too old: reinstall it from a cu128 or newer index (the ai-toolkit README uses cu130).
5. Start the UI:
   ```
   cd ui
   npm run build_and_start
   ```
   Then open http://localhost:8675.
6. Set up Hugging Face:
   - Create an account and run `huggingface-cli login`.
   - Open https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B and https://huggingface.co/black-forest-labs/FLUX.2-klein-4B. If either page asks you to accept terms, accept them.
7. Install the inference dependencies:
   ```
   pip install git+https://github.com/huggingface/diffusers.git peft accelerate fastapi uvicorn python-multipart
   ```
   **`peft` is required.** Without it, `load_lora_weights()` fails.

---

## 5. Phase 1: Baseline on Workers AI, no training (≈3–4 h) **MUST**

**Goal:** the best result possible with zero training. This becomes the fallback, and the bar your LoRA has to beat.

### 5.1 Inputs you'll receive from the main pipeline

The main team will send about 5 sample slices first, and the full set once the dataset is processed.

```
slices/<patient_id>_<slice_idx>/
  mri.png                    # 8-bit grayscale, windowed, square
  guide.png                  # flat-color segmentation map (palette below), same size, black background
  masks/skull.png            # 0/255, one per layer, mutually exclusive
  masks/gray_matter.png
  masks/white_matter.png
  masks/csf.png
  masks/lesion.png           # may be empty
```

### 5.2 Guide palette

This palette is provisional. It's kept in one shared `palette.py`, synced with the main app.

| Layer | id | RGB |
|---|---|---|
| Skull / scalp | `skull` | (232, 196, 140) tan |
| Gray matter | `gray_matter` | (214, 92, 120) rose |
| White matter | `white_matter` | (120, 160, 230) blue |
| CSF / ventricles | `csf` | (90, 210, 160) mint |
| Suspected lesion | `lesion` | (255, 140, 0) orange |
| Background | none | (0, 0, 0) |

**Always resize the guide with nearest-neighbor** (`Image.NEAREST`). Bilinear resizing creates in-between colors that don't belong to any layer.

### 5.3 Experiments to run

Log every run in `experiments.csv` with these columns: `id, model, refs, prompt_id, seed, guidance, mean_iou, phantom_lesion_pct, latency_s, notes`.

| Exp | Refs | Model | What it tests |
|---|---|---|---|
| E1 | `[guide]` | klein-4b | Can it follow the map? |
| E2 | `[guide, mri]` | klein-4b | Does the real MRI add realistic detail (folds/sulci)? |
| E3 | Best of E1/E2 | klein-9b | Is the bigger model worth $0.015/image? |
| E4 | Best so far | Best so far | Guidance sweep (e.g. 2, 3.5, 5) × 3 seeds |

Starting prompt for E1 (iterate on it):

> Redraw image 0 as a clean medical textbook illustration of an axial brain cross-section. Keep every region exactly where it is in image 0, with the same outline and the same color per region: tan = skull, rose = gray matter, blue = white matter, mint = fluid spaces, orange = lesion. Soft shading and subtle tissue texture inside each region. Plain black background. No text, no labels, no arrows.

For E2, add: *"Use image 1 only for fine anatomical detail such as folds; do not change any region boundaries."*

**Never put numbers, measurements or diagnoses in prompts.** Images must contain no text.

### 5.4 Fidelity score (`fidelity.py`, D2)

1. Resize the generated image to the guide's size.
2. Convert both to Lab color. For each pixel inside the head, assign the **nearest palette color**, which gives a predicted label map. Ignore near-black pixels.
3. Compute per-layer **IoU** against the masks. `mean_iou` is the average over the layers present in that slice.
4. Compute **`phantom_lesion_pct`**: pixels predicted as lesion *outside* the lesion mask, as a % of brain pixels. This must be about 0, because a phantom tumor is the worst possible failure.
5. **Gate:** pass if `mean_iou ≥ 0.80` and `phantom_lesion_pct < 0.5`.
   - On a fail, retry with a new seed, up to 2 retries.
   - If it still fails, fall back to a flat-color render, which is the guide with light shading.
   - Tune the thresholds on baseline results.

After the gate passes, the main app multiplies the image by each mask to cut out the 5 layers, so edges always match the segmentation exactly.

**Phase 1 is done when** `render_slice()` (§10) works on Workers AI, the best recipe is written up in the README, and baseline scores exist for the 20 eval slices.

---

## 6. Phase 2: Build the training set (≈3–4 h) *Should*

**We're training an *edit* LoRA, not a style LoRA.** A style LoRA only learns a look. An edit LoRA learns "take this map and render it in our look **without moving anything**", which is exactly our task.

**Approach: teacher → student distillation with a fidelity filter.**

1. **Split by patient.** Use 60–80 training slices from varied patients and slice levels, with about 40% containing a lesion. Hold out **20 eval slices from different patients**. Never train on an eval patient.
2. **Generate targets with the teacher.**
   - Run the best Phase-1 recipe on **klein-9b** with 3 seeds per training slice. That's about 240 images ≈ $3.60.
   - Score each with `fidelity.py`, and keep the best passing image per slice.
3. **Review by hand.** Make a contact sheet of the kept images and reject anything anatomically weird. Aim for **≥ 50 accepted pairs** (50–200 is the recommended range for edit LoRAs).
4. **Clean the targets.** Multiply each target by the union of its masks so the background is pure black. The student then learns clean edges.
5. **Captions.** Use a made-up trigger word, `CRTXVW`, followed by an instruction. Rotate through 5–10 phrasings, for example:
   - `CRTXVW render this segmentation map as an anatomical illustration, keep every region's shape and color`
   - `CRTXVW turn this color-coded brain map into a medical textbook drawing without moving any boundary`
   - Don't describe the style in words, so the LoRA learns it from the images.
6. **Folder layout** (ai-toolkit pairs files by filename):
   ```
   datasets/cortexview_edit/
     reference/  <id>.png           # guide.png, 512x512, nearest-neighbor resized
     target/     <id>.png  <id>.txt # teacher output + caption
   ```

---

## 7. Phase 3: Train the LoRA (≈2–4 h incl. setup) *Should*

In the ai-toolkit UI, go to **New Job**:

- **Model:** FLUX.2 klein **base 4B** (`black-forest-labs/FLUX.2-klein-base-4B`, arch `flux2_klein_4b`).
- **Dataset:** folder `datasets/cortexview_edit/target`, control/reference path `datasets/cortexview_edit/reference`.

**Starting settings.** These come from BFL and Hugging Face guidance, adjusted for 8 GB. Option names vary by ai-toolkit version: use the UI toggles, or start from the klein example in `config/examples/` if you prefer YAML.

| Setting | Value |
|---|---|
| Trigger word | `CRTXVW` |
| Resolution | 512 |
| Batch size | 1 |
| LoRA rank / alpha | 16 / 16 (raise to 32 if memory allows) |
| Learning rate | 1e-4 |
| Steps | 2000, save a checkpoint every 250 |
| Optimizer | AdamW 8-bit |
| Gradient checkpointing | On |
| Quantize transformer | On (float8) |
| Text encoder | Quantize **and** cache text embeddings (unload it after caching) |
| Cache latents | On (to disk) |
| Low-VRAM mode | On |
| Samples | Every 250 steps, using 3 **eval** guides with the training caption |

**8 GB go/no-go (timebox: 1 hour):**
- If it OOMs at startup with every option on, try rank 8, then switch to Plan B.
- Windows can silently spill VRAM into system RAM instead of crashing. If iterations/second suddenly drops 5–10×, that's what is happening. For an honest OOM signal, set **NVIDIA Control Panel → Manage 3D settings → CUDA – Sysmem Fallback Policy → Prefer No Sysmem Fallback**.
- **Plan B:** rent a 24 GB GPU (RTX 4090 or L4) on RunPod or Vast.ai with an ai-toolkit template, upload the dataset folder, and use the same settings, with rank raised to 32 and caching left on. Expect about 1 h.

**Choosing a checkpoint:** don't go by the loss. Use the samples plus `fidelity.py` on the eval set. Edit LoRAs typically peak around **steps 1000–1750**. Checkpoints are saved under `output/<job_name>/`.

---

## 8. Phase 4: Evaluate (≈1–2 h) **MUST** (D5)

Inference with the LoRA uses the **distilled** 4B, 4 steps, with CPU offload so it fits in 8 GB:

```python
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                          torch_dtype=torch.bfloat16)
pipe.load_lora_weights("output/cortexview_edit/<checkpoint>.safetensors")
pipe.enable_model_cpu_offload()

guide = Image.open("slices/<id>/guide.png").convert("RGB").resize((768, 768), Image.NEAREST)
img = pipe(prompt="CRTXVW render this segmentation map as an anatomical illustration, "
                  "keep every region's shape and color",
           image=guide, num_inference_steps=4,
           guidance_scale=1.0,        # distilled default; HF's edit-LoRA example used 4.0, so try both
           height=768, width=768,     # must be divisible by 16
           generator=torch.Generator("cuda").manual_seed(0)).images[0]
img.save("out.png")
```

Score each system on the **20 held-out slices × 3 seeds**:

| System | mean IoU | lesion IoU | phantom lesion % | gate pass % | latency (s) | cost/img |
|---|---|---|---|---|---|---|
| Workers klein-4b (baseline) | | | | | | |
| Workers klein-9b (teacher) | | | | | | |
| LoRA @ 1000 / 1250 / 1500 / 1750 | | | | | | $0 |

Also produce `eval_grid.png`, with rows = slices and columns = guide | MRI | baseline | teacher | best LoRA.

**Decision rule:** the LoRA goes into the live demo only if its gate pass rate is ≥ the baseline **and** its mean IoU is ≥ the baseline + 0.03, or it is clearly better visually. Otherwise the demo runs on the baseline and the LoRA appears in the slides as "our distilled 4B student, at X% of teacher quality, free and local". Either outcome works for the pitch.

---

## 9. Phase 5: Serve the LoRA via Cloudflare Tunnel (≈1–2 h) *Should* (D4)

Workers AI can't host the LoRA, so your PC serves it and **Cloudflare Tunnel** gives it a public HTTPS URL. No port forwarding is needed.

1. Write `serve.py` (FastAPI):
   - Load the pipeline once at startup.
   - `POST /render`: multipart with fields `guide` (PNG), optional `mri` (PNG) and `seed` (int). Returns a PNG.
   - `GET /health`
   - Require the header `X-CortexView-Key: <shared secret>`, and reject requests without it.
2. Run it with `uvicorn serve:app --host 127.0.0.1 --port 7860`.
3. Install cloudflared: `winget install --id Cloudflare.cloudflared`
4. Start a quick tunnel: `cloudflared tunnel --url http://localhost:7860`. It prints `https://<random>.trycloudflare.com`.
5. Send that URL and the secret to the main team. They set `RENDER_BACKEND=local`, `LOCAL_RENDER_URL=...` and `LOCAL_RENDER_KEY=...`.

Caveats:
- A quick-tunnel URL **changes every time you restart it**, so re-send it after a restart.
- Your PC must stay on and awake (disable sleep) during judging.
- The main app automatically falls back to Workers AI if your server times out (more than 30 s) or errors.

---

## 10. Interface contract with the main app

Keep this exact signature. Both backends implement it.

```python
@dataclass
class RenderResult:
    image_png: bytes          # full illustration, same aspect as guide
    backend: str              # "workers-klein-4b" | "workers-klein-9b" | "local-lora-v1" | "flat-fallback"
    mean_iou: float
    per_layer_iou: dict[str, float]
    phantom_lesion_pct: float
    attempts: int
    latency_ms: int

def render_slice(guide_png: bytes, masks: dict[str, bytes],
                 mri_png: bytes | None = None, seed: int = 0) -> RenderResult: ...
```

The main app owns cutting out the layers, all clinical metrics, and the UI. **You never compute or display clinical numbers.**

---

## 11. Ground rules

- **Only send PNG renders to Workers AI.** Never send DICOM files, filenames containing patient IDs, or any metadata.
- **Check the dataset's terms** allow processing through a third-party API before Phase 2 sends slices to Cloudflare. If they don't, generate targets locally with your LoRA's base model instead.
- **Never commit** `.env`, `*.safetensors`, `datasets/`, `output/` or `slices/`. Add them to `.gitignore`.
- Keep all experiments reproducible: log the seed, prompt id and model for every run.

---

## 12. Suggested timeline

Shift the blocks to fit the actual deadline. The order matters more than the hours.

| Block | Work | Exit criterion |
|---|---|---|
| 1 | Phase 0 setup | `smoke_test.png` generated; torch shows `sm_120` |
| 2 | Phase 1 baseline + fidelity | D1 + D2 done, **handed to the main team** |
| 3 | Phase 2 dataset | ≥ 50 reviewed pairs |
| 4 | Phase 3 training (can run overnight) | Checkpoints 250…2000 saved |
| 5 | Phase 4 eval | D5 table + grid |
| 6 | Phase 5 serving + D6 demo cache | Tunnel URL works from another network; 3–5 demo slices cached |

---

## 13. Risks and fallbacks

| Risk | Fallback |
|---|---|
| 8 GB OOM | Rent a 24 GB GPU for about 1 h (§7 Plan B) |
| Baseline ignores the map (low IoU) | Try E2/E3, raise guidance, or feed per-layer masks as extra refs; worst case use the flat-color render |
| Phantom lesions | Hard gate in `fidelity.py`, plus clip-to-mask in the main app |
| Workers AI rate limits or outage during the demo | D6 pre-rendered cache |
| Local server or tunnel down during the demo | Automatic fallback to Workers AI |
| Free neurons used up | Enable Workers Paid ($5/mo) or switch the remaining runs to klein-4b |

---

## 14. References

- Workers AI REST quickstart: https://developers.cloudflare.com/workers-ai/get-started/rest-api/
- FLUX.2 klein 4B on Workers AI (params, multipart, limits): https://developers.cloudflare.com/changelog/2026-01-15-flux-2-klein-4b-workers-ai
- FLUX.2 klein 9B on Workers AI: https://developers.cloudflare.com/changelog/post/2026-01-28-flux-2-klein-9b-workers-ai/
- Workers AI pricing: https://developers.cloudflare.com/workers-ai/platform/pricing/
- Workers AI LoRA docs (text models only, which is why we self-host): https://developers.cloudflare.com/workers-ai/features/fine-tunes/loras/
- BFL klein training guide: https://docs.bfl.ml/flux_2/flux2_klein_training
- HF: Fine-tune FLUX.2 klein with a LoRA (style + edit LoRA, inference code): https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora
- ai-toolkit: https://github.com/ostris/ai-toolkit
- Cloudflare Tunnel (quick tunnels): https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/
