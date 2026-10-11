"""
Step 2: generate candidate illustrations from guide.png files with FLUX.2 klein base (no LoRA).

Run on the cloud GPU (L40S / A100), not on an 8 GB laptop.
    python make_candidates.py --model ./flux2-klein-base-9b --guides guides --out candidates

Resumable: existing outputs are skipped, so you can rerun after a spot interruption.

Training candidates (train patients only, base model, no LoRA):
    python make_candidates.py --guides guides/guide --splits guides/splits.csv --split train
Evaluating a trained LoRA on held-out patients (distilled model, 4 steps):
    python make_candidates.py --model ./flux2-klein-9b --lora lora.safetensors --guides guides/guide \
        --splits guides/splits.csv --split test --samples 1 --steps 4 --guidance 4.0 \
        --prompt "turn this brain segmentation map into a CRTXV1EW illustration" --out eval_lora
"""
import argparse
import pathlib

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

# One fixed generation prompt keeps the style consistent across all candidates.
GEN_PROMPT = (
    "Redraw this color-coded brain MRI segmentation map as a clean medical textbook "
    "illustration. Keep every colored region exactly the same shape, position and color "
    "family. Add soft shading, subtle tissue texture and thin dark outlines. "
    "Plain background, no text, no labels, no arrows."
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="./flux2-klein-base-9b")
ap.add_argument("--guides", default="guides")
ap.add_argument("--out", default="candidates")
ap.add_argument("--samples", type=int, default=3, help="samples per guide")
ap.add_argument("--size", type=int, default=1024, help="must be divisible by 16")
ap.add_argument("--steps", type=int, default=50, help="base model needs ~50 steps")
ap.add_argument("--guidance", type=float, default=4.0)
ap.add_argument("--prompt", default=GEN_PROMPT)
ap.add_argument("--lora", default="", help="path to a LoRA .safetensors (use for evaluating a trained LoRA)")
ap.add_argument("--splits", default="", help="splits.csv from remind_synthseg.py guides")
ap.add_argument("--split", default="", choices=["", "train", "test"], help="only use guides from this split")
args = ap.parse_args()

assert args.size % 16 == 0, "size must be divisible by 16"
guides = sorted(pathlib.Path(args.guides).glob("*.png"))
if args.split:
    import csv
    keep = {r["PatientID"] for r in csv.DictReader(open(args.splits)) if r["split"] == args.split}
    guides = [g for g in guides if g.stem.rsplit("_s", 1)[0] in keep]
assert guides, f"no .png guides found in {args.guides}"
out_dir = pathlib.Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
if args.lora:
    pipe.load_lora_weights(args.lora)   # needs `pip install peft`

total = len(guides) * args.samples
done = 0
for gp in guides:
    # NEAREST keeps the label colors crisp when upscaling a 256/512 px guide.
    guide = Image.open(gp).convert("RGB").resize((args.size, args.size), Image.NEAREST)
    for k in range(args.samples):
        done += 1
        out = out_dir / f"{gp.stem}__s{k}.png"
        if out.exists():
            continue
        img = pipe(
            prompt=args.prompt,
            image=guide,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator("cuda").manual_seed(1000 + k),
        ).images[0]
        img.save(out)
        print(f"[{done}/{total}] {out.name}", flush=True)

print("done")
