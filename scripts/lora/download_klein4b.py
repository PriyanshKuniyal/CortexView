"""
Download FLUX.2 [klein] base-4B (Apache-2.0, not gated) into models/flux2-klein-base-4b for local training.

One copy of each component, laid out so both tools read the same folder:
  flux-2-klein-base-4b.safetensors   transformer, BFL single-file format (ai-toolkit; diffusers via from_single_file)
  ae.safetensors                     VAE in BFL format (ai-toolkit looks for this name next to the transformer)
  text_encoder/ tokenizer/ vae/ scheduler/ model_index.json   diffusers layout (make_candidates.py)
The diffusers transformer/ weights are skipped (only its config.json is fetched): they are the same 7.75 GB as
the single file.

    .venv-lora/Scripts/python scripts/lora/download_klein4b.py
Rerunning resumes/skips finished files.
"""
import pathlib
import shutil

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEST = ROOT / "models" / "flux2-klein-base-4b"

snapshot_download(
    "black-forest-labs/FLUX.2-klein-base-4B",
    local_dir=DEST,
    allow_patterns=[
        "flux-2-klein-base-4b.safetensors",
        "model_index.json",
        "LICENSE.md",
        "README.md",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "transformer/config.json",          # config only: diffusers builds the transformer from the single file
        "vae/*",
    ],
)
hf_hub_download("ai-toolkit/flux2_vae", "ae.safetensors", local_dir=DEST)

# ai-toolkit loads the Qwen3 text encoder and its tokenizer from one directory, so give text_encoder/ the
# tokenizer files too (a few MB) instead of downloading Qwen/Qwen3-4B (8 GB) a second time.
for f in (DEST / "tokenizer").iterdir():
    if f.is_file():
        shutil.copy2(f, DEST / "text_encoder" / f.name)

print("done:", DEST)
