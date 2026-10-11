"""
Run an ai-toolkit job against the local FLUX.2 [klein] base-4B in <models>/flux2-klein-base-4b
(<models> as in download_klein4b.py: $PLIE_MODELS, else C:/models on Windows, else <repo>/models).

    .venv-lora/Scripts/python scripts/lora/run_aitk.py scripts/lora/smoke_klein4b_edit.yaml

What it does on top of `python run.py <config>` (ai-toolkit in third_party/ stays unmodified):
  - sets PLIE=<repo root>, and PLIE_MODELS=<models>, so configs can write paths as ${PLIE}/... and
    ${PLIE_MODELS}/... (ai-toolkit expands ${VAR} in configs);
  - points flux2_klein_4b's text encoder at <models>/flux2-klein-base-4b/text_encoder. ai-toolkit hardcodes
    "Qwen/Qwen3-4B" from the Hub there, which would download the same 8 GB a second time into the HF cache.
"""
import os
import pathlib
import runpy
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
AITK = ROOT / "third_party" / "ai-toolkit"
MODELS = pathlib.Path(os.environ.get("PLIE_MODELS") or ("C:/models" if os.name == "nt" else ROOT / "models"))
TE_DIR = MODELS / "flux2-klein-base-4b" / "text_encoder"

if len(sys.argv) < 2:
    sys.exit(__doc__)
config = pathlib.Path(sys.argv[1]).resolve()
assert TE_DIR.is_dir(), f"missing {TE_DIR}; run scripts/lora/download_klein4b.py first"

os.environ["PLIE"] = ROOT.as_posix()
os.environ["PLIE_MODELS"] = MODELS.as_posix()
os.chdir(AITK)
sys.path.insert(0, str(AITK))

from extensions_built_in.diffusion_models.flux2 import flux2_klein_model  # noqa: E402

flux2_klein_model.Flux2Klein4BModel.flux2_klein_te_path = str(TE_DIR)

sys.argv = ["run.py", str(config), *sys.argv[2:]]
runpy.run_path(str(AITK / "run.py"), run_name="__main__")
