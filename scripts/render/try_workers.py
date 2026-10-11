"""Send one slice to Workers AI and save what the model drew, with its fidelity score.

Uses the same request code and backend/.env settings as the API (CF_MODEL, RENDER_STEPS, ...), so it's a
quick way to check a Workers AI key or compare models without the frontend. Unlike the API, it shows the
raw AI image even when it fails the fidelity gate (the API would show flat colours instead).

    backend/.venv/Scripts/python.exe scripts/render/try_workers.py                 # built-in phantom slice
    backend/.venv/Scripts/python.exe scripts/render/try_workers.py data/out/slices/s0001
    backend/.venv/Scripts/python.exe scripts/render/try_workers.py --model workers-flux2-dev --steps 28

Writes data/out/render_preview_<model>.png: guide | illustration | MRI. Only the guide PNG is sent out.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("slice_dir", nargs="?", help="a C2 slice folder (masks/*.png + mri.png); default: phantom")
ap.add_argument("--model", help="workers-klein-4b | workers-klein-9b | workers-flux2-dev (default: CF_MODEL)")
ap.add_argument("--steps", type=int, help="flux2-dev only (default: RENDER_STEPS)")
args = ap.parse_args()

env_file = ROOT / "backend" / ".env"
if env_file.is_file():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
if args.model:
    os.environ["CF_MODEL"] = args.model
if args.steps:
    os.environ["RENDER_STEPS"] = str(args.steps)
os.environ["RENDER_BACKENDS"] = "workers,flat"

import renderer  # noqa: E402
from renderer import fidelity  # noqa: E402
from cortexview.export import guide_rgb, label_map, window_8bit  # noqa: E402
from cortexview.layers import load_layers  # noqa: E402

cfg = renderer.Config.from_env()
if "workers" not in cfg.available():
    sys.exit("Workers AI isn't configured: set CF_ACCOUNT_ID and CF_API_TOKEN in backend/.env")

ids = [l.id for l in load_layers()]
if args.slice_dir:
    d = Path(args.slice_dir)
    masks = {i: np.asarray(Image.open(d / "masks" / f"{i}.png")) > 0 for i in ids}
    mri8 = np.asarray(Image.open(d / "mri.png").convert("L"))
else:
    from cortexview.phantom import make_phantom
    from cortexview.segment import analyze
    img, _, info = make_phantom()
    seg = analyze(img, info["spacing"], "T1_post")
    k = seg.slice_index
    masks = {i: seg.masks[i][k] for i in ids}
    mri8 = window_8bit(img[k], seg.head[k])

guide = guide_rgb(label_map(masks))
print(f"model {cfg.cf_backend}" + (f", {cfg.steps} steps" if cfg.cf_backend in renderer.CF_STEPPED else "")
      + f", slice {guide.shape[1]}x{guide.shape[0]}")
prompt = renderer.build_prompt([i for i in ids if masks[i].any()])
t0 = time.time()
try:
    raw = renderer._call_workers(cfg, guide, prompt, seed=0, timeout=180)
except renderer._Skip:
    sys.exit("Workers AI refused the request (400/401/403/404): check CF_ACCOUNT_ID, CF_API_TOKEN and CF_MODEL")
img = renderer._decode_to(raw, (guide.shape[1], guide.shape[0]))
s = fidelity.score(img, masks)
print(f"{time.time() - t0:.1f} s | mean IoU {s.mean_iou} | phantom lesion {s.phantom_lesion_pct}% "
      f"| passes the gate: {s.passed}")
print("per layer IoU:", s.per_layer_iou)

out = ROOT / "data" / "out" / f"render_preview_{cfg.cf_backend}.png"
out.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.concatenate([guide, img, np.repeat(mri8[..., None], 3, -1)], 1)).save(out)
print("saved", out.relative_to(ROOT))
