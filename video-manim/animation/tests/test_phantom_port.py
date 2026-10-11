"""Checks that animation/phantom.py reproduces frontend/src/cortexview/demo/phantom.ts.

Runs the original TypeScript with Node (>= 23.6 strips types natively), with its two imports replaced by
inline definitions from shared/layers.json and a stub canvas for the image-painting code (pixels are not
compared, only the label map, the greyscale image and the metrics). Skips when Node is not on PATH.

    python -m pytest video-manim/animation/tests -q        (or run this file directly)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phantom as P  # noqa: E402

TS = P.REPO / "frontend" / "src" / "cortexview" / "demo" / "phantom.ts"

DRIVER = r"""
import { writeFileSync } from 'node:fs'
// stub canvas: buildPhantomSlice() paints data URLs we don't need here
globalThis.document = { createElement: () => ({ getContext: () => ({
  createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData: () => {} }),
  toDataURL: () => '' }) }
const { LBL, MRI } = phantom(440, 0, mkRnd(7))
writeFileSync(process.argv[2] + '/lbl.bin', Buffer.from(LBL.buffer))
writeFileSync(process.argv[2] + '/mri.bin', Buffer.from(MRI.buffer))
const { M, intracranialPx, SP } = buildPhantomSlice()
writeFileSync(process.argv[2] + '/metrics.json', JSON.stringify({ M, intracranialPx, SP }))
"""


def run_ts(out: Path) -> None:
    src = TS.read_text(encoding="utf-8")
    src = "\n".join(line for line in src.splitlines() if not line.startswith("import "))
    layers = sorted(json.loads(P.LAYERS_JSON.read_text(encoding="utf-8"))["layers"], key=lambda l: l["peel_order"])
    head = (f"const LAYERS = {json.dumps(layers)};\n"
            "const byLabel = Object.fromEntries(LAYERS.map((l) => [l.label, l]));\n"
            "type LayerId = string;\n")
    script = out / "phantom_check.ts"
    script.write_text(head + src + DRIVER, encoding="utf-8")
    subprocess.run(["node", str(script), str(out)], check=True, capture_output=True, text=True)


def test_matches_typescript():
    if shutil.which("node") is None:
        import pytest
        pytest.skip("node not on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        run_ts(out)
        lbl_ts = np.frombuffer((out / "lbl.bin").read_bytes(), np.uint8).reshape(440, 440)
        mri_ts = np.frombuffer((out / "mri.bin").read_bytes(), np.float32).reshape(440, 440)
        ref = json.loads((out / "metrics.json").read_text())

    s = P.main_slice()
    assert ref["SP"] == P.SP_MAIN
    n_diff = int((s["lbl"] != lbl_ts).sum())
    assert n_diff == 0, f"{n_diff} label pixels differ"
    assert float(np.abs(s["mri"] - mri_ts).max()) < 1e-6
    assert ref["intracranialPx"] == s["metrics"]["_intracranial_px"]
    for layer_id, m in ref["M"].items():
        py = s["metrics"][layer_id]
        assert m["px"] == py.px, layer_id
        assert abs(m["mm2"] - py.mm2) < 1e-9, layer_id
        assert m["comps"] == py.comps, layer_id
        assert tuple(m["anchorPx"]) == py.anchor_px, layer_id
        assert (m["diam"] is None) == (py.diam is None), layer_id
        if m["diam"] is not None:
            assert abs(m["diam"] - py.diam) < 1e-9, layer_id
        if m["pct"] is not None:
            assert abs(m["pct"] - py.pct) < 1e-9, layer_id
    print("phantom port matches phantom.ts:", {k: v["px"] for k, v in ref["M"].items()})


if __name__ == "__main__":
    test_matches_typescript()
