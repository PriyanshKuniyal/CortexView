"""Serve the local CortexView Klein LoRA through the backend renderer contract."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import torch
from diffusers import BitsAndBytesConfig, Flux2KleinPipeline, Flux2Transformer2DModel
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("PLIE_MODELS") or "C:/models") / "flux2-klein-base-4b"
LORA = Path(os.environ.get("PLIE_LORA_PATH") or (ROOT / "lora_work" / "output" / "cortexview_klein4b_edit_v1" / "cortexview_klein4b_edit_v1.safetensors"))
DEVICE = "cuda"
SIZE = int(os.environ.get("LOCAL_RENDER_SIZE", "512"))
KEY = os.environ.get("LOCAL_RENDER_KEY", "")

app = FastAPI(title="CortexView local LoRA renderer")
pipe: Flux2KleinPipeline | None = None


def load_pipeline() -> Flux2KleinPipeline:
    quantization = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    }
    transformer = Flux2Transformer2DModel.from_pretrained(
        BASE / "transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(**quantization),
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        BASE / "text_encoder",
        torch_dtype=torch.bfloat16,
        quantization_config=TransformersBitsAndBytesConfig(**quantization),
    )
    loaded = Flux2KleinPipeline.from_pretrained(
        BASE,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    loaded.enable_model_cpu_offload()
    loaded.load_lora_weights(str(LORA), adapter_name="cortexview")
    if "cortexview" not in loaded.get_active_adapters():
        raise RuntimeError("CortexView LoRA did not become active")
    return loaded


@app.on_event("startup")
def startup() -> None:
    global pipe
    pipe = load_pipeline()
    print("CortexView local renderer ready with active LoRA: cortexview", flush=True)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": str(BASE), "lora": "cortexview", "loaded": pipe is not None}


@app.post("/render")
async def render(guide: UploadFile = File(...), seed: int = 0, x_cortexview_key: str | None = Header(default=None)) -> Response:
    if KEY and x_cortexview_key != KEY:
        raise HTTPException(status_code=403, detail="invalid renderer key")
    if pipe is None:
        raise HTTPException(status_code=503, detail="model is still loading")
    control = Image.open(io.BytesIO(await guide.read())).convert("RGB")
    control = control.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    prompt = (
        "CRTXV1EW, redraw this brain segmentation map as a clean educational medical illustration. "
        "Preserve every region's exact shape and position. Use tan beige for skull and scalp, "
        "pink-purple for gray matter, cream white for white matter, blue for CSF and ventricles, "
        "and bright red for lesion only when present. Add soft shading and subtle tissue texture. "
        "Plain black background, no labels, no text."
    )
    image = pipe(
        prompt=prompt,
        image=control,
        width=SIZE,
        height=SIZE,
        num_inference_steps=20,
        guidance_scale=4.0,
        generator=torch.Generator(DEVICE).manual_seed(seed),
    ).images[0]
    output = io.BytesIO()
    image.save(output, format="PNG")
    return Response(output.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)