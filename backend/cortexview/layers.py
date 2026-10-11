"""Layer registry (contract C1), read from shared/layers.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ROOT = BACKEND_ROOT.parent
LAYERS_JSON = (BACKEND_ROOT / "shared" / "layers.json" if (BACKEND_ROOT / "shared").exists()
               else ROOT / "shared" / "layers.json")


@dataclass(frozen=True)
class Layer:
    id: str
    label: int
    name: str
    rgb: tuple[int, int, int]
    peel_order: int


@lru_cache(maxsize=1)
def load_layers() -> tuple[Layer, ...]:
    """All layers, in peel order (outermost first)."""
    data = json.loads(LAYERS_JSON.read_text(encoding="utf-8"))
    layers = [
        Layer(l["id"], int(l["label"]), l["name"], tuple(l["rgb"]), int(l["peel_order"]))
        for l in data["layers"]
    ]
    return tuple(sorted(layers, key=lambda l: l.peel_order))


def layer_ids() -> list[str]:
    return [l.id for l in load_layers()]


def label_of(layer_id: str) -> int:
    return next(l.label for l in load_layers() if l.id == layer_id)


# Exclusivity priority (C1/TASKS §3): a voxel claimed by an earlier layer wins.
PRIORITY = ("lesion", "csf", "white_matter", "gray_matter", "skull")
