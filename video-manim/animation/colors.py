"""Colors and type for the CortexView explainer.

Tissue colors are NOT defined here: they are read from shared/layers.json (the project's frozen palette),
so the video can never drift from the app.
"""
from manim import ManimColor

from .phantom import load_layers

# theme
BG = "#0E1117"          # near-black, slightly blue
INK = "#E8E6E3"         # primary text
MUTED = "#8B919B"       # secondary text, captions
FAINT = "#2A2F38"       # grid lines, frames
PANEL = "#151A22"       # node / box fill

MATH = "#F4D35E"        # formulas and measured values (warm yellow)
GEN = "#B48EF0"         # generative side (violet)
DET = "#4FD1C5"         # deterministic side (teal)

STATUS = {              # pipeline status dots (scene 5)
    "working": "#7BD88F",
    "progress": "#F4B860",
    "planned": MUTED,
}

# type
SANS = "Segoe UI"
SERIF = "Cambria"       # formulas: italic variables, no LaTeX needed
MONO = "Consolas"

LAYERS = load_layers()                                   # outside in: skull, gm, wm, csf, lesion
LAYER = {L["id"]: L for L in LAYERS}


def layer_color(layer_id: str) -> ManimColor:
    r, g, b = LAYER[layer_id]["rgb"]
    return ManimColor.from_rgb((r, g, b))


# short on-screen names. The lesion in the phantom is simulated, so it is labelled that way.
SHORT_NAME = {
    "skull": "Skull / scalp",
    "gray_matter": "Gray matter",
    "white_matter": "White matter",
    "csf": "CSF / ventricles",
    "lesion": "Lesion (simulated)",
}
