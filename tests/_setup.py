"""Shared test setup: puts backend/ on the import path and caches phantoms (they take a few seconds)."""
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from cortexview.phantom import make_phantom  # noqa: E402
from cortexview.segment import analyze  # noqa: E402


@lru_cache(maxsize=None)
def phantom(sequence="T1_post", lesion=True):
    return make_phantom(sequence=sequence, lesion=lesion)


@lru_cache(maxsize=None)
def phantom_seg(sequence="T1_post", lesion=True):
    img, labels, info = phantom(sequence, lesion)
    return analyze(img, info["spacing"], sequence)


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s
