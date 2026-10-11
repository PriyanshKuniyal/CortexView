"""AI segmenter (C6): fallback rules, the pipeline switch, and (with a GPU + trained weights) the model itself.

Without PyTorch/GPU/weights (e.g. the classical venv) only the fallback tests run; the model tests skip.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from _setup import phantom
from cortexview import segment_ai
from cortexview.layers import layer_ids
from cortexview.phantom import write_series
from cortexview.pipeline import run

READY, WHY = segment_ai.available()


class TestFallback(unittest.TestCase):
    def test_non_t1_goes_to_classical(self):
        img, _, info = phantom("FLAIR")
        seg = segment_ai.analyze(img, info["spacing"], "FLAIR", tissue_slices=0)
        self.assertIn("segmenter_fallback", seg.warnings)
        self.assertEqual(set(seg.masks), set(layer_ids()))

    def test_missing_weights_go_to_classical(self):
        img, _, info = phantom("T1_post")
        with mock.patch.object(segment_ai, "WEIGHTS", Path(tempfile.gettempdir()) / "no_such_weights"):
            ok, why = segment_ai.available()
            self.assertFalse(ok)
            seg = segment_ai.analyze(img, info["spacing"], "T1_post", tissue_slices=0)
        self.assertIn("segmenter_fallback", seg.warnings)

    def test_single_slice_goes_to_classical(self):
        img, _, info = phantom("T1_post")
        k = img.shape[0] // 2
        seg = segment_ai.analyze(img[k], info["spacing"][1:], "T1_post")
        self.assertIn("segmenter_fallback", seg.warnings)


class TestPipelineSwitch(unittest.TestCase):
    def test_env_selects_ai_and_result_is_complete(self):
        img, _, info = phantom("T1_post")
        with tempfile.TemporaryDirectory() as tmp:
            write_series(img, info["spacing"], Path(tmp), "T1_post", name="p")
            with mock.patch.dict(os.environ, {"CORTEXVIEW_SEGMENTER": "ai"}):
                res = run(sorted(Path(tmp).rglob("*.dcm")))
        self.assertEqual(set(res.slice.masks), set(layer_ids()))
        if not READY:
            self.assertIn("segmenter_fallback", res.warnings)
        else:
            self.assertNotIn("segmenter_fallback", res.warnings)


@unittest.skipUnless(READY, f"AI segmenter not usable here: {WHY}")
class TestModel(unittest.TestCase):
    def test_deterministic_exclusive_and_complete(self):
        img, _, info = phantom("T1_post")
        a = segment_ai.analyze(img, info["spacing"], "T1_post")
        b = segment_ai.analyze(img, info["spacing"], "T1_post")
        self.assertNotIn("segmenter_fallback", a.warnings)
        for lid in layer_ids():
            self.assertTrue(np.array_equal(a.masks[lid], b.masks[lid]), lid)
        total = sum(a.masks[lid].astype(int) for lid in layer_ids())
        self.assertLessEqual(int(total.max()), 1)                 # one layer per pixel
        self.assertTrue(a.masks["white_matter"].any() and a.masks["gray_matter"].any())
        self.assertEqual(set(a.confidence.values()) - {"high", "medium", "low", "empty"}, set())


if __name__ == "__main__":
    unittest.main()
