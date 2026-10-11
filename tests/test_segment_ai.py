"""AI-first segmenter: fallback contracts and primary pipeline selection."""
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


class TestAiFallback(unittest.TestCase):
    def test_non_t1_uses_classical_fallback(self):
        image, _, info = phantom("FLAIR")
        result = segment_ai.analyze(image, info["spacing"], "FLAIR", tissue_slices=0)
        self.assertIn("segmenter_fallback", result.warnings)
        self.assertEqual(set(result.masks), set(layer_ids()))

    def test_missing_weights_uses_classical_fallback(self):
        image, _, info = phantom("T1_post")
        with mock.patch.object(segment_ai, "WEIGHTS", Path(tempfile.gettempdir()) / "missing_cortexview_weights"):
            self.assertFalse(segment_ai.available()[0])
            result = segment_ai.analyze(image, info["spacing"], "T1_post", tissue_slices=0)
        self.assertIn("segmenter_fallback", result.warnings)

    def test_single_slice_uses_classical_fallback(self):
        image, _, info = phantom("T1_post")
        result = segment_ai.analyze(image[image.shape[0] // 2], info["spacing"][1:], "T1_post")
        self.assertIn("segmenter_fallback", result.warnings)

    def test_pipeline_defaults_to_ai_segmenter(self):
        image, _, info = phantom("T1_post")
        with tempfile.TemporaryDirectory() as tmp:
            write_series(image, info["spacing"], Path(tmp), "T1_post", name="ai-default")
            with mock.patch.dict(os.environ, {}, clear=True):
                result = run(sorted(Path(tmp).rglob("*.dcm")))
        self.assertEqual(set(result.slice.masks), set(layer_ids()))
        if not segment_ai.available()[0]:
            self.assertIn("segmenter_fallback", result.warnings)

    def test_classical_can_be_requested_explicitly(self):
        image, _, info = phantom("T1_post")
        with tempfile.TemporaryDirectory() as tmp:
            write_series(image, info["spacing"], Path(tmp), "T1_post", name="classical-override")
            with mock.patch.dict(os.environ, {"CORTEXVIEW_SEGMENTER": "classical"}):
                result = run(sorted(Path(tmp).rglob("*.dcm")))
        self.assertNotIn("segmenter_fallback", result.warnings)


@unittest.skipUnless(segment_ai.available()[0], "AI segmenter is not usable in this environment")
class TestModel(unittest.TestCase):
    def test_deterministic_exclusive_and_complete(self):
        image, _, info = phantom("T1_post")
        first = segment_ai.analyze(image, info["spacing"], "T1_post")
        second = segment_ai.analyze(image, info["spacing"], "T1_post")
        self.assertNotIn("segmenter_fallback", first.warnings)
        for layer in layer_ids():
            self.assertTrue(np.array_equal(first.masks[layer], second.masks[layer]), layer)
        total = sum(first.masks[layer].astype(int) for layer in layer_ids())
        self.assertLessEqual(int(total.max()), 1)
        self.assertTrue(first.masks["white_matter"].any() and first.masks["gray_matter"].any())
        self.assertEqual(set(first.confidence.values()) - {"high", "medium", "low", "empty"}, set())


if __name__ == "__main__":
    unittest.main()
