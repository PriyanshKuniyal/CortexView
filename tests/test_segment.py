"""Classical segmenter on the synthetic phantom (known answer)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _setup import dice, phantom, phantom_seg
from cortexview import dicom_io as io
from cortexview.layers import label_of, layer_ids
from cortexview.phantom import write_seg, write_series
from cortexview.segment import analyze, pick_slice, segment, segment_volume

# Minimum Dice per layer on the phantom. The phantom has clean contrast, so these are well above what
# real clinical scans reach; they guard against regressions, not claims about real-world accuracy.
MIN_DICE = {
    "T1_post": {"skull": 0.85, "gray_matter": 0.55, "white_matter": 0.85, "csf": 0.55},
    "T2": {"skull": 0.85, "gray_matter": 0.55, "white_matter": 0.85, "csf": 0.55},
    "FLAIR": {"skull": 0.85, "gray_matter": 0.55, "white_matter": 0.85, "csf": 0.55, "lesion": 0.7},
}


class TestPhantomAccuracy(unittest.TestCase):
    def test_layers(self):
        for seq, mins in MIN_DICE.items():
            img, labels, info = phantom(seq)
            seg = phantom_seg(seq)
            for lid, lo in mins.items():
                with self.subTest(sequence=seq, layer=lid):
                    self.assertGreaterEqual(dice(seg.masks[lid], labels == label_of(lid)), lo)

    def test_intracranial_includes_lesion(self):
        img, labels, info = phantom("T1_post")
        seg = phantom_seg("T1_post")
        self.assertGreaterEqual((seg.intracranial & info["lesion"]).sum() / info["lesion"].sum(), 0.95)
        truth = np.isin(labels, [label_of(l) for l in ("gray_matter", "white_matter", "csf", "lesion")])
        self.assertGreaterEqual(dice(seg.intracranial, truth), 0.9)

    def test_no_auto_lesion_on_t1(self):
        seg = phantom_seg("T1_post")
        self.assertFalse(seg.masks["lesion"].any())
        self.assertEqual(seg.confidence["lesion"], "empty")
        self.assertIn("lesion_auto_needs_flair", seg.warnings)


class TestContract(unittest.TestCase):
    def test_exclusive_and_complete(self):
        for seq in MIN_DICE:
            seg = phantom_seg(seq)
            stack = np.stack([seg.masks[l] for l in layer_ids()]).sum(0)
            self.assertLessEqual(int(stack.max()), 1, seq)                 # never two labels
            np.testing.assert_array_equal(stack == 1, seg.head)              # every head voxel labelled
            self.assertEqual(list(seg.masks), layer_ids())

    def test_deterministic(self):
        img, labels, info = phantom("FLAIR")
        a = analyze(img, info["spacing"], "FLAIR")
        b = analyze(img.copy(), info["spacing"], "FLAIR")
        for lid in layer_ids():
            np.testing.assert_array_equal(a.masks[lid], b.masks[lid])
        self.assertEqual(a.confidence, b.confidence)

    def test_api_slice_mode_matches_full(self):
        img, labels, info = phantom("T1_post")
        full = phantom_seg("T1_post")
        fast = analyze(img, info["spacing"], "T1_post", tissue_slices="auto")
        k = fast.slice_index
        self.assertEqual(fast.tissue_range, (k, k + 1))
        for lid in layer_ids():
            np.testing.assert_array_equal(fast.masks[lid][k], full.masks[lid][k])
        cut = fast.cut(k)
        self.assertEqual(cut.masks["skull"].shape, img.shape[1:])
        with self.assertRaises(ValueError):
            fast.cut(0 if k > 0 else k + 1)

    def test_auto_slice_is_largest_lesion(self):
        img, labels, info = phantom("FLAIR")
        seg = phantom_seg("FLAIR")
        k = pick_slice(seg)
        area = seg.masks["lesion"].reshape(img.shape[0], -1).sum(1)
        self.assertEqual(k, int(np.argmax(area)))
        true_k = int(np.argmax(info["lesion"].reshape(img.shape[0], -1).sum(1)))
        self.assertLessEqual(abs(k - true_k), 3)

    def test_doctor_labels(self):
        img, labels, info = phantom("T1_post")
        vent = labels == label_of("csf")
        seg = analyze(img, info["spacing"], "T1_post", doctor={"lesion": info["lesion"]})
        np.testing.assert_array_equal(seg.masks["lesion"], info["lesion"])
        self.assertEqual(seg.source["lesion"], "doctor")
        self.assertEqual(seg.confidence["lesion"], "high")

    def test_empty_image(self):
        seg = analyze(np.zeros((20, 32, 32), np.float32), (1.0, 1.0, 1.0), "T1_post")
        self.assertIn("no_head_found", seg.warnings)
        self.assertTrue(all(c == "empty" for c in seg.confidence.values()))


class TestInterfaces(unittest.TestCase):
    def test_2d_fallback(self):
        img, labels, info = phantom("T1_post")
        k = img.shape[0] // 2
        masks = segment(img[k], {"pixel_spacing_mm": list(info["spacing"][1:]), "sequence": "T1_post"})
        self.assertEqual(set(masks), set(layer_ids()))
        ic_truth = np.isin(labels[k], [label_of(l) for l in ("gray_matter", "white_matter", "csf", "lesion")])
        ic = masks["gray_matter"] | masks["white_matter"] | masks["csf"] | masks["lesion"]
        self.assertGreaterEqual(dice(ic, ic_truth), 0.85)

    def test_end_to_end_from_dicom(self):
        img, labels, info = phantom("T1_post")
        with tempfile.TemporaryDirectory() as tmp:
            write_series(img, info["spacing"], Path(tmp) / "mr", "T1_post", name="e2e")
            write_seg(info["lesion"], info["spacing"], Path(tmp) / "seg.dcm", "Tumor", name="e2e")
            series, segs = io.group_series(io.read_folder(tmp))
            vol = io.load_series(next(iter(series.values())))
            doctor, _ = io.doctor_labels_on(vol, [m for s in segs for m in io.load_seg(s)])
            masks = segment_volume(vol, {"doctor": doctor})
            k = int(np.argmax(info["lesion"].reshape(img.shape[0], -1).sum(1)))
            sl = segment(None, {"volume": vol, "slice_index": k, "doctor": doctor})
        np.testing.assert_array_equal(masks["lesion"], info["lesion"])
        for lid in layer_ids():
            np.testing.assert_array_equal(sl[lid], masks[lid][k])


if __name__ == "__main__":
    unittest.main()
