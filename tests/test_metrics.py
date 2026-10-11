"""Contract C4 metrics: hand-computed geometry, hemisphere, text, and the segmenter/pipeline hand-off."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _setup import phantom
from cortexview.layers import label_of, layer_ids
from cortexview.metrics import compute, report, volumes_ml
from cortexview.phantom import write_seg, write_series
from cortexview.pipeline import run
from cortexview.segment import analyze


def masks_of(labels):
    return {lid: labels == label_of(lid) for lid in layer_ids()}


def metrics(labels, image=None, spacing=(1.0, 1.0), patient_left="image_left", conf="medium",
            source="auto", **kw):
    image = np.full(labels.shape, 0.5) if image is None else image
    ids = layer_ids()
    return compute(masks_of(labels), image, spacing, patient_left,
                   {i: conf for i in ids}, {i: source for i in ids}, **kw)


def ring_slice(n=128):
    """Concentric skull / GM / WM / CSF, plus a lesion right of centre. Image is left/right symmetric
    apart from the lesion."""
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n / 2, xx - n / 2)
    sl = np.zeros((n, n), np.uint8)
    for radius, lid in ((55, "skull"), (50, "gray_matter"), (40, "white_matter"), (10, "csf")):
        sl[r < radius] = label_of(lid)
    sl[np.hypot(yy - 64, xx - 85) < 6] = label_of("lesion")
    brightness = {"skull": 0.3, "gray_matter": 0.5, "white_matter": 0.7, "csf": 0.1, "lesion": 0.9}
    image = np.zeros((n, n))
    for lid, v in brightness.items():
        image[sl == label_of(lid)] = v
    image += 0.05 * np.cos(yy / 7.0)            # vertical texture, symmetric left/right
    return sl, image


class TestGeometry(unittest.TestCase):
    def test_square(self):
        lab = np.zeros((100, 100), np.uint8)
        lab[20:30, 30:40] = label_of("lesion")
        les = metrics(lab).layers["lesion"]
        self.assertTrue(les.present)
        self.assertEqual(les.area_px, 100)
        self.assertEqual(les.area_mm2, 100.0)
        self.assertEqual(les.n_components, 1)
        self.assertEqual(les.bbox_px, (30, 20, 39, 29))         # [x0, y0, x1, y1]
        self.assertEqual(les.centroid_px, (34.5, 24.5))         # [x, y]
        self.assertIn(les.anchor_px, [(34, 24), (34, 25), (35, 24), (35, 25)])

    def test_disk_diameter(self):
        yy, xx = np.mgrid[:100, :100]
        lab = np.zeros((100, 100), np.uint8)
        lab[np.hypot(yy - 50, xx - 50) <= 10] = label_of("lesion")
        self.assertAlmostEqual(metrics(lab).layers["lesion"].max_diameter_mm, 20.0, delta=1.0)

    def test_anisotropic_pixels(self):
        lab = np.zeros((100, 100), np.uint8)
        lab[20:30, 30:40] = label_of("lesion")
        les = metrics(lab, spacing=(1.0, 0.5)).layers["lesion"]
        self.assertEqual(les.area_mm2, 50.0)
        self.assertAlmostEqual(les.max_diameter_mm, np.hypot(9 * 1.0, 9 * 0.5), delta=0.1)

    def test_two_blobs_anchor_in_larger(self):
        lab = np.zeros((100, 100), np.uint8)
        lab[20:30, 20:30] = label_of("lesion")
        lab[70:73, 70:73] = label_of("lesion")
        les = metrics(lab).layers["lesion"]
        self.assertEqual(les.n_components, 2)
        x, y = les.anchor_px
        self.assertTrue(20 <= x <= 29 and 20 <= y <= 29)

    def test_diameter_only_for_lesion(self):
        sl, image = ring_slice()
        rep = metrics(sl, image)
        self.assertIsNotNone(rep.layers["lesion"].max_diameter_mm)
        self.assertTrue(all(rep.layers[i].max_diameter_mm is None for i in layer_ids() if i != "lesion"))

    def test_percent_and_brain_ratio(self):
        sl, image = ring_slice()
        rep = metrics(sl, image, spacing=(0.5, 0.5))
        ic_px = int(np.isin(sl, [label_of(i) for i in layer_ids() if i != "skull"]).sum())
        self.assertEqual(rep.intracranial_area_mm2, round(ic_px * 0.25, 1))
        les = rep.layers["lesion"]
        self.assertEqual(les.pct_intracranial, round(100 * les.area_px / ic_px, 1))
        self.assertIsNone(rep.layers["skull"].pct_intracranial)
        self.assertIsNone(rep.layers["skull"].intensity.ratio_to_brain)
        self.assertGreater(les.intensity.ratio_to_brain, 1.0)        # brighter than GM/WM
        self.assertLess(rep.layers["csf"].intensity.ratio_to_brain, 1.0)


class TestHemisphere(unittest.TestCase):
    def test_fitted_midline(self):
        sl, image = ring_slice()
        les = metrics(sl, image, patient_left="image_right").layers["lesion"]
        self.assertEqual(les.hemisphere, "left")                     # image right = patient left here
        self.assertNotIn("midline_estimated", les.warnings)
        self.assertEqual(metrics(sl, image).layers["white_matter"].hemisphere, "midline")

    def test_flat_image_falls_back_and_toggles(self):
        lab = np.zeros((100, 100), np.uint8)
        lab[10:90, 10:90] = label_of("gray_matter")
        lab[40:60, 15:25] = label_of("lesion")                       # image left
        left = metrics(lab, patient_left="image_left").layers["lesion"]
        right = metrics(lab, patient_left="image_right").layers["lesion"]
        self.assertEqual((left.hemisphere, right.hemisphere), ("left", "right"))
        self.assertIn("midline_estimated", left.warnings)

    def test_unknown_orientation(self):
        sl, image = ring_slice()
        rep = metrics(sl, image, patient_left=None)
        self.assertTrue(all(m.hemisphere is None for m in rep.layers.values()))


class TestEdgeCases(unittest.TestCase):
    def test_unknown_spacing_and_empty_layers(self):
        lab = np.zeros((100, 100), np.uint8)
        lab[20:30, 20:30] = label_of("lesion")
        rep = metrics(lab, spacing=None, patient_left=None)
        les = rep.layers["lesion"]
        self.assertIsNone(rep.intracranial_area_mm2)
        self.assertIsNone(les.area_mm2)
        self.assertIsNone(les.max_diameter_mm)
        gm = rep.layers["gray_matter"]
        self.assertFalse(gm.present)
        self.assertEqual((gm.confidence, gm.area_px, gm.intensity, gm.anchor_px), ("empty", 0, None, None))

    def test_small_region_confidence(self):
        lab = np.zeros((100, 100), np.uint8)
        lab[20:25, 20:25] = label_of("lesion")                       # 25 px
        auto = metrics(lab, conf="medium", source="auto").layers["lesion"]
        doctor = metrics(lab, conf="high", source="doctor").layers["lesion"]
        self.assertEqual((auto.confidence, doctor.confidence), ("low", "high"))
        self.assertIn("small_region", doctor.warnings)

    def test_deterministic(self):
        sl, image = ring_slice()
        self.assertEqual(metrics(sl, image).model_dump_json(), metrics(sl, image).model_dump_json())

    def test_contract_shape(self):
        sl, image = ring_slice()
        data = json.loads(metrics(sl, image).model_dump_json())
        self.assertEqual(set(data), {"intracranial_area_mm2", "layers", "text"})
        self.assertEqual(list(data["layers"]), layer_ids())          # every layer, in peel order


class TestText(unittest.TestCase):
    def test_lesion_sentences(self):
        sl, image = ring_slice()
        rep = metrics(sl, image, spacing=(0.5, 0.5), patient_left="image_right")
        by_id = {t.template_id: t for t in rep.text}
        les = rep.layers["lesion"]
        self.assertIn(f"about {les.area_mm2 / 100:.1f} cm²", by_id["lesion_area"].text)
        self.assertEqual(by_id["lesion_side"].text, "It is on the left side of the brain.")
        self.assertIn("lesion_diameter", by_id)
        self.assertNotIn("layer_volume", by_id)                      # no volumes given

    def test_no_lesion_no_lesion_text(self):
        sl, image = ring_slice()
        sl[sl == label_of("lesion")] = label_of("white_matter")
        self.assertEqual(metrics(sl, image).text, [])


class TestWithSegmenter(unittest.TestCase):
    """Whole-volume sizes must not come from a partial (API-mode) tissue split."""

    @classmethod
    def setUpClass(cls):
        img, _, info = phantom("FLAIR")
        cls.spacing = info["spacing"]
        cls.full = analyze(img, cls.spacing, "FLAIR")
        cls.fast = analyze(img, cls.spacing, "FLAIR", tissue_slices="auto")

    def test_volumes(self):
        full, fast = volumes_ml(self.full, self.spacing), volumes_ml(self.fast, self.spacing)
        for lid in ("csf", "gray_matter", "white_matter"):
            self.assertIsNone(fast[lid])
            self.assertGreater(full[lid], 10)
        for lid in ("skull", "lesion"):
            self.assertEqual(fast[lid], full[lid])
            self.assertGreater(full[lid], 0)

    def test_slice_metrics_match_full_run(self):
        def rep(seg):
            k = self.fast.slice_index
            sl = seg.cut(k)
            return compute(sl.masks, phantom("FLAIR")[0][k], self.spacing[1:], "image_right",
                           sl.confidence, sl.source, head=sl.head, intracranial=sl.intracranial)
        self.assertEqual(rep(self.full).model_dump_json(), rep(self.fast).model_dump_json())


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        t1, _, cls.info = phantom("T1_post")
        write_series(t1, cls.info["spacing"], root / "t1", "T1_post", name="p")
        write_seg(cls.info["lesion"], cls.info["spacing"], root / "seg" / "tumor.dcm", "Tumor", name="p")
        cls.res = run(sorted(root.rglob("*.dcm")))
        cls.rep = report(cls.res)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_doctor_lesion(self):
        les = self.rep.layers["lesion"]
        self.assertTrue(les.present)
        self.assertEqual(les.confidence, "high")
        true_ml = self.info["lesion"].sum() * float(np.prod(self.info["spacing"])) / 1000
        self.assertAlmostEqual(les.volume_ml, true_ml, delta=0.1)
        self.assertIsNotNone(les.hemisphere)

    def test_api_mode_volumes_and_text(self):
        self.assertIsNone(self.rep.layers["gray_matter"].volume_ml)   # tissue split on one slice only
        self.assertIsNotNone(self.rep.layers["skull"].volume_ml)
        volume_layers = [t.layer for t in self.rep.text if t.template_id == "layer_volume"]
        self.assertEqual(volume_layers, ["skull", "lesion"])


if __name__ == "__main__":
    unittest.main()
