"""DICOM loading, geometry and the C2 export, using phantom DICOM files written to a temp folder."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom

from _setup import phantom
from cortexview import dicom_io as io
from cortexview.export import pad_square, write_slice_folder
from cortexview.layers import PRIORITY, label_of, layer_ids, load_layers
from cortexview.phantom import write_seg, write_series
from cortexview.volume import Volume, reorient_to_axial, resample_nearest


class TestLayers(unittest.TestCase):
    def test_registry(self):
        layers = load_layers()
        self.assertEqual([l.id for l in layers], ["skull", "gray_matter", "white_matter", "csf", "lesion"])
        self.assertEqual(sorted(PRIORITY), sorted(layer_ids()))
        self.assertEqual(label_of("lesion"), 10)


class TestSequenceDetection(unittest.TestCase):
    def check(self, desc, expected, **extra):
        ds = pydicom.Dataset()
        ds.SeriesDescription = desc
        for k, v in extra.items():
            setattr(ds, k, v)
        self.assertEqual(io.detect_sequence(ds)[0], expected, desc)

    def test_descriptions(self):
        self.check("3D_AX_T1_postcontrast", "T1_post")
        self.check("3D_AX_T1_precontrast", "T1_pre")
        self.check("3D_SAG_T1_MP2RAGE", "T1_pre")
        self.check("3D_SAG_T1_MP2RAGE", "T1_post", ContrastBolusAgent="Gd")
        self.check("2D_AX_T2_FLAIR", "FLAIR")
        self.check("3D_SAG_T2_SPACE", "T2")
        self.check("AX T1 +C", "T1_post")
        self.check("localizer", "unknown")

    def test_timing_rules(self):
        self.check("series 7", "FLAIR", RepetitionTime=9000, EchoTime=90, InversionTime=2500)
        self.check("series 8", "T2", RepetitionTime=4000, EchoTime=100)
        self.check("series 9", "T1_pre", RepetitionTime=500, EchoTime=12)


class TestGeometry(unittest.TestCase):
    def test_reorient_sagittal_to_axial(self):
        # a sagittal stack: slice axis = patient x, rows = -z (top of image is superior), columns = +y
        data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        aff = np.eye(4)
        aff[:3, 0] = [2.0, 0, 0]       # slices 2 mm apart along +x
        aff[:3, 1] = [0, 0, -1.0]      # rows go down (inferior)
        aff[:3, 2] = [0, 0.5, 0]       # columns go posterior
        vol = reorient_to_axial(Volume(data, aff))
        self.assertEqual(vol.acquired_plane, "sagittal")
        self.assertEqual(vol.plane, "axial")
        self.assertEqual(vol.shape, (5, 6, 4))
        np.testing.assert_allclose(vol.spacing, [1.0, 0.5, 2.0])
        self.assertTrue((np.diag(vol.affine[:3, [2, 1, 0]]) > 0).all())   # x, y, z all increase
        # every voxel keeps its patient position
        for idx in [(0, 0, 0), (4, 5, 3), (2, 3, 1)]:
            p = vol.affine @ np.array([*idx, 1.0])
            src = np.rint(np.linalg.inv(aff) @ p).astype(int)[:3]
            self.assertEqual(vol.data[idx], data[tuple(src)])
        self.assertEqual(vol.patient_left, "image_right")

    def test_resample_identity_and_shift(self):
        m = np.zeros((6, 7, 8), np.uint8)
        m[2:4, 3:5, 1:6] = 1
        aff = np.diag([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(resample_nearest(m, aff, m.shape, aff), m)
        shifted = aff.copy()
        shifted[:3, 3] = [0, 0, 1.0]           # target grid starts one column further along
        out = resample_nearest(m, aff, m.shape, shifted)
        np.testing.assert_array_equal(out[:, :, :-1], m[:, :, 1:])


class TestDicomRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        img, labels, info = phantom("T1_post")
        cls.img, cls.labels, cls.info = img, labels, info
        root = Path(cls.tmp.name)
        write_series(img, info["spacing"], root / "mr", "T1_post", name="rt")
        write_seg(info["lesion"], info["spacing"], root / "seg" / "tumor.dcm", "Tumor", name="rt")
        cls.datasets = io.read_folder(root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_series(self):
        series, segs = io.group_series(self.datasets)
        self.assertEqual(len(series), 1)
        self.assertEqual(len(segs), 1)
        vol = io.load_series(next(iter(series.values())))
        self.assertEqual(vol.sequence, "T1_post")
        self.assertEqual(vol.shape, self.img.shape)
        np.testing.assert_allclose(vol.spacing, self.info["spacing"])
        np.testing.assert_allclose(vol.data, np.clip(np.rint(self.img), 0, 65535))
        self.assertEqual(vol.patient_left, "image_right")

    def test_seg(self):
        series, segs = io.group_series(self.datasets)
        vol = io.load_series(next(iter(series.values())))
        masks = io.load_seg(segs[0])
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].role, "lesion")
        on_grid, warnings = io.doctor_labels_on(vol, masks)
        np.testing.assert_array_equal(on_grid["lesion"], self.info["lesion"])

    def test_frame_mismatch(self):
        series, segs = io.group_series(self.datasets)
        vol = io.load_series(next(iter(series.values())))
        vol.frame_of_reference = "1.2.3.4"
        masks = io.load_seg(segs[0])
        on_grid, warnings = io.doctor_labels_on(vol, masks)
        self.assertNotIn("lesion", on_grid)
        self.assertIn("seg_frame_mismatch", warnings)
        on_grid, _ = io.doctor_labels_on(vol, masks, allow_frame_mismatch=True)
        self.assertIn("lesion", on_grid)


class TestExport(unittest.TestCase):
    def test_pad_square(self):
        a, pad = pad_square(np.ones((3, 6)))
        self.assertEqual(a.shape, (6, 6))
        self.assertEqual(pad, [1, 2, 0, 0])

    def test_folder(self):
        img, labels, info = phantom("T1_post")
        k = img.shape[0] // 2
        masks = {lid: labels[k] == label_of(lid) for lid in layer_ids()}
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_slice_folder(tmp, "s0001", img[k], masks,
                                        {"source": "phantom", "sequence": "T1_post", "slice_index": k,
                                         "n_slices": img.shape[0], "pixel_spacing_mm": [1.5, 1.5]},
                                        reference={"lesion": info["lesion"][k]})
            meta = json.loads((folder / "meta.json").read_text())
            self.assertEqual(meta["reference_layers"], ["lesion"])
            from PIL import Image
            lab = np.asarray(Image.open(folder / "label.png"))
            self.assertEqual(lab.shape[0], lab.shape[1])
            t, b, l, r = meta["pad"]
            np.testing.assert_array_equal(lab[t:lab.shape[0] - b, l:lab.shape[1] - r], labels[k])
            for lid in layer_ids():
                self.assertTrue((folder / "masks" / f"{lid}.png").exists())
            with self.assertRaises(ValueError):
                write_slice_folder(tmp, "ReMIND-001", img[k], masks, {})


if __name__ == "__main__":
    unittest.main()
