"""Contract C3 flow (pipeline.run) on phantom DICOM files."""
import io as _io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _setup import dice, phantom
from cortexview.layers import layer_ids
from cortexview.phantom import write_seg, write_series
from cortexview.pipeline import AnalyzeError, run


def files(folder):
    return sorted(Path(folder).rglob("*.dcm"))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.t1, cls.labels, cls.info = phantom("T1_post")
        flair, _, _ = phantom("FLAIR")
        write_series(cls.t1, cls.info["spacing"], root / "t1", "T1_post", name="p")
        write_series(flair, cls.info["spacing"], root / "flair", "FLAIR", name="p")
        write_seg(cls.info["lesion"], cls.info["spacing"], root / "seg" / "tumor.dcm", "Tumor", name="p")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_doctor_label_series_and_slice(self):
        res = run(files(self.root))
        self.assertEqual(res.volume.sequence, "T1_post")             # C3: tumour-label series, else T1-post
        true_k = int(np.argmax(self.info["lesion"].reshape(self.t1.shape[0], -1).sum(1)))
        self.assertEqual(res.slice_index, true_k)                   # largest lesion area
        self.assertEqual(res.slice.source["lesion"], "doctor")
        np.testing.assert_array_equal(res.slice.masks["lesion"], self.info["lesion"][true_k])
        self.assertEqual(set(res.slice.masks), set(layer_ids()))

    def test_lesion_from_flair_when_no_label(self):
        res = run(files(self.root / "t1") + files(self.root / "flair"))
        self.assertEqual(res.volume.sequence, "T1_post")
        self.assertEqual(res.slice.source["lesion"], "auto")
        self.assertIn("lesion_from_other_series", res.warnings)
        self.assertGreaterEqual(dice(res.segmentation.masks["lesion"], self.info["lesion"]), 0.7)

    def test_t1_alone_has_no_lesion(self):
        res = run(files(self.root / "t1"))
        self.assertFalse(res.slice.masks["lesion"].any())
        self.assertIn("lesion_auto_needs_flair", res.warnings)

    def test_explicit_slice(self):
        res = run(files(self.root / "t1"), slice_index=10)
        self.assertEqual(res.slice_index, 10)
        with self.assertRaises(AnalyzeError) as e:
            run(files(self.root / "t1"), slice_index=10_000)
        self.assertEqual(e.exception.code, "bad_slice_index")

    def test_file_objects(self):
        objs = [_io.BytesIO(p.read_bytes()) for p in files(self.root / "t1")]
        res = run(objs, slice_index=20)
        self.assertEqual(res.slice_index, 20)

    def test_zip_upload(self):
        import zipfile
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for p in files(self.root / "t1"):
                zf.write(p, arcname=f"study/{p.name}")
            zf.write(files(self.root / "seg")[0], arcname="study/seg/tumor.dcm")
        buf.seek(0)
        res = run([buf])
        self.assertEqual(res.slice.source["lesion"], "doctor")

    def test_zip_bomb_guard(self):
        import zipfile
        from cortexview.pipeline import expand_zips
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("big.dcm", b"\0" * 2_000_000)
        buf.seek(0)
        with self.assertRaises(AnalyzeError) as e:
            expand_zips([buf], max_bytes=1_000_000)
        self.assertEqual(e.exception.code, "too_large")

    def test_errors(self):
        with self.assertRaises(AnalyzeError) as e:
            run([_io.BytesIO(b"not a dicom file at all" * 10)])
        self.assertEqual(e.exception.code, "not_dicom")
        with self.assertRaises(AnalyzeError) as e:
            run(files(self.root / "seg"))
        self.assertEqual(e.exception.code, "not_mr")

    def test_few_slices_fall_back_to_2d(self):
        with tempfile.TemporaryDirectory() as tmp:
            k = self.t1.shape[0] // 2
            write_series(self.t1[k - 3:k + 3], self.info["spacing"], tmp, "T1_post", name="thin")
            res = run(files(tmp))
        self.assertIn("low_slice_count", res.warnings)
        self.assertEqual(res.slice.masks["skull"].shape, self.t1.shape[1:])


if __name__ == "__main__":
    unittest.main()
