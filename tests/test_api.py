"""Contract C3 API end to end: a real uvicorn server on a free port, phantom DICOM uploads over HTTP.

Rendering is limited to the flat backend (RENDER_BACKENDS=flat), so no network is used.
"""
import base64
import io as _io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from _setup import ROOT, phantom
from cortexview.layers import layer_ids, load_layers
from cortexview.phantom import write_seg, write_series

os.environ["RENDER_BACKENDS"] = "flat"
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
import uvicorn  # noqa: E402
from api.app import app  # noqa: E402
from shared.contracts import AnalyzeResponse, RenderDone  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _multipart(files, fields=None):
    boundary = uuid.uuid4().hex
    out = []
    for name, value in (fields or {}).items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for fname, data in files:
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{fname}"\r\n'
                   f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def _png(data_url):
    assert data_url.startswith("data:image/png;base64,")
    return np.asarray(Image.open(_io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.t1, cls.labels, cls.info = phantom("T1_post")
        write_series(cls.t1, cls.info["spacing"], root / "t1", "T1_post", name="p")
        write_seg(cls.info["lesion"], cls.info["spacing"], root / "seg" / "tumor.dcm", "Tumor", name="p")
        cls.files = [(p.name, p.read_bytes()) for p in sorted(root.rglob("*.dcm"))]

        port = _free_port()
        cls.base = f"http://127.0.0.1:{port}"
        cls.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def request(self, path, files=None, fields=None):
        if files is None:
            req = urllib.request.Request(self.base + path)
        else:
            body, ctype = _multipart(files, fields)
            req = urllib.request.Request(self.base + path, data=body, method="POST",
                                         headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def poll(self, url):
        for _ in range(120):
            status, body = self.request(url)
            if body.get("status") != "pending":
                return status, body
            time.sleep(0.5)
        self.fail("render never finished")

    def test_analyze_then_render(self):
        status, body = self.request("/api/analyze", self.files)
        self.assertEqual(status, 200, body)
        res = AnalyzeResponse.model_validate(body)
        self.assertEqual([l.id for l in res.layers], layer_ids())             # peel order
        self.assertEqual(set(res.metrics.layers), set(layer_ids()))
        self.assertEqual(res.render.status, "pending")
        self.assertEqual(res.slice.sequence, "T1_post")
        self.assertEqual(res.layers[-1].source, "doctor")                    # the uploaded tumour SEG
        mri = _png(res.slice.mri_png)
        self.assertEqual(mri.shape, (res.slice.height, res.slice.width))

        # flat layers: layer colour exactly inside the mask, transparent outside, never overlapping
        alpha_sum = np.zeros(mri.shape, int)
        for l, lr in zip(load_layers(), res.layers):
            rgba = _png(lr.flat_png)
            self.assertEqual(rgba.shape, mri.shape + (4,))
            on = rgba[..., 3] == 255
            alpha_sum += on
            self.assertEqual(on.sum(), res.metrics.layers[l.id].area_px)
            if on.any():
                self.assertTrue((rgba[on, :3] == l.rgb).all())
        self.assertLessEqual(alpha_sum.max(), 1)

        status, out = self.poll(res.render.poll_url)
        self.assertEqual(status, 200, out)
        done = RenderDone.model_validate(out)
        self.assertEqual(done.backend, "flat-fallback")
        self.assertTrue(done.fidelity.passed, done.fidelity)
        self.assertEqual(done.fidelity.phantom_lesion_pct, 0.0)
        for lr, il in zip(res.layers, done.layers):
            self.assertEqual(il.id, lr.id)
            # illustrated layers are cut with the same masks as the flat ones
            np.testing.assert_array_equal(_png(il.illustrated_png)[..., 3], _png(lr.flat_png)[..., 3])

        # C3: fetched once, then gone
        status, gone = self.request(res.render.poll_url)
        self.assertEqual((status, gone["error"]["code"]), (404, "not_found"))

    def test_zip_and_slice_index(self):
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in self.files:
                zf.writestr(f"scan/{name}", data)
        status, body = self.request("/api/analyze", [("scan.zip", buf.getvalue())], {"slice_index": "40"})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["slice"]["slice_index"], 40)

    def test_errors(self):
        status, body = self.request("/api/analyze", [("notes.txt", b"not a scan")])
        self.assertEqual((status, body["error"]["code"]), (422, "not_dicom"))
        status, body = self.request("/api/analyze", self.files, {"slice_index": "9999"})
        self.assertEqual((status, body["error"]["code"]), (422, "bad_slice_index"))
        status, body = self.request("/api/analyze", self.files, {"slice_index": "abc"})
        self.assertEqual((status, body["error"]["code"]), (422, "bad_slice_index"))
        status, body = self.request("/api/render/nope")
        self.assertEqual((status, body["error"]["code"]), (404, "not_found"))

    def test_health(self):
        status, body = self.request("/api/health")
        self.assertEqual((status, body["status"], body["render_backends"]), (200, "ok", ["flat"]))


if __name__ == "__main__":
    unittest.main()
