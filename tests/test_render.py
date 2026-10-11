"""Contract C5 render chain and fidelity gate, without any network."""
import unittest
import base64
from pathlib import Path
from unittest import mock

import numpy as np

import _setup  # noqa: F401  (puts backend/ on the path)
import renderer
from cortexview.export import guide_rgb, label_map
from cortexview.layers import load_layers
from renderer import fidelity


def toy_masks(n=64):
    """Concentric rings, one per layer, plus a lesion blob."""
    yy, xx = np.indices((n, n))
    r = np.hypot(yy - n / 2, xx - n / 2)
    masks = {"skull": (r < 30) & (r >= 26), "gray_matter": (r < 26) & (r >= 20),
             "white_matter": (r < 20) & (r >= 10), "csf": r < 10,
             "lesion": np.zeros((n, n), bool)}
    masks["lesion"][28:34, 40:46] = True
    for k in ("skull", "gray_matter", "white_matter", "csf"):
        masks[k] &= ~masks["lesion"]                 # C1: layers never overlap
    return masks


class TestFidelity(unittest.TestCase):
    def test_guide_scores_perfect(self):
        m = toy_masks()
        s = fidelity.score(guide_rgb(label_map(m)), m)
        self.assertEqual(s.mean_iou, 1.0)
        self.assertEqual(s.phantom_lesion_pct, 0.0)
        self.assertTrue(s.passed)

    def test_phantom_lesion_fails(self):
        m = toy_masks()
        img = guide_rgb(label_map(m))
        img[10:20, 30:40] = next(l.rgb for l in load_layers() if l.id == "lesion")   # a red blob in healthy tissue
        s = fidelity.score(img, m)
        self.assertGreater(s.phantom_lesion_pct, fidelity.MAX_PHANTOM_PCT)
        self.assertFalse(s.passed)

    def test_flat_render_keeps_colour_families(self):
        m = toy_masks()
        g = guide_rgb(label_map(m))
        mri = np.random.default_rng(0).integers(0, 256, g.shape[:2]).astype(np.uint8)
        self.assertEqual(fidelity.score(renderer.flat_render(g, mri), m).mean_iou, 1.0)


class TestPrompt(unittest.TestCase):
    def test_prompt_follows_layers_and_has_no_numbers(self):
        with_lesion = renderer.build_prompt(["skull", "gray_matter", "white_matter", "csf", "lesion"])
        without = renderer.build_prompt(["skull", "gray_matter", "white_matter", "csf"])
        self.assertIn("red = lesion", with_lesion)
        self.assertNotIn("red = lesion", without)
        self.assertIn("do not add any red", without)
        self.assertFalse(any(ch.isdigit() for ch in with_lesion.replace("image 0", "")))


class TestChain(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(backends=["local", "flat"], local_url="http://tunnel.invalid", attempts=3,
                    cache_dir=Path("does-not-exist"))
        return renderer.Config(**{**base, **kw})

    def test_bad_ai_output_falls_back_to_flat(self):
        m = toy_masks()
        g = guide_rgb(label_map(m))
        noise = renderer._png(np.full_like(g, 128))                    # grey mush: fails the gate
        with mock.patch.object(renderer, "_call_local", return_value=noise) as call:
            r = renderer.render(g, m, cfg=self.cfg())
        self.assertEqual(r.backend, "flat-fallback")
        self.assertEqual(call.call_count, 3)                          # every seed was tried
        self.assertEqual(r.attempts, 3)

    def test_good_ai_output_is_used(self):
        m = toy_masks()
        g = guide_rgb(label_map(m))
        with mock.patch.object(renderer, "_call_local", return_value=renderer._png(g)):
            r = renderer.render(g, m, cfg=self.cfg())
        self.assertEqual((r.backend, r.attempts, r.passed), ("local-lora-v1", 1, True))

    def test_unreachable_backend_is_skipped_at_once(self):
        m = toy_masks()
        with mock.patch.object(renderer, "_call_local", side_effect=renderer._Skip()) as call:
            r = renderer.render(guide_rgb(label_map(m)), m, cfg=self.cfg())
        self.assertEqual((r.backend, call.call_count), ("flat-fallback", 1))

    def test_workers_rotate_tokens_then_use_fallback(self):
        m = toy_masks()
        g = guide_rgb(label_map(m))
        cfg = renderer.Config(backends=["workers", "flat"], cf_account="original-account", cf_token="original-token",
                              cf_accounts=("account-one", "account-two"),
                              cf_tokens=("bad-token", "good-token"), cf_backend="workers-klein-4b")
        response = ('{"success": true, "result": {"image": "' + base64.b64encode(renderer._png(g)).decode() + '"}}').encode()
        with mock.patch.object(renderer, "_post", side_effect=[renderer._Skip(), response]) as post:
            result = renderer.render(g, m, cfg=cfg)
        self.assertEqual(result.backend, "workers-klein-4b")
        self.assertEqual(post.call_count, 2)
        self.assertIn("/accounts/original-account/", post.call_args_list[0].args[0])
        self.assertIn("/accounts/account-one/", post.call_args_list[1].args[0])
        self.assertEqual(post.call_args_list[0].args[2]["Authorization"], "Bearer original-token")
        self.assertEqual(post.call_args_list[1].args[2]["Authorization"], "Bearer bad-token")

    def test_legacy_pair_precedes_extra_rotating_pairs(self):
        cfg = renderer.Config(cf_account="first-account", cf_token="first-token",
                              cf_accounts=("second-account",), cf_tokens=("second-token",))
        self.assertEqual(cfg.worker_credentials,
                         (("first-account", "first-token"), ("second-account", "second-token")))

    def test_mismatched_rotating_lists_disable_workers(self):
        cfg = renderer.Config(backends=["workers", "flat"], cf_accounts=("account-one", "account-two"),
                              cf_tokens=("only-one-token",))
        self.assertNotIn("workers", cfg.available())

    def test_cut_layers_stay_inside_masks(self):
        m = toy_masks()
        img = np.full((64, 64, 3), 200, np.uint8)
        for lid, rgba in renderer.cut_layers(img, m).items():
            np.testing.assert_array_equal(rgba[..., 3] == 255, m[lid])

    def test_cache_round_trip(self):
        import tempfile
        m = toy_masks()
        g = guide_rgb(label_map(m))
        with tempfile.TemporaryDirectory() as d:
            rec = self.cfg(cache_dir=Path(d), cache_record=True, backends=["cache", "local", "flat"])
            with mock.patch.object(renderer, "_call_local", return_value=renderer._png(g)):
                renderer.render(g, m, cfg=rec)
            r = renderer.render(g, m, cfg=self.cfg(cache_dir=Path(d), backends=["cache", "flat"]))
        self.assertEqual(r.backend, "cache")


if __name__ == "__main__":
    unittest.main()
