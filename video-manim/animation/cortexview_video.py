"""CortexView explainer, about 90 s, Manim Community Edition (tested with 0.22.0).

Scenes (each can be rendered alone for debugging; CortexViewFull plays all six back to back):

    S1Hook       0-10 s   an image is a grid of numbers; numbers become labels
    S2Layers    10-25 s   the synthetic slice, its label map, the five layers
    S3Measure   25-43 s   A = N s^2 and the lesion's largest boundary-to-boundary distance
    S4Design    43-60 s   generative where it helps, deterministic where it matters (design)
    S5Status    60-75 s   what works today, what is in progress, what is planned
    S6Closing   75-90 s   layered slice with measurement pins, title

Every image and number comes from animation/phantom.py, a verified port of the frontend's synthetic
phantom (frontend/src/cortexview/demo/phantom.ts). Timing lives in animation/config.py.

Render (from the repo root; see README.md for the helper script):
    manim -pql video-manim/animation/cortexview_video.py CortexViewFull
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `animation` imports as a package

import numpy as np
from manim import *  # noqa: F403

from animation import colors as C
from animation import components as K
from animation.config import BEATS, SEGMENTS
from animation.phantom import coarse_slice, grey, lesion_boundary_pair, main_slice

config.background_color = C.BG
SEG_LEN = dict(SEGMENTS)


class Timeline(MovingCameraScene):
    """Adds beat-based timing: `self.at("beat")` waits until that beat of the current segment."""

    def begin(self, key: str):
        self.seg_key, self.seg_t0, self.beats = key, self.time, BEATS[key]

    def at(self, beat: str | float):
        t = self.beats[beat] if isinstance(beat, str) else beat
        dt = self.seg_t0 + t - self.time
        if dt > 1e-3:
            self.wait(dt)
        elif dt < -0.1:
            logger.warning(f"[{self.seg_key}] beat {beat!r} starts {-dt:.2f}s late; loosen config.BEATS")

    def finish(self):
        """Fade everything out so the segment ends exactly on its configured length."""
        self.at("out" if "out" in self.beats else "fade")
        remaining = self.seg_t0 + SEG_LEN[self.seg_key] - self.time
        mobs = [m for m in self.mobjects if m is not self.camera.frame]
        if mobs and remaining > 0.05:
            self.play(*[FadeOut(m) for m in mobs], run_time=remaining)
        elif remaining > 1e-3:
            self.wait(remaining)
        self.clear()
        self.camera.frame.set(width=config.frame_width).move_to(ORIGIN)

    # ------------------------------------------------------------------ 1. hook

    def seg_hook(self):
        self.begin("hook")
        S = main_slice()
        big = K.image(K.mri_rgba(S["mri"], S["lbl"]), 5.8).shift(DOWN * 0.3)
        tag = K.phantom_tag()

        self.at("image_in")
        self.play(FadeIn(big, scale=0.96), FadeIn(tag), run_time=0.9)
        self.at("headline")
        head = K.headline("An image is more than pixels.")
        self.play(FadeIn(head, shift=DOWN * 0.15), run_time=1.0)

        # the same phantom function sampled on a 40 x 40 grid: visible pixels
        n = 40
        clbl, cmri = coarse_slice(n)
        g = grey(cmri)
        side = 5.8 / n
        grid = K.pixel_grid(np.repeat(g[..., None], 3, axis=2), side).shift(DOWN * 0.3)
        order = np.argsort([np.linalg.norm(c.get_center() - grid.get_center()) for c in grid])
        self.at("to_pixels")
        self.play(FadeOut(big, run_time=1.0),
                  LaggedStart(*[FadeIn(grid[k]) for k in order], lag_ratio=0.0006, run_time=1.2))

        # the 5 x 5 patch with the most different labels: show its numbers
        best, pj, pi = -1, 0, 0
        for jj in range(14, 22):
            for ii in range(0, n - 5):
                k = len(np.unique(clbl[jj:jj + 5, ii:ii + 5]))
                if k > best:
                    best, pj, pi = k, jj, ii
        self.at("numbers")
        self.play(grid.animate.shift(LEFT * 3.3), run_time=0.8)
        corners = [grid[pj * n + pi], grid[(pj + 4) * n + pi + 4]]
        frame = Rectangle(width=5 * side, height=5 * side, color=C.MATH, stroke_width=3)
        frame.move_to((corners[0].get_center() + corners[1].get_center()) / 2)
        cell = 0.78
        vals = VGroup()
        for jj in range(5):
            for ii in range(5):
                v = int(round(float(g[pj + jj, pi + ii])))
                t = K.text(str(v), 26, C.INK, font=C.MONO)
                t.move_to(np.array([3.6 + (ii - 2) * cell, (2 - jj) * cell - 0.1, 0]))
                vals.add(t)
        table_box = Rectangle(width=5 * cell, height=5 * cell, color=C.MATH, stroke_width=2)
        table_box.move_to([3.6, -0.1, 0])
        links = VGroup(
            Line(frame.get_corner(UR), table_box.get_corner(UL), stroke_width=1.2, color=C.MATH),
            Line(frame.get_corner(DR), table_box.get_corner(DL), stroke_width=1.2, color=C.MATH),
        ).set_opacity(0.5)
        cap = K.text("brightness values", 20, C.MUTED).next_to(table_box, DOWN, buff=0.25)
        self.play(Create(frame), run_time=0.4)
        self.play(Create(links), FadeIn(table_box), LaggedStart(*[FadeIn(v) for v in vals], lag_ratio=0.03),
                  FadeIn(cap), run_time=0.9)

        # numbers become labels: every pixel takes its layer's color
        self.at("to_labels")
        lab_vals, lab_anims = VGroup(), []
        for idx, v in enumerate(vals):
            lab = int(clbl[pj + idx // 5, pi + idx % 5])
            col = C.MUTED if lab == 0 else C.layer_color(next(L["id"] for L in C.LAYERS if L["label"] == lab))
            lab_vals.add(K.text(str(lab), 30, col, font=C.MONO, weight="BOLD").move_to(v))
        cap2 = K.text("labels", 20, C.MUTED).move_to(cap)
        rgb = np.zeros((n, n, 3))
        for L in C.LAYERS:
            rgb[clbl == L["label"]] = L["rgb"]
        for k in order:
            col = rgb[k // n, k % n]
            if clbl[k // n, k % n] > 0:
                lab_anims.append(grid[k].animate.set_fill(K._hex(col), 1))
        self.play(ReplacementTransform(vals, lab_vals), ReplacementTransform(cap, cap2), run_time=1.0)
        self.play(LaggedStart(*lab_anims, lag_ratio=0.0015, run_time=1.6))
        self.finish()

    # ------------------------------------------------------------------ 2. layers

    def seg_layers(self):
        self.begin("layers")
        S = main_slice()
        lbl, mri = S["lbl"], S["mri"]
        pos, h = np.array([-3.6, -0.2, 0]), 5.4
        mri_im = K.image(K.mri_rgba(mri, lbl), h).move_to(pos)
        tag = K.phantom_tag()
        note = K.text("Synthetic phantom · generated by code · not a patient", 19, C.MATH)
        note.next_to(mri_im, DOWN, buff=0.3)

        # the grey slice arrives as an upload (the demo uses the synthetic phantom)
        drop = DashedVMobject(RoundedRectangle(corner_radius=0.2, width=h * 0.95, height=h + 0.3), num_dashes=60)
        drop.set_stroke(C.MUTED, 1.5).move_to(pos)
        up_lab = K.text("↑ uploaded MRI slice", 20, C.MUTED).next_to(drop, UP, buff=0.1)
        self.at("image_in")
        self.play(Create(drop), FadeIn(up_lab), run_time=0.5)
        self.play(FadeIn(mri_im, shift=UP * 0.6), run_time=0.9)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.6)
        self.wait(0.4)
        self.play(ReplacementTransform(note, tag), run_time=0.7)

        # segmentation sweeps down the slice: grey pixels become labelled tissue colors
        lab_arr = K.labels_rgba(lbl)
        n = lab_arr.shape[0]
        band = 11
        top = pos[1] + h / 2
        strips = Group()
        for r0 in range(0, n, band):
            r1 = min(n, r0 + band)
            st = K.image(lab_arr[r0:r1], h * (r1 - r0) / n)
            st.move_to([pos[0], top - h * (r0 + r1) / 2 / n, 0])
            strips.add(st)
        scan = Line(LEFT * h * 0.45, RIGHT * h * 0.45, color=C.DET, stroke_width=3).move_to([pos[0], top, 0])
        dim = K.image(K.mri_rgba(mri, lbl, 0.35), h).move_to(pos)
        cap = K.text("segmentation: one label per pixel", 20, C.MUTED).move_to(up_lab)
        self.at("label_map")
        self.add(scan)
        self.play(FadeIn(dim), FadeOut(mri_im), FadeOut(up_lab),
                  LaggedStart(*[FadeIn(s) for s in strips], lag_ratio=0.12),
                  scan.animate.move_to([pos[0], top - h, 0]), FadeIn(cap), run_time=1.6, rate_func=linear)
        labels = K.image(lab_arr, h).move_to(pos)          # swap the strips for one seamless image
        self.add(labels)
        self.remove(*strips)
        self.play(FadeOut(scan), FadeOut(drop), run_time=0.3)

        # empty tiles on the right, then each layer lifts out of the label map into its tile
        tile_pos = {"skull": [0.85, 1.25], "gray_matter": [3.2, 1.25], "white_matter": [5.55, 1.25],
                    "csf": [1.85, -1.75], "lesion": [4.85, -1.75]}
        frames, legends = {}, {}
        for lid, (x, y) in tile_pos.items():
            fr = RoundedRectangle(corner_radius=0.12, width=2.15, height=2.15, stroke_width=1.2,
                                  stroke_color=C.FAINT).move_to([x, y, 0])
            frames[lid] = fr
            legends[lid] = K.legend_row(lid, 18).next_to(fr, DOWN, buff=0.14)
        self.at("split")
        self.play(LaggedStart(*[Create(f) for f in frames.values()], lag_ratio=0.12), run_time=1.2)

        for L in C.LAYERS:
            lid = L["id"]
            self.at(lid)
            layer = K.image(K.labels_rgba(lbl, only=[L["label"]]), h).move_to(pos)
            self.add(layer)
            target = layer.copy().set(height=2.0).move_to(frames[lid])
            self.play(Transform(layer, target), frames[lid].animate.set_stroke(C.layer_color(lid), opacity=0.7),
                      FadeIn(legends[lid], shift=UP * 0.1), run_time=0.9)

        self.at("headline")
        self.play(FadeIn(K.headline("One image. Multiple layers of meaning."), shift=DOWN * 0.15), run_time=1.0)
        self.finish()

    # ------------------------------------------------------------------ 3. measurement

    def seg_measure(self):
        self.begin("measure")
        S = main_slice()
        lbl, D, sp, M = S["lbl"], S["D"], S["sp"], S["metrics"]
        les = M["lesion"]
        nfull = lbl.shape[0]
        n = 72                                                   # crop around the lesion, in pixels
        ci, cj = int(round(les.cx)), int(round(les.cy))
        i0, j0 = ci - n // 2, cj - n // 2
        crop = lbl[j0:j0 + n, i0:i0 + n]
        CL = np.array([-3.4, -0.1, 0])
        side = 6.0 / n

        # zoom: the whole label map, then a pixel-exact blow-up of the lesion's neighbourhood
        full_h = 6.0
        full = K.image(K.labels_rgba(lbl), full_h, nearest=True).move_to(CL)
        unit = full_h / nfull
        oc = CL + np.array([(i0 + n / 2 - nfull / 2) * unit, (nfull / 2 - j0 - n / 2) * unit, 0])
        box = Square(n * unit, color=C.MATH, stroke_width=2.5).move_to(oc)
        tag = K.phantom_tag()
        self.at("zoom")
        self.play(FadeIn(full), FadeIn(tag), run_time=0.5)
        self.play(Create(box), run_time=0.35)
        k = nfull / n
        self.play(full.animate.scale(k, about_point=oc).shift(CL - oc),
                  box.animate.scale(k, about_point=oc).shift(CL - oc).set_stroke(opacity=0), run_time=1.0)
        bright = K.image(K.labels_rgba(crop), 6.0, nearest=True).move_to(CL)
        dimmed = bright.copy()
        dimmed.pixel_array = dimmed.pixel_array.copy()
        dimmed.pixel_array[..., :3] = (dimmed.pixel_array[..., :3] * 0.32).astype(np.uint8)
        lines = K.grid_lines(n, side, color=C.BG, width=0.7).move_to(CL).set_opacity(0.55)
        self.add(bright, lines, tag)
        self.remove(box)
        self.play(FadeOut(full), run_time=0.25)

        # one pixel: an s x s square
        pi_, pj_ = les.anchor_px[0] - i0, les.anchor_px[1] - j0
        px_small = Square(side, color=C.MATH, stroke_width=3).move_to(CL + K.pixel_center(pi_, pj_, n, side))
        big_sq = Square(1.2, color=C.MATH, stroke_width=3).set_fill(C.layer_color("lesion"), 1)
        big_sq.move_to([1.7, 2.25, 0])
        links = VGroup(Line(px_small.get_corner(UR), big_sq.get_corner(UL)),
                       Line(px_small.get_corner(DR), big_sq.get_corner(DL))).set_stroke(C.MATH, 1.2, 0.5)
        brace = Brace(big_sq, DOWN, buff=0.08, color=C.MUTED)
        s_lab = K.math(f"<i>s</i> = {sp:.2f} mm", 30, C.INK).next_to(brace, DOWN, buff=0.08)
        s_area = K.math(f"area = <i>s</i>² ≈ {sp * sp:.2f} mm²", 32, C.MATH).next_to(big_sq, RIGHT, buff=0.5)
        self.at("pixel")
        self.play(Create(px_small), run_time=0.5)
        self.play(TransformFromCopy(px_small, big_sq), Create(links), run_time=1.0)
        self.play(GrowFromCenter(brace), FadeIn(s_lab), run_time=0.7)
        self.play(FadeIn(s_area, shift=LEFT * 0.1), run_time=0.7)

        # count: lesion pixels light up row by row, N ticks up to the real count
        runs, counts = K.run_rects(crop == 10, side, C.layer_color("lesion"))
        runs.shift(CL)                                           # run_rects is centred at the origin
        cum = np.cumsum([0] + counts)
        assert cum[-1] == les.px, "crop must contain the whole lesion"
        tracker = ValueTracker(0)
        n_pos = np.array([3.9, 0.25, 0])
        counter = always_redraw(lambda: K.math(f"<i>N</i> = {int(tracker.get_value()):,}", 40, C.INK).move_to(n_pos))
        self.at("count")
        self.add(counter)
        self.play(FadeIn(dimmed), FadeOut(bright), FadeOut(links), px_small.animate.set_stroke(opacity=0),
                  run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in runs], lag_ratio=0.08),
                  tracker.animate.set_value(les.px), run_time=1.3, rate_func=linear)
        self.remove(counter)
        n_final = K.math(f"<i>N</i> = {les.px:,} pixels", 40, C.INK).move_to(n_pos)
        self.add(n_final)

        f1 = K.math("<i>A</i> = <i>N</i> · <i>s</i>²", 58, C.MATH).move_to([3.9, -0.75, 0])
        f2 = K.math(f"= {les.px:,} × ({sp:.2f} mm)² ≈ {les.mm2:,.0f} mm²", 28, C.INK)
        f2.next_to(f1, DOWN, buff=0.3)
        self.at("formula")
        self.play(Write(f1), run_time=1.0)
        self.play(FadeIn(f2, shift=UP * 0.1), run_time=0.9)

        # geometry: boundary pixels and the longest distance between two of them
        edge = np.argwhere((lbl == 10) & (D == 0))
        edges = VGroup(*[Square(side, stroke_width=1.6, color=WHITE).move_to(CL + K.pixel_center(jj_i - i0, jj_j - j0, n, side))
                         for jj_j, jj_i in edge])
        self.at("boundary")
        self.play(runs.animate.set_opacity(0.55), LaggedStart(*[FadeIn(e) for e in edges], lag_ratio=0.004),
                  run_time=1.2)
        (ai, aj), (bi, bj) = lesion_boundary_pair(lbl, D)
        pa, pb = CL + K.pixel_center(ai - i0, aj - j0, n, side), CL + K.pixel_center(bi - i0, bj - j0, n, side)
        seg = Line(pa, pb, color=C.MATH, stroke_width=4)
        ends = VGroup(Dot(pa, 0.07, color=C.MATH), Dot(pb, 0.07, color=C.MATH))
        f3 = K.math(f"<i>d</i> = max ‖<i>p</i>ᵢ − <i>p</i>ⱼ‖ · <i>s</i> ≈ {les.diam:.1f} mm", 30, C.MATH)
        f3.move_to([3.9, -2.75, 0])
        self.at("diameter")
        self.play(FadeIn(ends), Create(seg), run_time=1.0)
        self.play(FadeIn(f3, shift=UP * 0.1), run_time=0.9)

        self.at("headline")
        self.play(FadeIn(K.headline("Pixels  →  Geometry  →  Measurements"), shift=DOWN * 0.15), run_time=1.0)
        self.finish()

    # ------------------------------------------------------------------ 4. design

    def seg_design(self):
        self.begin("design")
        S = main_slice()
        lbl, mri, D, M = S["lbl"], S["mri"], S["D"], S["metrics"]
        g_txt = K.text("Generative where it helps.", 36, C.GEN)
        d_txt = K.text("Deterministic where it matters.", 36, C.DET)
        motto = VGroup(g_txt, d_txt).arrange(RIGHT, buff=0.45).to_edge(UP, buff=0.55)
        self.at("motto")
        self.play(FadeIn(g_txt, shift=DOWN * 0.15), run_time=1.0)
        self.play(FadeIn(d_txt, shift=DOWN * 0.15), run_time=1.0)

        masks = K.image(K.labels_rgba(lbl), 2.4).move_to([0, 0.6, 0])
        m_cap = K.text("masks (label map)", 20, C.MUTED).next_to(masks, DOWN, buff=0.12)
        self.at("masks")
        self.play(FadeIn(masks, scale=0.95), FadeIn(m_cap), run_time=0.8)

        # left: the image model only styles; its output is clipped to the masks (design, not trained)
        gen_box = K.node("Image model", "FLUX.2 klein + LoRA · in development", "progress", 3.3, C.GEN)
        gen_box.move_to([-4.5, 1.35, 0])
        illus = K.image(K.illus_rgba(lbl, mri, D), 2.0).move_to([-4.5, -1.05, 0])
        frame_l = SurroundingRectangle(illus, color=C.GEN, buff=0.08, corner_radius=0.1, stroke_width=1.5)
        g_cap = K.text("decides how it looks", 22, C.GEN).next_to(frame_l, DOWN, buff=0.15)
        g_note = K.text("mock-up shaded by code;\nfinal look stays clipped to the masks", 15, C.MUTED,
                        line_spacing=0.8).next_to(g_cap, DOWN, buff=0.1)
        a1 = Arrow(masks.get_left() + UP * 0.6, gen_box.get_right(), buff=0.15, color=C.GEN, stroke_width=3)
        a2 = Arrow(gen_box.get_bottom(), frame_l.get_top(), buff=0.1, color=C.GEN, stroke_width=3)
        self.at("generative")
        self.play(GrowArrow(a1), FadeIn(gen_box, shift=LEFT * 0.15), run_time=1.0)
        self.play(GrowArrow(a2), FadeIn(illus), Create(frame_l), run_time=1.1)
        self.play(FadeIn(g_cap), FadeIn(g_note), run_time=0.8)

        # right: deterministic code counts pixels; every number shown comes from here
        det_box = K.node("Count pixels", "A = N·s² · runs today", "working", 3.3, C.DET).move_to([4.5, 1.35, 0])
        nums = VGroup(
            K.text(f"Lesion area   {M['lesion'].mm2:,.0f} mm²", 24, C.INK),
            K.text(f"Lesion diameter   {M['lesion'].diam:.1f} mm", 24, C.INK),
            K.text(f"White matter   {M['white_matter'].pct:.1f} %", 24, C.INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([4.5, -1.0, 0])
        frame_r = SurroundingRectangle(nums, color=C.DET, buff=0.25, corner_radius=0.1, stroke_width=1.5)
        d_cap = K.text("decides every number", 22, C.DET).next_to(frame_r, DOWN, buff=0.18)
        a3 = Arrow(masks.get_right() + UP * 0.6, det_box.get_left(), buff=0.15, color=C.DET, stroke_width=3)
        a4 = Arrow(det_box.get_bottom(), frame_r.get_top(), buff=0.1, color=C.DET, stroke_width=3)
        self.at("deterministic")
        self.play(GrowArrow(a3), FadeIn(det_box, shift=RIGHT * 0.15), run_time=1.0)
        self.play(GrowArrow(a4), Create(frame_r), LaggedStart(*[FadeIn(x) for x in nums], lag_ratio=0.25),
                  run_time=1.3)
        self.play(FadeIn(d_cap), run_time=0.6)

        foot = K.text("The design. The image model is not trained yet; the measurements already run in the phantom demo.",
                      17, C.MUTED).to_edge(DOWN, buff=0.3)
        same = K.text("same input  →  same numbers", 22, C.DET).move_to(d_cap)
        self.at("same")
        self.play(Circumscribe(frame_r, color=C.DET, run_time=1.0), ReplacementTransform(d_cap, same),
                  FadeIn(foot), run_time=1.0)
        self.finish()

    # ------------------------------------------------------------------ 5. status

    def seg_status(self):
        self.begin("status")
        title = K.headline("Where it stands")
        self.at("title")
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.9)

        xs, top, low = [-5.3, -2.65, 0.0, 2.65, 5.3], 1.15, -1.35
        w = 2.42
        main = [
            K.node("MRI series (DICOM)", "Python reader, tested", "working", w),
            K.node("Classical segmentation", "no learned weights, tested", "working", w),
            K.node("Label map", "5 layers", "working", w),
            K.node("Measurements", "N·s², in the web demo", "working", w),
            K.node("Layered viewer", "web demo, phantom", "working", w),
        ]
        for nd, x in zip(main, xs):
            nd.move_to([x, top, 0])
        arrows = [Arrow(a.get_right(), b.get_left(), buff=0.04, stroke_width=3, color=C.MUTED,
                        max_tip_length_to_length_ratio=0.5) for a, b in zip(main, main[1:])]
        self.at("main_row")
        self.play(FadeIn(main[0], shift=RIGHT * 0.1), run_time=0.6)
        for nd, ar in zip(main[1:], arrows):
            self.play(GrowArrow(ar), FadeIn(nd, shift=RIGHT * 0.1), run_time=0.75)

        remind = K.node("ReMIND, 53 patients", "public MRI, CC BY 4.0", "working", w).move_to([xs[1], low, 0])
        guide = K.node("Guide images", "80 built with SynthSeg", "working", w).move_to([xs[2], low, 0])
        a_r = Arrow(remind.get_right(), guide.get_left(), buff=0.04, stroke_width=3, color=C.MUTED,
                    max_tip_length_to_length_ratio=0.5)
        a_g = Arrow(main[2].get_bottom(), guide.get_top(), buff=0.06, stroke_width=3, color=C.MUTED,
                    max_tip_length_to_length_ratio=0.3)
        self.at("guides")
        self.play(FadeIn(remind, shift=UP * 0.1), run_time=0.7)
        self.play(GrowArrow(a_r), GrowArrow(a_g), FadeIn(guide, shift=UP * 0.1), run_time=0.9)

        model = K.node("FLUX.2 klein 4B + LoRA", "local fine-tune being set up", "progress", w, C.GEN)
        model.move_to([xs[3], low, 0])
        fid = K.node("Fidelity check", "IoU script written", "progress", w).move_to([xs[4], low, 0])
        a_m = Arrow(guide.get_right(), model.get_left(), buff=0.04, stroke_width=3, color=C.GEN,
                    max_tip_length_to_length_ratio=0.5)
        a_f = Arrow(model.get_right(), fid.get_left(), buff=0.04, stroke_width=3, color=C.MUTED,
                    max_tip_length_to_length_ratio=0.5)
        a_v = DashedLine(fid.get_top(), main[4].get_bottom(), buff=0.06, color=C.MUTED, stroke_width=2.5)
        self.at("model")
        self.play(GrowArrow(a_m), FadeIn(model, shift=RIGHT * 0.1), run_time=0.8)
        self.play(GrowArrow(a_f), FadeIn(fid, shift=RIGHT * 0.1), Create(a_v), run_time=0.8)

        upload = K.node("Upload in the web app", "API server planned", "planned", w).move_to([xs[0], low, 0])
        a_u = DashedLine(upload.get_top(), main[0].get_bottom(), buff=0.06, color=C.MUTED, stroke_width=2.5)
        legend = K.status_legend().to_edge(DOWN, buff=0.45)
        self.at("planned")
        self.play(FadeIn(upload, shift=UP * 0.1), Create(a_u), FadeIn(legend), run_time=0.9)
        self.finish()

    # ------------------------------------------------------------------ 6. closing

    def seg_closing(self):
        self.begin("closing")
        S = main_slice()
        lbl, mri, M, sp = S["lbl"], S["mri"], S["metrics"], S["sp"]
        h = 6.2
        frame = self.camera.frame
        frame.save_state()
        frame.scale(0.86)

        mri_im = K.image(K.mri_rgba(mri, lbl, 0.6), h)
        layers = [K.image(K.labels_rgba(lbl, only=[L["label"]]), h) for L in C.LAYERS]
        self.at("image_in")
        self.play(FadeIn(mri_im, scale=0.97), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(l) for l in layers], lag_ratio=0.45), run_time=1.7)

        n = lbl.shape[0]
        unit = h / n
        to_pt = lambda ij: np.array([(ij[0] - n / 2 + 0.5) * unit, (n / 2 - ij[1] - 0.5) * unit, 0])
        pin_spec = [  # layer, label, side (+1 right, -1 left), label y
            ("lesion", f"Lesion (simulated)\n{M['lesion'].mm2:,.0f} mm² · {M['lesion'].diam:.1f} mm", 1, 1.9),
            ("gray_matter", f"Gray matter  {M['gray_matter'].pct:.1f} %", 1, -1.6),
            ("white_matter", f"White matter  {M['white_matter'].pct:.1f} %", -1, 0.9),
            ("csf", f"CSF / ventricles  {M['csf'].pct:.1f} %", -1, -0.9),
        ]
        pins = VGroup()
        for lid, label, sgn, ly in pin_spec:
            p = to_pt(M[lid].anchor_px)
            dot = Dot(p, 0.075, color=C.INK).set_stroke(C.BG, 2)
            t = K.text(label, 20, C.INK, line_spacing=0.8)
            elbow = np.array([sgn * 2.75, ly, 0])
            t.next_to(elbow, RIGHT if sgn > 0 else LEFT, buff=0.12)
            # dark underlay keeps the line visible over tissue of its own color (e.g. white matter)
            ln = VGroup(Line(p, elbow, stroke_width=4.5, color=C.BG, stroke_opacity=0.8),
                        Line(p, elbow, stroke_width=1.8, color=C.layer_color(lid)))
            pins.add(VGroup(ln, dot, t))
        pct_note = K.text("% = share of the intracranial area", 15, C.MUTED).move_to([0, -3.15, 0])
        self.at("pins")
        self.play(LaggedStart(*[AnimationGroup(Create(pg[0]), FadeIn(pg[1]), FadeIn(pg[2])) for pg in pins],
                              lag_ratio=0.3), FadeIn(pct_note), run_time=2.2)

        # pull back: the slice moves aside, the title arrives
        stack = Group(mri_im, *layers)
        title = K.text("CortexView", 76, C.INK, weight="BOLD")
        tagline = K.text("Turning complex brain images into\nstructured, measurable visual information.", 26,
                         C.MUTED, line_spacing=0.9)
        sub = K.text("Visualization  ·  Segmentation  ·  Measurement", 24, C.MATH)
        right = VGroup(title, tagline, sub).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([3.0, 0.3, 0])
        rule = Line(LEFT, RIGHT, color=C.FAINT).set_width(right.width).next_to(title, DOWN, buff=0.18)
        rule.align_to(title, LEFT)
        self.at("title")
        self.play(FadeOut(pins), FadeOut(pct_note), run_time=0.5)
        self.play(Restore(frame), stack.animate.set(height=5.6).move_to([-3.7, 0, 0]), run_time=1.4)
        self.play(FadeIn(title, shift=UP * 0.15), Create(rule), run_time=0.9)
        self.play(FadeIn(tagline), run_time=0.6)
        self.at("subtitle")
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.9)
        disclaimer = K.text("Educational aid · not for diagnosis", 19, C.MUTED).to_edge(DOWN, buff=0.75)
        tag = K.phantom_tag()
        self.at("disclaimer")
        self.play(FadeIn(disclaimer), FadeIn(tag), run_time=0.9)
        self.finish()


# ---------------------------------------------------------------------- scenes to render

class CortexViewFull(Timeline):
    """The whole video, all six segments in one file."""

    def construct(self):
        for key, _ in SEGMENTS:
            getattr(self, f"seg_{key}")()


class S1Hook(Timeline):
    def construct(self):
        self.seg_hook()


class S2Layers(Timeline):
    def construct(self):
        self.seg_layers()


class S3Measure(Timeline):
    def construct(self):
        self.seg_measure()


class S4Design(Timeline):
    def construct(self):
        self.seg_design()


class S5Status(Timeline):
    def construct(self):
        self.seg_status()


class S6Closing(Timeline):
    def construct(self):
        self.seg_closing()
