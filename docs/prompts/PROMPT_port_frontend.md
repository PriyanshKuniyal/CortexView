# Prompt: port the CortexView page into the Plie frontend

Save `cortexview_artifact.html` to `Plie/docs/design/cortexview_artifact.html`. Then open Claude Code in `Plie` and paste everything below the line.

---

`docs/design/cortexview_artifact.html` is a finished, self-contained prototype of the CortexView UI: one HTML file, vanilla JS, no dependencies besides Google Fonts. Port it into our frontend (the Vite + React + TS app in this repo; find it first) **so it looks and behaves the same**, but driven by our real backend instead of the built-in synthetic scan.

## What the prototype contains

It's a 7-scene scroll "film":
1. Title, with an exploded 3D layer stack
2. Slice contact sheet
3. Before/after slider
4. Outside-in peel
5. Numbers
6. Guardrails
7. "Try it": the actual app, with upload, pipeline status, peel slider, view toggle (Illustration / Flat / MRI), and clickable data pins with a measurement card

**Everything is fake right now:**
- `phantom()` generates a synthetic brain slice.
- Masks, metrics and the "painting" delay are all computed or simulated in the browser.
- Dropped files are ignored.

## Do this

1. **Read the whole HTML first**, then the repo: frontend structure, `shared/layers.json`, the FastAPI routes and Pydantic models (generate TS types from OpenAPI if that's already set up), and `docs/TASKS.md`.
2. **Port it as React components**, one per scene, plus a `useScrollFilm` hook for the scene engine (smoothed scroll playhead, scene in/out, scrubber, skip button, snapping). Keep the CSS as-is, moved into a CSS module or one stylesheet. Don't redesign anything.
3. **Layer registry.** Replace the hard-coded `LAYERS` array with `shared/layers.json` (ids, labels, names, RGB). If they disagree, the JSON wins. Tell me about any mismatch.
4. **Wire "Try it" (scene 7) to the real API:**
   - Upload → `POST /api/analyze`. Show the pipeline steps from the real response or timings, and show layers and metrics immediately.
   - Poll `GET /api/render/{id}` every second until the illustration is ready. Show the "Painting illustration" badge with the Flat view meanwhile.
   - Peel slider, view toggle, pins and measurement card all use the API's layers, metrics and pin anchors. **No numbers computed in the browser**, except formatting.
   - Errors: show a plain message in the panel. Never a blank screen.
   - Keep "Educational aid, not for diagnosis" visible on the viewer at all times.
5. **The film scenes (1–6) can keep using demo data**, but load it from a static demo bundle instead of generating it:
   - Export one real ReMIND slice from the backend (MRI PNG, per-layer PNGs, illustration, metrics JSON) into `frontend/public/demo/`.
   - If the backend can't produce that yet, keep the synthetic phantom for those scenes behind one flag, `DEMO_SOURCE = 'synthetic' | 'remind'`, and tell me.
6. **Fix copy that's now wrong**, and list every change you made:
   - The model is **FLUX.2 klein base-9B**, not 4B, and its license is the **FLUX non-commercial license**, not Apache-2.0. Scene 5 says "4B … Apache-2.0".
   - Any count in scene 5 that the repo/data doesn't support. Ask me rather than inventing a number.
   - Overlay labels like "Synth-001" and "Phantom, not a patient" must match whatever is actually shown.
7. **Keep what already works:** reduced-motion support, keyboard navigation (arrows, Page, Home/End, `S` to skip), mobile layout under 900 px, focus styles.
8. **Check it:**
   - `npm run build` passes with no TS errors.
   - Run it locally against the backend (or a mocked API if the backend isn't running) and check all 7 scenes plus a full upload → render cycle.
   - Tell me what you couldn't test.

Don't commit or push without asking me.
