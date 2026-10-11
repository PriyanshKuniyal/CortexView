/**
 * Where the film scenes (1–6) get their demo slice.
 * - 'synthetic': the built-in phantom, generated in the browser (no patient data).
 * - 'remind':    a real ReMIND slice exported by the backend into `public/demo/` (see demo/remind.ts).
 * The backend can't export that bundle yet, so this stays 'synthetic' for now.
 */
export const DEMO_SOURCE: 'synthetic' | 'remind' = 'synthetic'

/** Give up on POST /api/analyze after this long (a whole scan folder usually takes 10–30 s). */
export const ANALYZE_TIMEOUT_MS = 180_000

/** Contract C3: stop polling for the illustration after 60 s and keep the flat layers. */
export const RENDER_TIMEOUT_MS = 60_000
