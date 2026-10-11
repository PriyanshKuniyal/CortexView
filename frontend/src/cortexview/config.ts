/**
 * Where the film scenes (1–6) get their demo slice.
 * - 'synthetic': the built-in phantom, generated in the browser (no patient data).
 * - 'remind':    a real ReMIND slice exported by the backend into `public/demo/` (see demo/remind.ts).
 * The backend can't export that bundle yet, so this stays 'synthetic' for now.
 */
export const DEMO_SOURCE: 'synthetic' | 'remind' = 'synthetic'

/** Use the in-browser mock API for "Try it" instead of the real backend (same flag as the chat UI). */
export const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

/** Upload cap (contract C3: 300 MB). */
export const MAX_UPLOAD_MB = 300

/** Contract C3: stop polling for the illustration after 60 s and keep the flat layers. */
export const RENDER_TIMEOUT_MS = 60_000
