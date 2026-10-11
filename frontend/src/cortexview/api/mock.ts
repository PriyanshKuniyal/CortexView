/**
 * Mock of the CortexView backend (VITE_USE_MOCK=true), for working on the UI without a server.
 * It plays the server's part: segments and measures the built-in phantom and answers in the
 * contract C3/C4 shapes. Uploaded files are not read.
 *
 * Test hooks, by file name:
 *   anything that isn't .dcm or .zip  -> 422 not_dicom
 *   a name containing "render-fail"  -> the render job fails, flat layers stay
 */
import { LAYERS } from '../layers'
import { buildPhantomSlice } from '../demo/phantom'
import { ApiError } from './client'
import type { AnalyzeResponse, LayerMetrics, MetricsReport, RenderResponse } from './types'

const ANALYZE_MS = 1150
const RENDER_MS = 2600

let slice: ReturnType<typeof buildPhantomSlice> | null = null
const jobs = new Map<string, { readyAt: number; fail: boolean }>()

function sleep(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => {
      clearTimeout(t)
      reject(new DOMException('Aborted', 'AbortError'))
    })
  })
}

export async function mockAnalyze(files: File[], signal?: AbortSignal): Promise<AnalyzeResponse> {
  await sleep(ANALYZE_MS, signal)
  if (files.some((f) => !/\.(dcm|zip)$/i.test(f.name)))
    throw new ApiError('not_dicom', "One of the files isn't a DICOM file or a .zip.", 422)

  slice ??= buildPhantomSlice()
  const { N, SP, IMG, M, intracranialPx } = slice
  const job_id = crypto.randomUUID()
  jobs.set(job_id, { readyAt: Date.now() + RENDER_MS, fail: files.some((f) => /render-fail/i.test(f.name)) })

  const layers: MetricsReport['layers'] = {}
  for (const L of LAYERS) {
    const m = M[L.id]
    const lm: LayerMetrics = {
      present: m.px > 0,
      area_px: m.px,
      area_mm2: m.mm2,
      pct_intracranial: m.pct,
      n_components: m.comps,
      centroid_px: [m.cx, m.cy],
      bbox_px: null,
      max_diameter_mm: m.diam,
      volume_ml: null,
      intensity: null,
      hemisphere: L.id === 'lesion' ? m.hemi : null,
      anchor_px: m.anchorPx,
      confidence: 'high',
      warnings: [],
    }
    layers[L.id] = lm
  }

  return {
    job_id,
    disclaimer: 'Educational visualization. Not a diagnostic device.',
    slice: {
      width: N,
      height: N,
      mri_png: IMG.mriBlack,
      slice_index: 95, // 0-based, shown as SL 96
      n_slices: 192,
      plane: 'axial',
      pixel_spacing_mm: [SP, SP],
      sequence: 'T1_post',
      sequence_source: 'header',
      patient_left: 'image_right',
    },
    layers: LAYERS.map((L) => ({ id: L.id, source: 'auto', confidence: 'high', flat_png: IMG.flatLayer[L.id], illustrated_png: null })),
    metrics: { intracranial_area_mm2: intracranialPx * SP * SP, layers, text: [] },
    render: { status: 'pending', poll_url: `/api/render/${job_id}`, poll_interval_ms: 1000 },
    warnings: [],
  }
}

export async function mockRender(url: string, signal?: AbortSignal): Promise<RenderResponse> {
  await sleep(60, signal)
  const id = url.split('/').pop()!
  const job = jobs.get(id)
  if (!job || !slice) throw new ApiError('not_found', 'This result has expired. Upload the scan again.', 404)
  if (Date.now() < job.readyAt) return { status: 'pending' }
  jobs.delete(id) // fetched once, then gone (contract C3)
  if (job.fail) return { status: 'failed', reason: 'timeout' }
  const { IMG } = slice
  return {
    status: 'done',
    backend: 'mock',
    layers: LAYERS.map((L) => ({ id: L.id, illustrated_png: IMG.layer[L.id] })),
  }
}
