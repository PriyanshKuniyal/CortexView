/**
 * API types, hand-mirrored from docs/contracts.md (C3 API, C4 metrics, DRAFT v0).
 * There is no backend or OpenAPI schema in the repo yet; regenerate these from OpenAPI once there is.
 */
import type { LayerId } from '../layers'

export type Confidence = 'high' | 'medium' | 'low' | 'empty'

export interface SliceInfo {
  width: number
  height: number
  mri_png: string
  slice_index: number
  n_slices: number
  plane: string
  /** [row, col] in mm; null if unknown */
  pixel_spacing_mm: [number, number] | null
  sequence: 'T1_pre' | 'T1_post' | 'T2' | 'FLAIR' | 'unknown'
  sequence_source: string
  patient_left: string | null
}

export interface LayerResult {
  id: LayerId
  source: 'auto' | 'doctor'
  confidence: Confidence
  flat_png: string
  illustrated_png: string | null
}

export interface LayerMetrics {
  present: boolean
  area_px: number
  area_mm2: number | null
  pct_intracranial: number | null
  n_components: number
  centroid_px: [number, number] | null
  bbox_px: [number, number, number, number] | null
  max_diameter_mm: number | null
  volume_ml: number | null
  /** null when the layer is empty; ratio_to_brain is null for skull */
  intensity: { mean: number; std: number; ratio_to_brain: number | null } | null
  hemisphere: 'left' | 'right' | 'both' | 'midline' | null
  /** [x, y] in slice pixels; null when the layer is empty */
  anchor_px: [number, number] | null
  confidence: Confidence
  warnings: string[]
}

export interface MetricsReport {
  intracranial_area_mm2: number | null
  layers: Partial<Record<LayerId, LayerMetrics>>
  text: { layer: LayerId; template_id: string; text: string }[]
}

export interface ApiWarning {
  code: string
  message: string
}

export interface RenderRef {
  status: 'pending' | 'done' | 'failed'
  poll_url?: string
  poll_interval_ms?: number
}

export interface AnalyzeResponse {
  job_id: string
  disclaimer: string
  slice: SliceInfo
  layers: LayerResult[]
  metrics: MetricsReport
  render: RenderRef
  warnings: ApiWarning[]
}

export type RenderResponse =
  | { status: 'pending' }
  | {
      status: 'done'
      backend: string
      fidelity?: { mean_iou: number; phantom_lesion_pct: number; passed: boolean }
      layers: { id: LayerId; illustrated_png: string }[]
    }
  | { status: 'failed'; reason: string }

export interface ApiErrorBody {
  error: { code: string; message: string }
}
