import type { LayerId } from '../layers'

/** Everything the film scenes (1–6) need about the one demo slice. */
export interface DemoData {
  /** Scanner-overlay labels; must describe what's actually shown. */
  caseLabel: string
  sequenceLabel: string
  sourceNote: [string, string]
  sliceIndex: number
  nSlices: number
  /** mm per pixel (square pixels) */
  spacingMm: number
  images: {
    /** brain-only MRI on transparent, the base plane of the title stack */
    mri: string
    /** full MRI on black */
    mriBlack: string
    mriDim: string
    /** the whole illustration */
    illus: string
    /** lesion white, other tissue dark grey */
    mask: string
    lesionOnMri: string
    layer: Record<LayerId, string>
  }
  sheet: {
    /** greyscale tiles, drawn into canvases */
    tiles: (ImageData | HTMLImageElement)[]
    sliceNumbers: number[]
    /** index of the tile for the demo slice */
    key: number
    /** slice step between tiles, for "Every 4th slice shown" */
    step: number
  }
  metrics: {
    intracranialMm2: number
    lesionAreaPx: number
    pixelAreaMm2: number
    layers: Record<LayerId, { areaMm2: number; pct: number | null; maxDiameterMm: number | null }>
  }
}
