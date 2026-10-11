import { LAYERS, type LayerId } from '../layers'
import { buildPhantomSlice, phantomSheet } from './phantom'
import type { DemoData } from './types'

/** The prototype's built-in phantom as demo data. Computed in the browser; it's not a patient. */
export function buildSyntheticDemo(): DemoData {
  const { SP, IMG, M, intracranialPx } = buildPhantomSlice()
  const sheet = phantomSheet()
  const layers = {} as DemoData['metrics']['layers']
  for (const L of LAYERS) layers[L.id as LayerId] = { areaMm2: M[L.id].mm2, pct: M[L.id].pct, maxDiameterMm: M[L.id].diam }
  return {
    caseLabel: 'Synth-001',
    sequenceLabel: 'Ax T1 + C',
    sourceNote: ['Phantom', 'not a patient'],
    sliceIndex: 96,
    nSlices: 192,
    spacingMm: SP,
    images: {
      mri: IMG.mri,
      mriBlack: IMG.mriBlack,
      mriDim: IMG.mriDim,
      illus: IMG.illus,
      mask: IMG.mask,
      lesionOnMri: IMG.lesionOnMri,
      layer: IMG.layer,
    },
    sheet: { tiles: sheet.tiles, sliceNumbers: sheet.sliceNumbers, key: sheet.key, step: 4 },
    metrics: { intracranialMm2: intracranialPx * SP * SP, lesionAreaPx: M.lesion.px, pixelAreaMm2: SP * SP, layers },
  }
}
