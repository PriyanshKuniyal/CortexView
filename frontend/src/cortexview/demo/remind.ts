/**
 * Loads the real-slice demo bundle from `public/demo/` (DEMO_SOURCE = 'remind').
 *
 * Expected layout, exported by the backend for one ReMIND slice:
 *   demo/manifest.json   see DemoManifest below; paths are relative to demo/
 *   demo/mri.png         greyscale slice on black (the API's slice.mri_png)
 *   demo/layers/<id>.png         illustrated layer, RGBA (the API's illustrated_png)
 *   demo/layers/<id>_flat.png    flat layer, RGBA (the API's flat_png)
 *   demo/metrics.json    MetricsReport (contract C4), with pixel spacing known
 *   demo/sheet/*.png     greyscale contact-sheet slices of the same scan
 * All numbers come from metrics.json. The browser only composites images.
 */
import { LAYERS, type LayerId } from '../layers'
import type { MetricsReport } from '../api/types'
import type { DemoData } from './types'

export interface DemoManifest {
  version: 1
  case_label: string
  sequence_label: string
  source_note: [string, string]
  slice_index: number
  n_slices: number
  pixel_spacing_mm: [number, number]
  mri: string
  /** optional whole illustration; otherwise the illustrated layers are composited */
  illustration?: string
  layers: Record<LayerId, { flat: string; illustrated: string }>
  metrics: string
  sheet: { tiles: string[]; slice_numbers: number[]; key: number; step: number }
}

const BASE = `${import.meta.env.BASE_URL}demo/`

function loadImage(path: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const im = new Image()
    im.onload = () => resolve(im)
    im.onerror = () => reject(new Error(`Demo bundle: couldn't load ${path}`))
    im.src = BASE + path
  })
}

async function loadJson<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path)
  if (!res.ok) throw new Error(`Demo bundle: couldn't load ${path} (${res.status})`)
  return res.json()
}

export async function loadRemindDemo(): Promise<DemoData> {
  const man = await loadJson<DemoManifest>('manifest.json')
  const [metrics, mri, illusImg, sheetTiles, layerImgs] = await Promise.all([
    loadJson<MetricsReport>(man.metrics),
    loadImage(man.mri),
    man.illustration ? loadImage(man.illustration) : Promise.resolve(null),
    Promise.all(man.sheet.tiles.map(loadImage)),
    Promise.all(LAYERS.map(async (L) => [L.id, await loadImage(man.layers[L.id].illustrated)] as const)),
  ])
  const layer = Object.fromEntries(layerImgs) as Record<LayerId, HTMLImageElement>

  const W = mri.naturalWidth,
    H = mri.naturalHeight
  const canvas = () => {
    const c = document.createElement('canvas')
    c.width = W
    c.height = H
    return [c, c.getContext('2d')!] as const
  }
  const url = (c: HTMLCanvasElement) => c.toDataURL('image/png')
  /** the given layers, recoloured to one solid colour */
  const solid = (ids: LayerId[], color: string) => {
    const [c, g] = canvas()
    for (const id of ids) g.drawImage(layer[id], 0, 0, W, H)
    g.globalCompositeOperation = 'source-in'
    g.fillStyle = color
    g.fillRect(0, 0, W, H)
    return c
  }
  const allIds = LAYERS.map((L) => L.id)
  const otherIds = allIds.filter((id) => id !== 'lesion')

  const [mriC, mriG] = canvas()
  mriG.drawImage(mri, 0, 0)
  mriG.globalCompositeOperation = 'destination-in'
  mriG.drawImage(solid(allIds, '#fff'), 0, 0)

  const [dimC, dimG] = canvas()
  dimG.drawImage(mri, 0, 0)
  dimG.fillStyle = 'rgba(0,0,0,.72)'
  dimG.fillRect(0, 0, W, H)

  const [illC, illG] = canvas()
  if (illusImg) illG.drawImage(illusImg, 0, 0, W, H)
  else for (const id of allIds) illG.drawImage(layer[id], 0, 0, W, H)

  const [maskC, maskG] = canvas()
  maskG.fillStyle = '#000'
  maskG.fillRect(0, 0, W, H)
  maskG.drawImage(solid(otherIds, '#222'), 0, 0)
  maskG.drawImage(solid(['lesion'], '#fff'), 0, 0)

  const [lomC, lomG] = canvas()
  lomG.drawImage(mri, 0, 0)
  lomG.fillStyle = 'rgba(0,0,0,.25)'
  lomG.fillRect(0, 0, W, H)
  const ring = solid(['lesion'], '#fff')
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) lomG.drawImage(ring, dx, dy)
  lomG.drawImage(layer.lesion, 0, 0, W, H)

  const [sr, sc] = man.pixel_spacing_mm
  const layers = {} as DemoData['metrics']['layers']
  for (const L of LAYERS) {
    const m = metrics.layers[L.id]
    if (!m || m.area_mm2 == null) throw new Error(`Demo bundle: metrics for ${L.id} need a known pixel spacing`)
    layers[L.id] = { areaMm2: m.area_mm2, pct: m.pct_intracranial, maxDiameterMm: m.max_diameter_mm }
  }
  if (metrics.intracranial_area_mm2 == null) throw new Error('Demo bundle: intracranial_area_mm2 is missing')

  return {
    caseLabel: man.case_label,
    sequenceLabel: man.sequence_label,
    sourceNote: man.source_note,
    sliceIndex: man.slice_index,
    nSlices: man.n_slices,
    spacingMm: sc,
    images: {
      mri: url(mriC),
      mriBlack: BASE + man.mri,
      mriDim: url(dimC),
      illus: url(illC),
      mask: url(maskC),
      lesionOnMri: url(lomC),
      layer: Object.fromEntries(LAYERS.map((L) => [L.id, BASE + man.layers[L.id].illustrated])) as Record<LayerId, string>,
    },
    sheet: { tiles: sheetTiles, sliceNumbers: man.sheet.slice_numbers, key: man.sheet.key, step: man.sheet.step },
    metrics: {
      intracranialMm2: metrics.intracranial_area_mm2,
      lesionAreaPx: metrics.layers.lesion?.area_px ?? 0,
      pixelAreaMm2: sr * sc,
      layers,
    },
  }
}
