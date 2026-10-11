import registry from './layers.json'

/** One tissue layer from `shared/layers.json` (contract C1). The JSON is the source of truth. */
export interface Layer {
  id: LayerId
  label: number
  name: string
  rgb: [number, number, number]
  peel_order: number
  description: string
}

export type LayerId = 'skull' | 'gray_matter' | 'white_matter' | 'csf' | 'lesion'

/** All layers, outside in (peel order 1 = removed first). */
export const LAYERS: readonly Layer[] = (registry.layers as Layer[]).slice().sort((a, b) => a.peel_order - b.peel_order)

export const N_LAYERS = LAYERS.length

export const css = (rgb: readonly number[]) => `rgb(${rgb.join(',')})`

export const byId = (id: string): Layer => {
  const layer = LAYERS.find((l) => l.id === id)
  if (!layer) throw new Error(`Unknown layer id: ${id}`)
  return layer
}

export const byLabel: Record<number, Layer> = Object.fromEntries(LAYERS.map((l) => [l.label, l]))
