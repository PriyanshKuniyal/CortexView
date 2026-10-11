/** Display formatting only. No measurements are computed here. */
import type { SliceInfo } from './api/types'

export const numberWord = (n: number) =>
  ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][n] ?? String(n)

export const pad2 = (n: number) => String(n).padStart(2, '0')

const PLANE: Record<string, string> = { axial: 'Ax', coronal: 'Cor', sagittal: 'Sag' }
const SEQUENCE: Record<SliceInfo['sequence'], string> = {
  T1_post: 'T1 + C',
  T1_pre: 'T1',
  T2: 'T2',
  FLAIR: 'FLAIR',
  unknown: 'MR',
}

/** e.g. "Ax T1 + C", as on the prototype's scanner overlay */
export const sequenceLabel = (s: SliceInfo) => `${PLANE[s.plane] ?? s.plane} ${SEQUENCE[s.sequence] ?? s.sequence}`

export function spacingLabel(spacing: [number, number] | null) {
  if (!spacing) return 'Spacing unknown'
  const [r, c] = spacing
  return Math.abs(r - c) < 1e-3 ? `${c.toFixed(2)} mm/px` : `${r.toFixed(2)} × ${c.toFixed(2)} mm/px`
}

export const describeFiles = (files: File[]) => (files.length === 1 ? `“${files[0].name}”` : `${files.length} files`)
