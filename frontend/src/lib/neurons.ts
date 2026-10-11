/**
 * Procedural ASCII neurons. A seeded generator draws one large neuron and
 * two small partners, wires them with curved axons, grows branching
 * dendrites, and sprinkles microscopic debris — so the composition is
 * organic but identical on every load.
 */

export const COLS = 90
export const ROWS = 26
export const ASPECT = 2.1 // a monospace cell is ~2x taller than it is wide

export interface Cell {
  x: number
  y: number
}

export interface Soma {
  x: number
  y: number
  cells: Cell[]
}

export interface Edge {
  from: number
  to: number
  /** visible cells along the axon, ordered from -> to */
  path: Cell[]
}

export interface NeuronField {
  structure: string[][]
  dust: string[][]
  somas: Soma[]
  edges: Edge[]
}

const DUST = ['.', '.', '.', '·', ':', "'"]

function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const emptyGrid = () => Array.from({ length: ROWS }, () => Array<string>(COLS).fill(' '))

export const gridToText = (g: string[][]) => g.map((row) => row.join('')).join('\n')

/** Pick the stroke glyph that best matches a direction (in cell units). */
function strokeChar(dx: number, dy: number) {
  const a = (((Math.atan2(dy * ASPECT, dx) * 180) / Math.PI) + 180) % 180
  if (a < 22.5 || a >= 157.5) return '-'
  if (a < 67.5) return '\\'
  if (a < 112.5) return '|'
  return '/'
}

export function buildField(seed = 11): NeuronField {
  const rand = mulberry32(seed)
  const structure = emptyGrid()
  const dust = emptyGrid()

  const cx = (COLS - 1) / 2
  const cy = (ROWS - 1) / 2
  const radius = (x: number, y: number) => Math.hypot((x - cx) / (COLS / 2 - 1), (y - cy) / (ROWS / 2 - 0.5))
  const inBounds = (x: number, y: number) => x >= 0 && y >= 0 && x < COLS && y < ROWS
  const put = (g: string[][], x: number, y: number, ch: string, overwrite = false) => {
    if (!inBounds(x, y) || (!overwrite && g[y][x] !== ' ')) return
    g[y][x] = ch
  }

  // 1. Composition: one large neuron at the centre, two small partners
  const centers: (Cell & { big: boolean })[] = [
    { x: Math.round(COLS * 0.5), y: Math.round(ROWS * 0.46), big: true },
    { x: Math.round(COLS * 0.14), y: Math.round(ROWS * 0.72), big: false },
    { x: Math.round(COLS * 0.86), y: Math.round(ROWS * 0.26), big: false },
  ]

  // 2. Axons: gentle quadratic curves from the large neuron to each partner
  const curve = (a: Cell, b: Cell): Cell[] => {
    const dx = b.x - a.x
    const dy = (b.y - a.y) * ASPECT
    const len = Math.hypot(dx, dy)
    const bend = (rand() < 0.5 ? -1 : 1) * len * (0.12 + rand() * 0.1)
    const mx = (a.x + b.x) / 2 + (-dy / len) * bend
    const my = (a.y + b.y) / 2 + ((dx / len) * bend) / ASPECT
    const steps = Math.ceil(len * 2.5)
    const cells: Cell[] = []
    for (let s = 0; s <= steps; s++) {
      const t = s / steps
      const u = 1 - t
      const x = Math.round(u * u * a.x + 2 * u * t * mx + t * t * b.x)
      const y = Math.round(u * u * a.y + 2 * u * t * my + t * t * b.y)
      const last = cells.at(-1)
      if (!last || last.x !== x || last.y !== y) cells.push({ x, y })
    }
    return cells
  }

  const edges: Edge[] = centers.slice(1).map((c, i) => ({ from: 0, to: i + 1, path: curve(centers[0], c) }))

  for (const { path } of edges) {
    path.forEach((c, k) => {
      const prev = path[Math.max(0, k - 1)]
      const next = path[Math.min(path.length - 1, k + 1)]
      put(structure, c.x, c.y, strokeChar(next.x - prev.x, next.y - prev.y))
    })
    // synaptic bouton midway along the axon
    const b = path[Math.floor(path.length * 0.55)]
    put(structure, b.x, b.y, 'o', true)
  }

  // 3. Dendrites: wandering branches that fork and end in spines
  const branch = (x: number, y: number, angle: number, length: number, depth: number) => {
    let px = x
    let py = y
    for (let s = 0; s < length; s++) {
      angle += (rand() - 0.5) * 0.5
      const nx = px + Math.cos(angle)
      const ny = py + Math.sin(angle) / ASPECT
      const gx = Math.round(nx)
      const gy = Math.round(ny)
      if (gx !== Math.round(px) || gy !== Math.round(py)) {
        if (!inBounds(gx, gy) || radius(gx, gy) > 1.02) return
        put(structure, gx, gy, strokeChar(Math.cos(angle), Math.sin(angle) / ASPECT))
      }
      px = nx
      py = ny
    }
    if (depth > 0 && rand() < 0.7) {
      branch(px, py, angle - 0.55, 2 + Math.floor(rand() * 3), depth - 1)
      branch(px, py, angle + 0.55, 2 + Math.floor(rand() * 3), depth - 1)
    } else {
      put(structure, Math.round(px + Math.sign(Math.cos(angle))), Math.round(py), rand() < 0.3 ? '*' : '.')
    }
  }

  for (const c of centers) {
    const n = c.big ? 6 : 3
    const base = rand() * Math.PI * 2
    for (let k = 0; k < n; k++) {
      const angle = base + (k * Math.PI * 2) / n + (rand() - 0.5) * 0.4
      branch(c.x, c.y, angle, c.big ? 9 + Math.floor(rand() * 4) : 5 + Math.floor(rand() * 2), c.big ? 2 : 1)
    }
  }

  // Shallow diagonals land as runs like "\\\" or "///"; redraw them as
  // staircases ("\__" and "__/") the way hand-made ASCII art does.
  for (const row of structure) {
    for (let x = 0; x < COLS; x++) {
      const ch = row[x]
      if (ch !== '\\' && ch !== '/') continue
      let end = x
      while (end + 1 < COLS && row[end + 1] === ch) end++
      for (let k = x; k <= end && end > x; k++) {
        const keep = ch === '\\' ? k === x : k === end
        if (!keep) row[k] = '_'
      }
      x = end
    }
  }

  // 4. Somas, drawn last so they sit cleanly on top of their processes
  const somaKeys = new Set<string>()
  const somas: Soma[] = centers.map(({ x, y, big }) => {
    const shape = big
      ? ['   .---.   ', " .'     '. ", '(   (@)   )', " '.     .' ", "   '---'   "]
      : [' .-. ', '( o )', " `-' "]
    const cells: Cell[] = []
    shape.forEach((row, r) => {
      const first = row.search(/\S/)
      const last = row.length - 1 - [...row].reverse().join('').search(/\S/)
      ;[...row].forEach((ch, k) => {
        if (k < first || k > last) return // keep the rounded outline's outside transparent
        const gx = x - Math.floor(row.length / 2) + k
        const gy = y - Math.floor(shape.length / 2) + r
        if (!inBounds(gx, gy)) return
        put(structure, gx, gy, ch, true)
        if (ch !== ' ') cells.push({ x: gx, y: gy })
        somaKeys.add(`${gx},${gy}`)
      })
    })
    return { x, y, cells }
  })

  // Pulses travel only over visible axon glyphs
  for (const e of edges) {
    e.path = e.path.filter((c) => !somaKeys.has(`${c.x},${c.y}`) && structure[c.y][c.x] !== ' ')
  }

  // 5. Microscopic debris, densest toward the centre of the field
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      if (structure[y][x] !== ' ') continue
      const r = radius(x, y)
      if (r < 1.05 && rand() < 0.008 * (1.1 - r)) dust[y][x] = DUST[Math.floor(rand() * DUST.length)]
    }
  }

  return { structure, dust, somas, edges: edges.filter((e) => e.path.length > 1) }
}
