/**
 * Synthetic axial brain phantom, ported unchanged from docs/design/CortexView.html.
 * Used by the film scenes when DEMO_SOURCE = 'synthetic'. Never a patient.
 */
import { LAYERS, byLabel, type LayerId } from '../layers'

export type Rnd = () => number
export const mkRnd = (seed: number): Rnd => {
  let s = seed
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296
}

const vent: [number, number, number][] = []
for (const s of [-1, 1])
  for (let t = 0; t <= 1.0001; t += 1 / 36)
    vent.push([s * (0.045 + 0.12 * t * t), -0.3 + 0.6 * t, 0.028 + 0.024 * Math.sin(Math.PI * t)])

/** z = slice position, 0 = mid. Returns a label map and a 0–1 MRI image, n×n. */
export function phantom(n: number, z: number, rnd: Rnd) {
  const h = n / 2,
    LBL = new Uint8Array(n * n),
    MRI = new Float32Array(n * n)
  const sc = Math.max(0.12, Math.sqrt(Math.max(0, 1 - z * z))),
    vOn = Math.abs(z) < 0.42,
    lesS = Math.sqrt(Math.max(0, 1 - (z / 0.22) ** 2))
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const k = j * n + i,
        x = (i - h + 0.5) / h / sc,
        y = (j - h + 0.5) / h / sc
      const ax = 0.74 * (1 + 0.05 * y),
        ay = 0.9,
        nx = x / ax,
        ny = y / ay,
        r = Math.hypot(nx, ny),
        th = Math.atan2(ny, nx)
      let lab = 0,
        m = 0
      if (r < 1) {
        const rb = 0.86 + 0.006 * Math.sin(31 * th) + 0.004 * Math.sin(17 * th + 1)
        if (r >= rb) {
          lab = 1
          const q = (r - rb) / (1 - rb)
          m = q < 0.12 ? 0.14 : q < 0.52 ? (Math.abs(q - 0.32) < 0.06 ? 0.48 : 0.06) : 0.74 * (1 - (0.5 * Math.max(0, q - 0.85)) / 0.15)
        } else {
          const ph = 10 * th + 0.7 * Math.sin(3 * th) + 0.4 * Math.sin(5 * th + 2),
            sn = Math.abs(Math.sin(ph))
          const wm = 0.5 + 0.22 * Math.pow(sn, 0.55) - 0.08 * Math.exp(-(x * x) / 0.004) * (Math.abs(y) > 0.35 ? 1 : 0)
          lab = 2
          m = 0.42
          if (r < wm) {
            lab = 3
            m = 0.68
          }
          if (r > rb - 0.024) {
            lab = 4
            m = 0.08
          } else if (lab === 2 && sn < 0.075) {
            lab = 4
            m = 0.1
          }
          if (Math.abs(x) < 0.012 && Math.abs(y) > 0.36) {
            lab = 4
            m = 0.08
          }
          if (vOn && Math.abs(x) < 0.26 && Math.abs(y) < 0.42) {
            for (const [px, py, w] of vent) {
              if (Math.hypot(x - px, y - py) < w) {
                lab = 4
                m = 0.07
                break
              }
            }
          }
          if (lesS > 0) {
            const lx = x - 0.31,
              ly = y + 0.3,
              lr = Math.hypot(lx, ly),
              la = Math.atan2(ly, lx)
            const R = 0.12 * lesS * (1 + 0.13 * Math.sin(3 * la + 1) + 0.07 * Math.sin(5 * la))
            if (lr < R) {
              lab = 10
              const q = lr / R
              m = q > 0.72 ? 0.95 : 0.24 + 0.1 * q
            } else if (lr < R * 1.6 && (lab === 2 || lab === 3)) m *= 0.82
          }
        }
      }
      LBL[k] = lab
      MRI[k] = lab ? Math.max(0, m + (rnd() - 0.5) * 0.07) : rnd() * 0.02
    }
  const t = new Float32Array(MRI)
  for (let j = 1; j < n - 1; j++)
    for (let i = 1; i < n - 1; i++) {
      const k = j * n + i
      MRI[k] = (t[k] * 4 + t[k - 1] + t[k + 1] + t[k - n] + t[k + n]) / 8
    }
  return { LBL, MRI }
}

/** Distance to the nearest label boundary (chamfer). */
function chamfer(LBL: Uint8Array, N: number) {
  const D = new Float32Array(N * N)
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const k = j * N + i,
        l = LBL[k]
      D[k] =
        (i > 0 && LBL[k - 1] !== l) || (i < N - 1 && LBL[k + 1] !== l) || (j > 0 && LBL[k - N] !== l) || (j < N - 1 && LBL[k + N] !== l)
          ? 0
          : 1e9
    }
  for (let j = 1; j < N; j++)
    for (let i = 1; i < N - 1; i++) {
      const k = j * N + i
      D[k] = Math.min(D[k], D[k - 1] + 1, D[k - N] + 1, D[k - N - 1] + 1.414, D[k - N + 1] + 1.414)
    }
  for (let j = N - 2; j >= 0; j--)
    for (let i = N - 2; i > 0; i--) {
      const k = j * N + i
      D[k] = Math.min(D[k], D[k + 1] + 1, D[k + N] + 1, D[k + N + 1] + 1.414, D[k + N - 1] + 1.414)
    }
  return D
}

export const grey = (v: number) => Math.min(255, Math.pow(v, 0.9) * 270)

type Px = readonly number[] | null

export interface PhantomMetrics {
  px: number
  mm2: number
  comps: number
  cx: number
  cy: number
  /** deepest point inside the mask, in pixels */
  anchorPx: [number, number]
  diam: number | null
  hemi: 'left' | 'right'
  pct: number | null
}

/** The main 440 px phantom slice with its images and mask metrics, exactly as the prototype computed them. */
export function buildPhantomSlice() {
  const N = 440,
    H = N / 2,
    SP = 0.56 // px, mm per px
  const { LBL, MRI } = phantom(N, 0, mkRnd(7))
  const D = chamfer(LBL, N)

  const cv = document.createElement('canvas')
  cv.width = cv.height = N
  const cx = cv.getContext('2d')!
  const paint = (fn: (k: number) => Px) => {
    const im = cx.createImageData(N, N),
      d = im.data
    for (let k = 0; k < N * N; k++) {
      const c = fn(k)
      if (!c) continue
      d[k * 4] = c[0]
      d[k * 4 + 1] = c[1]
      d[k * 4 + 2] = c[2]
      d[k * 4 + 3] = c[3] ?? 255
    }
    cx.putImageData(im, 0, 0)
    return cv.toDataURL('image/png')
  }
  const illus = (k: number): Px => {
    const l = LBL[k]
    if (!l) return null
    const L = byLabel[l],
      [R, G, B] = L.rgb
    const i = k % N,
      j = (k / N) | 0,
      x = (i - H) / H,
      y = (j - H) / H,
      d = D[k]
    let s = (0.74 + 0.26 * Math.min(1, d / 7)) * (1 + 0.12 * (-x - y) * 0.6)
    if (l === 1) s *= 0.72 + 0.42 * MRI[k]
    else if (l === 10) s *= 0.85 + 0.35 * MRI[k]
    else s *= 1 + (MRI[k] - 0.4) * 0.25
    if (d < 1.1) s *= 0.6
    return [Math.min(255, R * s), Math.min(255, G * s), Math.min(255, B * s)]
  }

  const IMG = {
    mri: paint((k) => {
      const v = grey(MRI[k])
      return LBL[k] ? [v, v, v] : null
    }),
    mriBlack: paint((k) => {
      const v = grey(MRI[k])
      return [v, v, v]
    }),
    mriDim: paint((k) => {
      const v = grey(MRI[k]) * 0.28
      return [v, v, v]
    }),
    illus: paint(illus),
    mask: paint((k) => (LBL[k] === 10 ? [255, 255, 255] : LBL[k] ? [34, 34, 34] : [0, 0, 0])),
    lesionOnMri: paint((k) => {
      if (LBL[k] === 10) return D[k] < 1.5 ? [255, 255, 255] : illus(k)
      const v = grey(MRI[k]) * 0.75
      return [v, v, v]
    }),
    layer: {} as Record<LayerId, string>,
    flatLayer: {} as Record<LayerId, string>,
  }
  for (const L of LAYERS) {
    IMG.layer[L.id] = paint((k) => (LBL[k] === L.label ? illus(k) : null))
    IMG.flatLayer[L.id] = paint((k) => (LBL[k] === L.label ? L.rgb : null))
  }

  // metrics (math on masks)
  const M = {} as Record<LayerId, PhantomMetrics>
  let intracranial = 0
  for (const L of LAYERS) {
    let n = 0,
      sx = 0,
      sy = 0,
      best = -1,
      ax = 0,
      ay = 0
    const pts: [number, number][] = []
    for (let k = 0; k < N * N; k++)
      if (LBL[k] === L.label) {
        n++
        const i = k % N,
          j = (k / N) | 0
        sx += i
        sy += j
        if (D[k] > best) {
          best = D[k]
          ax = i
          ay = j
        }
        if (L.id === 'lesion' && D[k] === 0) pts.push([i, j])
      }
    const seen = new Uint8Array(N * N)
    let comps = 0
    for (let k = 0; k < N * N; k++)
      if (LBL[k] === L.label && !seen[k]) {
        comps++
        const st = [k]
        seen[k] = 1
        while (st.length) {
          const p = st.pop()!,
            i = p % N
          for (const q of [p - 1, p + 1, p - N, p + N]) {
            if (q < 0 || q >= N * N) continue
            if ((q === p - 1 && i === 0) || (q === p + 1 && i === N - 1)) continue
            if (!seen[q] && LBL[q] === L.label) {
              seen[q] = 1
              st.push(q)
            }
          }
        }
      }
    let diam: number | null = null
    if (pts.length) {
      let m = 0
      for (let a = 0; a < pts.length; a++)
        for (let b = a + 1; b < pts.length; b++) {
          const dd = (pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2
          if (dd > m) m = dd
        }
      diam = Math.sqrt(m) * SP
    }
    M[L.id] = { px: n, mm2: n * SP * SP, comps, cx: sx / n, cy: sy / n, anchorPx: [ax, ay], diam, hemi: sx / n > H ? 'left' : 'right', pct: null }
    if (L.id !== 'skull') intracranial += n
  }
  for (const L of LAYERS) M[L.id].pct = L.id === 'skull' ? null : (M[L.id].px / intracranial) * 100

  return { N, SP, IMG, M, intracranialPx: intracranial }
}

/** Contact-sheet tiles: every 4th of 192 phantom slices, 96 px each, as greyscale ImageData. */
export function phantomSheet() {
  const TN = 96,
    NT = 48
  const tiles: ImageData[] = [],
    sliceNumbers: number[] = []
  for (let t = 0; t < NT; t++) {
    const sl = (t + 1) * 4,
      z = sl / 96 - 1,
      { MRI: mr } = phantom(TN, z, mkRnd(100 + t))
    const im = new ImageData(TN, TN)
    for (let k = 0; k < TN * TN; k++) {
      const v = grey(mr[k])
      im.data[k * 4] = im.data[k * 4 + 1] = im.data[k * 4 + 2] = v
      im.data[k * 4 + 3] = 255
    }
    tiles.push(im)
    sliceNumbers.push(sl)
  }
  return { tiles, sliceNumbers, key: 23 }
}
