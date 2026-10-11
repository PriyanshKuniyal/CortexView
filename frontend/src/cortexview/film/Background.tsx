import { useEffect, useRef } from 'react'
import { T, clamp, ease, type Film } from './useScrollFilm'

/** Moving greyscale light, one mood per scene, blended across transitions. */
const MOOD = [
  [[0.18, 0.2], [0.86, 0.3], [0.6, 0.92], [0.1, 0.9]],
  [[0.5, 0.05], [0.5, 0.98], [0.05, 0.5], [0.95, 0.5]],
  [[0.95, 0.15], [0.75, 0.9], [0.1, 0.15], [0.3, 0.85]],
  [[0.12, 0.2], [0.25, 0.85], [0.9, 0.1], [0.95, 0.85]],
  [[0.85, 0.12], [0.15, 0.95], [0.5, 0.5], [0.95, 0.75]],
  [[0.05, 0.3], [0.95, 0.7], [0.4, 0.05], [0.6, 0.95]],
  [[0.9, 0.1], [0.08, 0.92], [0.55, 0.98], [0.95, 0.55]],
]
const SRC = [
  [255, 255, 255, 0.95],
  [150, 150, 152, 0.3],
  [255, 255, 255, 0.85],
  [120, 120, 122, 0.22],
]

export function Background({ film }: { film: Film }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const { starts, last, addFrameListener } = film

  useEffect(() => {
    const bgc = ref.current!
    const g = bgc.getContext('2d')!
    return addFrameListener((ps, now) => {
      const W = bgc.width,
        Hh = bgc.height
      g.clearRect(0, 0, W, Hh)
      let i = 0
      for (; i < last; i++) if (ps < starts[i + 1]) break
      const f = i < last ? ease(clamp((ps - (starts[i + 1] - T)) / T)) : 0,
        a = MOOD[i],
        b = MOOD[Math.min(last, i + 1)]
      const t = now / 1000
      SRC.forEach(([r, gg, bb, al], k) => {
        const x = (a[k][0] + (b[k][0] - a[k][0]) * f + 0.05 * Math.sin(t * 0.21 + k * 1.7)) * W
        const y = (a[k][1] + (b[k][1] - a[k][1]) * f + 0.06 * Math.cos(t * 0.17 + k * 2.3)) * Hh
        const rad = (0.42 + 0.08 * Math.sin(t * 0.3 + k)) * W
        g.save()
        g.translate(x, y)
        g.rotate(t * 0.05 + k)
        g.scale(1, 0.62)
        const gr = g.createRadialGradient(0, 0, 0, 0, 0, rad)
        gr.addColorStop(0, `rgba(${r},${gg},${bb},${al})`)
        gr.addColorStop(1, `rgba(${r},${gg},${bb},0)`)
        g.fillStyle = gr
        g.fillRect(-rad, -rad, rad * 2, rad * 2)
        g.restore()
      })
    })
  }, [starts, last, addFrameListener])

  return (
    <div id="bg" aria-hidden="true">
      <canvas ref={ref} width={160} height={100} />
    </div>
  )
}
