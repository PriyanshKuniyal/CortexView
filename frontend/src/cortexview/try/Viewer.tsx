import { useEffect, useRef, useState } from 'react'
import { LAYERS, css, type Layer, type LayerId } from '../layers'
import type { AnalyzeResponse, LayerMetrics } from '../api/types'
import { fmt } from './compare'

export type View = '3d' | 'slice' | 'mri'

/** The headline number for a layer's label. Formatting only; the numbers come from the metrics engine. */
export function headline(L: Layer, m: LayerMetrics | undefined): string {
  if (!m || !m.present) return 'not on this slice'
  const area = m.area_mm2 != null ? `${fmt(m.area_mm2 / 100)} cm²` : `${m.area_px.toLocaleString()} px`
  if (L.id === 'lesion') return m.max_diameter_mm != null ? `${area} · ${m.max_diameter_mm.toFixed(0)} mm across` : area
  if (m.pct_intracranial != null) return `${m.pct_intracranial.toFixed(0)}% of the brain space`
  return area
}

interface Props {
  res: AnalyzeResponse
  illustrated: Partial<Record<LayerId, string>>
  painting: boolean
  view: View
  peel: number // layers with peel_order <= peel are lifted off
  sel: LayerId | null
  hover: LayerId | null
  onSelect: (id: LayerId) => void
  onHover: (id: LayerId | null) => void
}

/** Where the slice sits inside a square box, keeping its aspect ratio. */
function fit(w: number, h: number) {
  const s = Math.max(w, h)
  return { x: (s - w) / 2 / s, y: (s - h) / 2 / s, w: w / s, h: h / s }
}

export function Viewer(props: Props) {
  return props.view === '3d' ? <Stack3D {...props} /> : <SliceView {...props} />
}

// ------------------------------------------------------------------------------------------ 3D

function Stack3D({ res, illustrated, painting, peel, sel, hover, onSelect, onHover }: Props) {
  const [rot, setRot] = useState({ rx: 56, rz: -32 })
  const [open, setOpen] = useState(false)
  const drag = useRef<{ x: number; y: number; rx: number; rz: number } | null>(null)
  const [dragging, setDragging] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const dots = useRef<Partial<Record<LayerId, HTMLSpanElement | null>>>({})
  const [pts, setPts] = useState<Partial<Record<LayerId, { x: number; y: number }>>>({})
  const [size, setSize] = useState({ w: 0, h: 0 })

  // Assemble, then fan the layers out, so the user sees one slice become five.
  useEffect(() => {
    setOpen(false)
    const t = setTimeout(() => setOpen(true), 350)
    return () => clearTimeout(t)
  }, [res.job_id])

  // Where each layer's anchor dot actually lands on screen (after 3D transforms). Measured every frame
  // while things move, so the label column and leader lines follow the layers.
  useEffect(() => {
    let raf = 0
    const until = performance.now() + 1600
    const measure = () => {
      const el = root.current
      if (!el) return
      const b = el.getBoundingClientRect()
      const next: Partial<Record<LayerId, { x: number; y: number }>> = {}
      for (const L of LAYERS) {
        const d = dots.current[L.id]
        if (!d) continue
        const r = d.getBoundingClientRect()
        next[L.id] = { x: r.left + r.width / 2 - b.left, y: r.top + r.height / 2 - b.top }
      }
      setPts(next)
      setSize({ w: b.width, h: b.height })
      if (performance.now() < until || drag.current) raf = requestAnimationFrame(measure)
    }
    raf = requestAnimationFrame(measure)
    const ro = new ResizeObserver(() => requestAnimationFrame(measure))
    if (root.current) ro.observe(root.current)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [open, rot, peel, dragging, res.job_id])

  const { slice } = res
  const box = fit(slice.width, slice.height)
  // bottom to top: the MRI, then the layers from the inside out, so the skull sits on top
  const planes = [...LAYERS].reverse()

  // label column on the right, in screen order, spread so labels never overlap
  const LABEL_H = 46
  const labelX = size.w * 0.74
  const items = LAYERS.flatMap((L) => {
    const m = res.metrics.layers[L.id]
    const p = pts[L.id]
    if (!open || !m?.present || !m.anchor_px || !p || L.peel_order <= peel) return []
    return [{ L, m, p, y: p.y }]
  }).sort((a, b) => a.p.y - b.p.y)
  let y = LABEL_H / 2
  for (const it of items) it.y = y = Math.max(y, Math.min(size.h - LABEL_H / 2, it.p.y))
  for (let i = items.length - 1, lim = size.h - LABEL_H / 2; i >= 0; i--, lim -= LABEL_H) items[i].y = Math.min(items[i].y, lim)
  for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + LABEL_H)

  return (
    <div
      ref={root}
      className={'x3d' + (dragging ? ' dragging' : '')}
      onPointerDown={(e) => {
        if ((e.target as HTMLElement).closest('button')) return
        drag.current = { x: e.clientX, y: e.clientY, ...rot }
        setDragging(true)
        ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
      }}
      onPointerMove={(e) => {
        const d = drag.current
        if (!d) return
        setRot({ rz: d.rz + (e.clientX - d.x) * 0.35, rx: Math.min(80, Math.max(0, d.rx - (e.clientY - d.y) * 0.25)) })
      }}
      onPointerUp={() => {
        drag.current = null
        setDragging(false)
      }}
    >
      <div className="x3d-stack" style={{ transform: `rotateX(${rot.rx}deg) rotateZ(${rot.rz}deg)` }}>
        <div className="xp" style={{ '--i': 0 } as React.CSSProperties}>
          <div className="xp-img" style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%` }}>
            <img src={slice.mri_png} alt="MRI slice" draggable={false} />
          </div>
        </div>
        {planes.map((L, k) => {
          const lr = res.layers.find((l) => l.id === L.id)
          if (!lr) return null
          const m = res.metrics.layers[L.id]
          const src = illustrated[L.id] ?? lr.flat_png
          return (
            <div
              key={L.id}
              className={'xp' + (L.peel_order <= peel ? ' lifted' : '') + (hover && hover !== L.id ? ' dim' : '')}
              style={{ '--i': open ? k + 1 : (k + 1) * 0.04 } as React.CSSProperties}
            >
              <div className="xp-img" style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%` }}>
                <img src={src} alt={`${L.name} layer`} draggable={false} />
                {m?.present && m.anchor_px && (
                  <span
                    ref={(el) => void (dots.current[L.id] = el)}
                    className="xd"
                    style={{ left: `${(m.anchor_px[0] / slice.width) * 100}%`, top: `${(m.anchor_px[1] / slice.height) * 100}%` }}
                  />
                )}
              </div>
            </div>
          )
        })}
      </div>

      <svg className="x3d-lines" width={size.w} height={size.h} aria-hidden="true">
        {items.map((it) => (
          <polyline
            key={it.L.id}
            className={(hover ?? sel) === it.L.id ? 'on' : ''}
            points={`${it.p.x},${it.p.y} ${labelX - 22},${it.y} ${labelX - 6},${it.y}`}
          />
        ))}
      </svg>
      {items.map((it) => (
        <span key={it.L.id} className={'x3d-dot' + (it.L.id === 'lesion' ? ' hot' : '')} style={{ left: it.p.x, top: it.p.y, background: css(it.L.rgb) }} />
      ))}
      {items.map((it) => (
        <button
          key={it.L.id}
          type="button"
          className={'xl-card' + (it.L.id === 'lesion' ? ' hot' : '') + ((hover ?? sel) === it.L.id ? ' on' : '')}
          style={{ left: labelX, top: it.y }}
          onClick={() => onSelect(it.L.id)}
          onMouseEnter={() => onHover(it.L.id)}
          onMouseLeave={() => onHover(null)}
        >
          <i style={{ background: css(it.L.rgb) }} />
          <span>
            <b>{it.L.name}</b>
            <em>{headline(it.L, it.m)}</em>
          </span>
        </button>
      ))}
      {painting && <div className="x3d-paint mono">Painting the illustration…</div>}
      <div className="x3d-hint mono">Drag to rotate · hover a label to isolate its layer</div>
    </div>
  )
}

// ------------------------------------------------------------------------------------------ 2D

const FRAME_X = 0.21 // the frame's left edge, as a share of the viewer width (labels live outside it)
const FRAME_W = 0.58

function SliceView({ res, illustrated, painting, view, peel, sel, hover, onSelect, onHover }: Props) {
  const { slice } = res
  const box = fit(slice.width, slice.height)
  const focus = hover ?? sel
  const raw = view === 'mri'

  // Callouts: anchor in viewer coordinates (0..1), label on the nearer side, spread so they don't overlap.
  const items = LAYERS.flatMap((L) => {
    const m = res.metrics.layers[L.id]
    if (!m?.present || !m.anchor_px || L.peel_order <= peel || raw) return []
    const fx = box.x + (m.anchor_px[0] / slice.width) * box.w
    const fy = box.y + (m.anchor_px[1] / slice.height) * box.h
    return [{ L, m, ax: FRAME_X + fx * FRAME_W, ay: fy, side: (fx < 0.5 ? 'l' : 'r') as 'l' | 'r', ly: fy }]
  })
  for (const side of ['l', 'r'] as const) {
    const col = items.filter((c) => c.side === side).sort((a, b) => a.ay - b.ay)
    let y = 0.07
    for (const c of col) {
      c.ly = y = Math.max(y, Math.min(0.93, c.ay))
      y += 0.16
    }
    const over = (col.at(-1)?.ly ?? 0) - 0.93 // ran off the bottom: shift the column up
    if (over > 0) col.forEach((c) => (c.ly -= over))
  }
  const placed = items

  // L/R markers: patient's left/right on the image edges
  const leftIsPatient = slice.patient_left === 'image_left'

  return (
    <div className="sv">
      <div className="sv-frame" style={{ left: `${FRAME_X * 100}%`, width: `${FRAME_W * 100}%` }}>
        <div style={{ position: 'absolute', left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%` }}>
          <img src={slice.mri_png} alt="MRI slice" />
          {LAYERS.map((L) => {
            const lr = res.layers.find((l) => l.id === L.id)
            if (!lr) return null
            const src = view === 'slice' && illustrated[L.id] ? illustrated[L.id] : lr.flat_png
            const off = raw || L.peel_order <= peel
            return (
              <img
                key={L.id}
                src={src}
                alt={`${L.name} layer`}
                className={'sv-layer' + (off ? ' off' : '') + (hover && hover !== L.id ? ' dim' : '')}
              />
            )
          })}
        </div>
        {slice.patient_left && (
          <>
            <span className="sv-side l mono">{leftIsPatient ? 'L' : 'R'}</span>
            <span className="sv-side r mono">{leftIsPatient ? 'R' : 'L'}</span>
          </>
        )}
        {painting && !raw && <div className="sv-paint mono">Painting the illustration…</div>}
      </div>
      <svg className="sv-lines" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {placed.map((c) => {
          const ex = c.side === 'l' ? FRAME_X * 100 - 1.5 : (FRAME_X + FRAME_W) * 100 + 1.5
          return (
            <polyline
              key={c.L.id}
              className={focus === c.L.id ? 'on' : ''}
              points={`${c.ax * 100},${c.ay * 100} ${ex},${c.ly * 100} ${c.side === 'l' ? ex - 2 : ex + 2},${c.ly * 100}`}
            />
          )
        })}
      </svg>
      {placed.map((c) => (
        <span key={c.L.id} className={'sv-dot' + (c.L.id === 'lesion' ? ' hot' : '')} style={{ left: `${c.ax * 100}%`, top: `${c.ay * 100}%`, background: css(c.L.rgb) }} />
      ))}
      {placed.map((c) => (
        <button
          key={c.L.id}
          type="button"
          className={'sv-label ' + c.side + (focus === c.L.id ? ' on' : '') + (c.L.id === 'lesion' ? ' hot' : '')}
          style={{ top: `${c.ly * 100}%`, [c.side === 'l' ? 'right' : 'left']: `${(c.side === 'l' ? 1 - FRAME_X : FRAME_X + FRAME_W) * 100 + 3.5}%` }}
          onClick={() => onSelect(c.L.id)}
          onMouseEnter={() => onHover(c.L.id)}
          onMouseLeave={() => onHover(null)}
        >
          <i style={{ background: css(c.L.rgb) }} />
          <b>{c.L.name}</b>
          <em>{headline(c.L, c.m)}</em>
        </button>
      ))}
    </div>
  )
}
