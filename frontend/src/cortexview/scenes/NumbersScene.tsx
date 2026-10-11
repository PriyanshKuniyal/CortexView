import { useEffect, useRef } from 'react'
import { LAYERS, N_LAYERS, css } from '../layers'
import type { DemoData } from '../demo/types'
import { clamp, ease, type Film } from '../film/useScrollFilm'
import { MAX_UPLOAD_MB } from '../config'

/**
 * Counted-up figures. Sources:
 * - 114 ReMIND patients: scripts/analysis/download_remind.py (all 114 Preop patients).
 * - 3929 slices: data/archive/kaggle_3m holds 3929 image/mask pairs.
 * - 663 series / 10.9 GB: from the prototype; not verifiable from the repo (the download manifest isn't checked in).
 * - 4B, Apache-2.0: FLUX.2 klein base-4B (scripts/lora/train_klein4b_edit.yaml).
 * - 300 MB / 5 min: contract C3.
 */
const FIGURES: { count: number; suffix?: string; d: string }[] = [
  { count: 0, d: 'numbers written by AI' },
  { count: 114, d: 'ReMIND patients, surgeon-reviewed outlines · TCIA' },
  { count: 663, d: 'pre-op MRI and label series, 10.9 GB' },
  { count: 3929, d: 'extra slices with checked lesion masks' },
  { count: 4, suffix: 'B', d: 'parameters, FLUX.2 klein base · Apache-2.0' },
  { count: N_LAYERS, suffix: 'layers', d: 'tissue layers, scalp to lesion' },
  { count: MAX_UPLOAD_MB, suffix: 'MB', d: 'largest upload' },
  { count: 5, suffix: 'min', d: 'max time in memory. Nothing hits disk' },
]

const show = (el: HTMLElement, n: number, suffix?: string) => {
  el.textContent = String(n)
  if (suffix) {
    const s = document.createElement('small')
    s.textContent = suffix
    el.appendChild(s)
  }
}

export function NumbersScene({ film, demo }: { film: Film; demo: DemoData }) {
  const barRef = useRef<HTMLDivElement>(null)
  const valEls = useRef<(HTMLDivElement | null)[]>([])

  // The counter text is owned here, not by React (it's rewritten every frame while counting).
  useEffect(() => {
    valEls.current.forEach((el, i) => el && show(el, FIGURES[i].count, FIGURES[i].suffix))
    return film.setHook(4, (_u, inn) => {
      const q = ease(clamp(inn * 1.15))
      barRef.current?.style.setProperty('--f', q.toFixed(3))
      valEls.current.forEach((el, i) => el && show(el, Math.round(FIGURES[i].count * q), FIGURES[i].suffix))
    })
  }, [film])

  const inside = LAYERS.filter((L) => L.id !== 'skull')
  const totalCm = (demo.metrics.intracranialMm2 / 100).toFixed(0)

  return (
    <section className="sc s4" data-name="Numbers" ref={film.bindScene(4)}>
      <div className="in">
        <div className="mono" data-d="0.4">
          Scene 5 · By the numbers
        </div>
        <h2 data-d="0.6">Measured on this slice. Tested on real scans.</h2>
        <div className="comp" data-d="0.8">
          <div className="bar" ref={barRef}>
            {inside.map((L) => (
              <i key={L.id} style={{ width: `${demo.metrics.layers[L.id].pct}%`, background: css(L.rgb) }} title={L.name} />
            ))}
          </div>
          <div className="key">
            {inside.map((L) => (
              <span key={L.id}>
                <i style={{ background: css(L.rgb) }} />
                {L.name} <b>{demo.metrics.layers[L.id].pct?.toFixed(1)}%</b>
              </span>
            ))}
            <span>
              Total <b>{totalCm} cm²</b> inside the skull
            </span>
          </div>
        </div>
        <div className="nums" data-d="1.0">
          {FIGURES.map((f, i) => (
            <div className="nm" key={i}>
              <div className="v" ref={(el) => void (valEls.current[i] = el)} />
              <div className="d">{f.d}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
