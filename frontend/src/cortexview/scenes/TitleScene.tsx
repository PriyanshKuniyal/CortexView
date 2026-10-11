import { useEffect, useRef } from 'react'
import { LAYERS, css } from '../layers'
import type { DemoData } from '../demo/types'
import type { Film } from '../film/useScrollFilm'
import { numberWord } from '../format'

export function TitleScene({ film, demo }: { film: Film; demo: DemoData }) {
  const stageRef = useRef<HTMLDivElement>(null)
  const stackRef = useRef<HTMLDivElement>(null)
  const tilt = useRef({ mx: 0, my: 0, tmx: 0, tmy: 0 }) // pointer tilt, eased per frame

  useEffect(
    () =>
      film.setHook(0, (u, _inn, out) => {
        const m = tilt.current
        const stack = stackRef.current
        if (!stack) return
        m.tmx += (m.mx - m.tmx) * 0.08
        m.tmy += (m.my - m.tmy) * 0.08
        stack.style.setProperty('--gap', 34 + u * 22 + out * 120 + 'px')
        stack.style.setProperty('--rx', 58 - m.tmy * 10 - out * 14 + 'deg')
        stack.style.setProperty('--rz', -34 + m.tmx * 14 + u * 8 + out * 20 + 'deg')
      }),
    [film],
  )

  // base MRI at the bottom, then layers inside out (lesion lowest, skull on top)
  const planes = [['base', demo.images.mri], ...[...LAYERS].reverse().map((L) => [L.id, demo.images.layer[L.id]])]

  return (
    <section className="sc s0" data-name="Title" ref={film.bindScene(0)}>
      <div className="in split">
        <div>
          <div className="mono" data-d="0.4">
            One brain slice, {numberWord(LAYERS.length)} layers
          </div>
          <h1 data-d="0.8">
            <span className="em">Generative</span> where it helps. Deterministic where it matters.
          </h1>
          <p className="lede" data-d="1.1">
            A brain MRI, turned into an illustration you can take apart. Every number is measured from the pixels.
          </p>
          <div data-d="1.3">
            <button className="cta" type="button" onClick={() => film.go(1)}>
              Start the film ↓
            </button>
          </div>
          <div className="chips" data-d="1.5">
            <span>
              <i style={{ background: '#777' }} />
              MRI
            </span>
            {LAYERS.map((L) => (
              <span key={L.id}>
                <i style={{ background: css(L.rgb) }} />
                {L.name}
              </span>
            ))}
          </div>
        </div>
        <div
          className="stage"
          data-d="0.3"
          ref={stageRef}
          onPointerMove={(e) => {
            const r = stageRef.current!.getBoundingClientRect()
            tilt.current.mx = (e.clientX - r.left) / r.width - 0.5
            tilt.current.my = (e.clientY - r.top) / r.height - 0.5
          }}
          onPointerLeave={() => {
            tilt.current.mx = tilt.current.my = 0
          }}
        >
          <div className="stack" ref={stackRef}>
            {planes.map(([id, src], i) => (
              <div key={id} className={'plane' + (id === 'base' ? ' base' : '')} style={{ '--i': i } as React.CSSProperties}>
                <img alt="" src={src} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
