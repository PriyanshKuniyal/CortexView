import { useEffect, useRef, useState } from 'react'
import { LAYERS, N_LAYERS, css, type Layer } from '../layers'
import type { DemoData } from '../demo/types'
import { clamp, ease, type Film } from '../film/useScrollFilm'
import { pad2 } from '../format'

/** The card sentence for one layer. Numbers come from the demo metrics, formatted only. */
function sentence(L: Layer, demo: DemoData) {
  const m = demo.metrics.layers[L.id]
  const cm2 = (m.areaMm2 / 100).toFixed(1)
  const pct = m.pct?.toFixed(1)
  switch (L.id) {
    case 'skull':
      return `Skull and scalp: about ${cm2} cm² of this slice.`
    case 'csf':
      return `Fluid: about ${cm2} cm², ${pct}% of the space inside the skull.`
    case 'lesion':
      return `About ${cm2} cm², ${pct}% of the space inside the skull.` + (m.maxDiameterMm != null ? ` Longest span about ${m.maxDiameterMm.toFixed(0)} mm.` : '')
    default:
      return `${L.name}: about ${cm2} cm², ${pct}% of the space inside the skull.`
  }
}

export function PeelScene({ film, demo }: { film: Film; demo: DemoData }) {
  const imgEls = useRef<(HTMLImageElement | null)[]>([])
  const barEls = useRef<(HTMLSpanElement | null)[]>([])
  const [card, setCard] = useState(0)
  const cardRef = useRef(0)

  useEffect(
    () =>
      film.setHook(3, (u) => {
        const steps = N_LAYERS + 1
        LAYERS.forEach((L, i) => {
          const im = imgEls.current[i]
          if (!im) return
          const off = ease(clamp(u * steps - L.peel_order))
          im.style.opacity = (1 - off).toFixed(3)
          im.style.transform = `translate3d(0,${(-off * 14).toFixed(2)}%,0) scale(${(1 + off * 0.22).toFixed(3)})`
          im.style.filter = off > 0 && off < 1 && !film.reduce ? `blur(${(off * 5).toFixed(1)}px)` : 'none'
        })
        const c = clamp(Math.floor(u * steps), 0, N_LAYERS)
        if (c !== cardRef.current) {
          cardRef.current = c
          setCard(c)
        }
        barEls.current.forEach((s, i) => s?.style.setProperty('--f', clamp(u * steps - (i + 1)).toFixed(3)))
      }),
    [film],
  )

  const totalCm = (demo.metrics.intracranialMm2 / 100).toFixed(0)
  const cards = [
    {
      lab: `All ${N_LAYERS} layers`,
      title: 'The whole slice',
      c: null as string | null,
      desc: 'Scroll and each layer lifts away, scalp first.',
      num: `${totalCm} cm² inside the skull, counted pixel by pixel.`,
    },
    ...LAYERS.map((L) => ({
      lab: `Layer ${pad2(L.peel_order)} / ${pad2(N_LAYERS)}`,
      title: L.name,
      c: css(L.rgb),
      desc: L.description,
      num: sentence(L, demo),
    })),
  ]

  return (
    <section className="sc s3" data-name="Peel" ref={film.bindScene(3)}>
      <div className="in split">
        <div className="frame peel" data-d="0.35">
          <img alt="" src={demo.images.mriBlack} />
          {LAYERS.map((L, i) => (
            <img key={L.id} ref={(el) => void (imgEls.current[i] = el)} alt={L.name} src={demo.images.layer[L.id]} />
          ))}
          <div className="tick" />
          <div className="ov tl">
            {demo.caseLabel}
            <br />
            Peel
          </div>
          <div className="ov tr">
            {N_LAYERS - card} / {N_LAYERS} layers
          </div>
          <div className="ov bl">
            SL {demo.sliceIndex} / {demo.nSlices}
          </div>
          <div className="ov br">Outside in</div>
        </div>
        <div>
          <div className="mono" data-d="0.5">
            Scene 4 · Peel it, outside in
          </div>
          <div className="pcard" data-d="0.8">
            {cards.map((c, i) => (
              <div key={i} className={i === card ? '' : 'off'}>
                <div className="mono">{c.lab}</div>
                <h3>
                  {c.c && <i style={{ background: c.c }} />}
                  {c.title}
                </h3>
                <p className="lede">{c.desc}</p>
                <p className="num">{c.num}</p>
              </div>
            ))}
          </div>
          <div className="pbar" data-d="1.1">
            {LAYERS.map((L, i) => (
              <span key={L.id} ref={(el) => void (barEls.current[i] = el)} style={{ '--c': css(L.rgb) } as React.CSSProperties} />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
