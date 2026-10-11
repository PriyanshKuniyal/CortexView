import { useCallback, useEffect, useRef, useState } from 'react'
import type { DemoData } from '../demo/types'
import { clamp, ease, type Film } from '../film/useScrollFilm'
import { spacingLabel } from '../format'

const sideLabel = (x: number) => (x > 66 ? 'Raw' : x < 34 ? 'CortexView' : 'Raw · CortexView')

export function BeforeAfterScene({ film, demo }: { film: Film; demo: DemoData }) {
  const baRef = useRef<HTMLDivElement>(null)
  const rangeRef = useRef<HTMLInputElement>(null)
  const manual = useRef(false)
  const [label, setLabel] = useState(sideLabel(90))
  const labelRef = useRef(label)

  const setBA = useCallback((x: number) => {
    baRef.current?.style.setProperty('--x', x + '%')
    if (rangeRef.current) rangeRef.current.value = String(x)
    const l = sideLabel(x)
    if (l !== labelRef.current) {
      labelRef.current = l
      setLabel(l)
    }
  }, [])

  useEffect(
    () =>
      film.setHook(2, (u) => {
        if (!manual.current) setBA(+(92 - 84 * ease(clamp(u * 1.1))).toFixed(2))
      }),
    [film, setBA],
  )
  // dragging the divider takes over until you leave the scene
  useEffect(() => {
    if (film.active !== 2) manual.current = false
  }, [film.active])

  return (
    <section className="sc s2" data-name="Before / after" ref={film.bindScene(2)}>
      <div className="in split">
        <div>
          <div className="mono" data-d="0.4">
            Scene 3 · Before and after
          </div>
          <h2 data-d="0.7">Same slice. Now readable.</h2>
          <p className="lede" data-d="1">
            Scroll, or drag across the scan.
          </p>
          <div className="facts" data-d="1.3">
            <div>
              <b>&lt;1 s</b>
              <span>find and measure five layers</span>
            </div>
            <div>
              <b>~10 s</b>
              <span>illustration arrives</span>
            </div>
            <div>
              <b>1:1</b>
              <span>colour only on real tissue</span>
            </div>
          </div>
        </div>
        <div className="frame ba" ref={baRef} data-d="0.5">
          <img src={demo.images.mriBlack} alt="Raw MRI slice" />
          <img src={demo.images.illus} className="top" alt="The same slice as illustrated layers" />
          <div className="tick" />
          <div className="ov tl">
            {demo.caseLabel}
            <br />
            {demo.sequenceLabel}
          </div>
          <div className="ov tr">{label}</div>
          <div className="ov bl">
            SL {demo.sliceIndex} / {demo.nSlices}
            <br />
            {spacingLabel([demo.spacingMm, demo.spacingMm])}
          </div>
          <div className="ov br">
            {demo.sourceNote[0]}
            <br />
            {demo.sourceNote[1]}
          </div>
          <div className="bar" />
          <div className="knob">◂ ▸</div>
          <input
            type="range"
            ref={rangeRef}
            min={0}
            max={100}
            defaultValue={90}
            aria-label="Move the before and after divider"
            onInput={(e) => {
              manual.current = true
              setBA(+e.currentTarget.value)
            }}
          />
        </div>
      </div>
    </section>
  )
}
