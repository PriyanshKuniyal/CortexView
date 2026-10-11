import { useEffect, useRef, useState } from 'react'
import type { DemoData } from '../demo/types'
import { clamp, type Film } from '../film/useScrollFilm'

function Tile({ src }: { src: ImageData | HTMLImageElement }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current!
    if (src instanceof ImageData) {
      c.width = src.width
      c.height = src.height
      c.getContext('2d')!.putImageData(src, 0, 0)
    } else {
      c.width = src.naturalWidth
      c.height = src.naturalHeight
      c.getContext('2d')!.drawImage(src, 0, 0)
    }
  }, [src])
  return <canvas ref={ref} />
}

const ordinal = (n: number) => {
  const s = n % 100 >= 11 && n % 100 <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th'
  return `${n}${s}`
}

export function SlicesScene({ film, demo }: { film: Film; demo: DemoData }) {
  const { tiles, sliceNumbers, key, step } = demo.sheet
  const tileEls = useRef<(HTMLDivElement | null)[]>([])
  const [pick, setPick] = useState(false)
  const pickRef = useRef(false)

  useEffect(
    () =>
      film.setHook(1, (u) => {
        const on = Math.round(clamp(u / 0.55) * tiles.length)
        tileEls.current.forEach((t, i) => t?.classList.toggle('on', i < on))
        const p = u > 0.68
        if (p !== pickRef.current) {
          pickRef.current = p
          setPick(p)
        }
      }),
    [film, tiles.length],
  )

  return (
    <section className="sc s1" data-name="The problem" ref={film.bindScene(1)}>
      <div className="in">
        <div className="s1head">
          <div>
            <div className="mono" data-d="0.4">
              Scene 2 · The problem
            </div>
            <h2 data-d="0.6">{demo.nSlices} grey slices. One scan.</h2>
          </div>
          <p className="lede" data-d="0.8">
            Radiologists train for years to read these. Everyone else, including the patient, sees fog.
          </p>
        </div>
        <div className={'sheet' + (pick ? ' pick' : '')} data-d="1">
          {tiles.map((src, t) => (
            <div key={t} className={'tile' + (t === key ? ' key' : '')} ref={(el) => void (tileEls.current[t] = el)}>
              <Tile src={src} />
              <span>{sliceNumbers[t]}</span>
            </div>
          ))}
        </div>
        <div className="s1foot mono" data-d="1.1">
          <span>Axial · inferior → superior</span>
          <span>{pick ? `CortexView works on SL ${demo.sliceIndex}` : step === 1 ? 'Every slice shown' : `Every ${ordinal(step)} slice shown`}</span>
        </div>
      </div>
    </section>
  )
}
