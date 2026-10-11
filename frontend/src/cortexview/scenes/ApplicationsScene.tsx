import { useEffect, useRef, useState } from 'react'
import { LAYERS, css, type LayerId } from '../layers'
import type { DemoData } from '../demo/types'
import { clamp, type Film } from '../film/useScrollFilm'

/**
 * Scene 8 · Applications. Five places CortexView changes the conversation, stepped through by scroll (hover
 * overrides). The right side reuses the title scene's floating layer stack (.stage/.stack/.plane) and re-poses
 * it per application. Each place says honestly whether it is live in this demo or on the roadmap.
 */
type Pose = { gap: number; rx: number; rz: number; focus: LayerId | 'all' | 'none'; lift?: LayerId }

interface App {
  where: string
  who: string
  title: string
  body: string
  live: boolean
  pose: Pose
  callout: (d: DemoData) => string
}

const APPS: App[] = [
  {
    where: 'The consult room',
    who: 'Patients · neurologists',
    title: 'Show it, don’t describe it.',
    body: 'The doctor peels the scan in front of the patient. The lesion lifts out with its size in plain words: “about the width of a golf ball”.',
    live: true,
    pose: { gap: 40, rx: 56, rz: -30, focus: 'lesion', lift: 'lesion' },
    callout: (d) => {
      const l = d.metrics.layers.lesion
      return `Lesion · ${(l.areaMm2 / 100).toFixed(1)} cm²${l.maxDiameterMm != null ? ` · ${l.maxDiameterMm.toFixed(0)} mm across` : ''}`
    },
  },
  {
    where: 'The classroom',
    who: 'Medical students · teachers',
    title: 'Anatomy from real brains, not diagrams.',
    body: 'Every layer from scalp to lesion comes apart, labelled and measured, over the real MRI it was traced from.',
    live: true,
    pose: { gap: 62, rx: 60, rz: -38, focus: 'all' },
    callout: () => '5 layers · scalp → lesion',
  },
  {
    where: 'The tumor board',
    who: 'Surgeons · oncologists · radiologists',
    title: 'One picture the whole team trusts.',
    body: 'Load the radiologist’s own outline (DICOM-SEG) and everyone discusses the same shape, with numbers counted the same way every time.',
    live: true,
    pose: { gap: 34, rx: 50, rz: -24, focus: 'lesion' },
    callout: () => 'Doctor’s outline · imported',
  },
  {
    where: 'Every follow-up visit',
    who: 'Patients on surveillance',
    title: '“Is it growing?” answered visually.',
    body: 'Deterministic measurements make visit-to-visit volumes comparable. A side-by-side timeline of the lesion is the next build.',
    live: false,
    pose: { gap: 26, rx: 64, rz: -44, focus: 'lesion', lift: 'lesion' },
    callout: (d) => `Visit 1 · ${(d.metrics.layers.lesion.areaMm2 / 100).toFixed(1)} cm² on this slice`,
  },
  {
    where: 'Clinics without a radiologist nearby',
    who: 'Rural care · telehealth',
    title: 'A readable scan, anywhere.',
    body: 'DICOM in, nothing stored, runs on a laptop with a flat-colour fallback when offline. A patient-friendly view to share remotely is on the roadmap.',
    live: false,
    pose: { gap: 14, rx: 48, rz: -20, focus: 'none' },
    callout: () => 'Laptop · in memory · offline fallback',
  },
]

export function ApplicationsScene({ film, demo }: { film: Film; demo: DemoData }) {
  const [scrollIdx, setScrollIdx] = useState(0)
  const [hover, setHover] = useState<number | null>(null)
  const last = useRef(0)

  useEffect(
    () =>
      film.setHook(7, (u) => {
        const i = clamp(Math.floor(u * APPS.length), 0, APPS.length - 1)
        if (i !== last.current) {
          last.current = i
          setScrollIdx(i)
        }
      }),
    [film],
  )

  const active = hover ?? scrollIdx
  const app = APPS[active]
  const p = app.pose
  // base MRI at the bottom, then the layers inside out (lesion lowest, skull on top), as in the title scene
  const planes: [string, string, LayerId | null][] = [
    ['base', demo.images.mri, null],
    ...[...LAYERS].reverse().map((L) => [L.id, demo.images.layer[L.id], L.id] as [string, string, LayerId]),
  ]

  return (
    <section className="sc s7" data-name="Applications" ref={film.bindScene(7)}>
      <div className="in split">
        <div className="apx-copy">
          <div className="mono" data-d="0.3">
            Scene 8 · Applications
          </div>
          <div className="apx-eyebrow" data-d="0.45">
            Applications
          </div>
          <h2 data-d="0.6">
            One scan. <span className="em">Five rooms it changes.</span>
          </h2>
          <ol className="apx-list" data-d="0.8" onMouseLeave={() => setHover(null)}>
            {APPS.map((a, i) => (
              <li key={a.where} className={i === active ? 'on' : ''}>
                <button type="button" onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onClick={() => setHover(i)}>
                  <span className="apx-n mono">{String(i + 1).padStart(2, '0')}</span>
                  <span className="apx-tx">
                    <span className="apx-where">{a.where}</span>
                    <span className="apx-who mono">{a.who}</span>
                  </span>
                  <span className={'apx-st mono' + (a.live ? ' live' : '')}>{a.live ? 'Live in this demo' : 'Roadmap'}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="apx-detail" key={active}>
            <h3>{app.title}</h3>
            <p>{app.body}</p>
          </div>
        </div>

        <div className="stage apx-stage" data-d="0.5">
          <div
            className="stack"
            style={{ '--gap': `${p.gap}px`, '--rx': `${p.rx}deg`, '--rz': `${p.rz}deg` } as React.CSSProperties}
          >
            {planes.map(([key, src, id], i) => {
              const dim = id != null && p.focus !== 'all' && p.focus !== 'none' && p.focus !== id
              const lifted = id != null && p.lift === id
              return (
                <div
                  key={key}
                  className={'plane' + (key === 'base' ? ' base' : '') + (dim ? ' apx-dim' : '') + (lifted ? ' apx-lift' : '')}
                  style={{ '--i': i } as React.CSSProperties}
                >
                  <img alt="" src={src} />
                </div>
              )
            })}
          </div>
          <div className="apx-callout mono" key={`c${active}`}>
            <i style={{ background: app.pose.focus === 'lesion' ? css(LAYERS.find((l) => l.id === 'lesion')!.rgb) : 'var(--ink)' }} />
            {app.callout(demo)}
          </div>
          <div className="apx-count mono">
            {String(active + 1).padStart(2, '0')} / {String(APPS.length).padStart(2, '0')} · {app.where}
          </div>
        </div>
      </div>
    </section>
  )
}
